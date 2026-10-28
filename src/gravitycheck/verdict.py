"""Combine clocks into a verdict a non-expert can read."""

from dataclasses import dataclass
from itertools import combinations

import numpy as np

from .clocks import Clock

# Placeholder mapping from clock disagreement (in standard deviations) to the AI score.
# Fit both on the labelled benchmark before quoting the score as a probability.
SCORE_MIDPOINT_SIGMA = 3.0
SCORE_SLOPE = 1.5


@dataclass
class Verdict:
    label: str  # normal_speed | slow_motion | fast_motion | inconsistent | insufficient
    k: float | None
    tension: float  # > 1 means the clocks cannot all be right; how far apart they are
    clocks: list[Clock]
    explanation: str
    ai_score: float | None = None  # 0-100; None when there are not two clocks to compare
    gap_sigma: float = 0.0  # largest disagreement between two clocks, in standard deviations


def _log_normal(c: Clock) -> tuple[float, float]:
    """Treat a clock's interval as about +-2 sigma in log k."""
    return float(np.log(c.k)), max((np.log(c.hi) - np.log(c.lo)) / 4, 1e-3)


def disagreement(clocks: list[Clock]) -> tuple[float, tuple[Clock, Clock] | None]:
    """Largest pairwise gap in log k, measured in combined standard deviations.

    Ratios, not differences: 4x slow motion moves every clock by the same factor,
    so it leaves every pairwise gap at zero.
    """
    best, pair = 0.0, None
    for a, b in combinations(clocks, 2):
        (ma, sa), (mb, sb) = _log_normal(a), _log_normal(b)
        gap = abs(ma - mb) / np.hypot(sa, sb)
        if gap > best:
            best, pair = float(gap), (a, b)
    return best, pair


def ai_score(gap_sigma: float) -> float:
    """Map clock disagreement to 0-100. Uncalibrated until fitted on the benchmark."""
    return float(100 / (1 + np.exp(-SCORE_SLOPE * (gap_sigma - SCORE_MIDPOINT_SIGMA))))


def judge(clocks: list[Clock | None]) -> Verdict:
    clocks = [c for c in clocks if c is not None]
    if not clocks:
        return Verdict("insufficient", None, 0.0, [], "No usable physical clock found in this clip.")
    if len(clocks) == 1:
        c = clocks[0]
        what = "real time" if c.lo <= 1 <= c.hi else f"{c.k:.2f}x speed"
        return Verdict("insufficient", c.k, 0.0, clocks,
                       f"Only one clock ({c.name}) reads {what}. A second, independent clock is needed "
                       "to tell slow motion from a generated clip.")
    gap, _ = disagreement(clocks)
    score = ai_score(gap)
    lo, hi = max(c.lo for c in clocks), min(c.hi for c in clocks)
    if lo <= hi:
        k = float(np.sqrt(lo * hi))
        if lo <= 1 <= hi:
            return Verdict("normal_speed", 1.0, lo / hi, clocks, "All clocks agree with real time.", score, gap)
        label = "slow_motion" if hi < 1 else "fast_motion"
        return Verdict(label, k, lo / hi, clocks,
                       f"All clocks agree, at {k:.2f}x speed: consistent with real footage played "
                       f"{'slower' if hi < 1 else 'faster'} than real time.", score, gap)
    fast, slow = max(clocks, key=lambda c: c.lo), min(clocks, key=lambda c: c.hi)
    tension = fast.lo / slow.hi
    return Verdict("inconsistent", None, tension, clocks,
                   f"The clocks disagree: {slow.name} runs at {slow.k:.2f}x while {fast.name} runs at "
                   f"{fast.k:.2f}x, at least {tension:.1f}x apart. Real footage, even slow motion, "
                   "keeps one clock for everything; this clip does not.", score, gap)
