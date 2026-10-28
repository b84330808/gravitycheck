"""Physical clocks: each one turns a tracked motion into a playback-speed factor k.

k is how many physical seconds pass per second of video: 1 is real time, 0.25 is
4x slow motion. In a real clip every clock reads the same k, slow motion included.
A generated clip can show objects falling at a fraction of g while people walk at
a normal pace, so its clocks disagree.
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.signal import find_peaks, lombscargle

from .track import Track

G = 9.81
WALK_CADENCE_HZ = (1.6, 2.0)  # adult walking, steps per second


@dataclass
class Clock:
    name: str
    k: float
    lo: float
    hi: float
    detail: dict = field(default_factory=dict)

    def __str__(self) -> str:
        return f"{self.name}: k={self.k:.2f} [{self.lo:.2f}, {self.hi:.2f}]"


@dataclass
class Arc:
    """One ballistic flight between release, bounces and rest."""

    t0: float
    t1: float
    n: int
    accel_px: float  # downward acceleration, px/s^2
    accel_err: float
    rms_px: float


def _moving_runs(y: np.ndarray, min_step: float) -> list[tuple[int, int]]:
    step = np.abs(np.diff(y))
    moving = np.r_[False, step > min_step] | np.r_[step > min_step, False]
    runs, start = [], None
    for i, m in enumerate(moving):
        if m and start is None:
            start = i
        elif not m and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(y)))
    return runs


def despike(y: np.ndarray, min_jump: float) -> np.ndarray:
    """Mask single-frame glitches: a point far from both neighbours that agree with each other.

    SAM occasionally snaps to the wrong region for one frame. A real bounce is not
    removed, because the frames before and after an impact are far apart too.
    """
    keep = np.ones(len(y), bool)
    for i in range(1, len(y) - 1):
        jump = min(abs(y[i] - y[i - 1]), abs(y[i] - y[i + 1]))
        if jump > min_jump and jump > 5 * abs(y[i + 1] - y[i - 1]):
            keep[i] = False
    return keep


def flight_arcs(track: Track, min_points: int = 6, min_drop_px: float = 8.0, min_step_px: float = 0.8) -> list[Arc]:
    """Split a falling object's track at its bounces and fit y = a t^2 / 2 + ... to each flight."""
    v = track.valid
    t, y = track.times[v], track.cy[v]
    keep = despike(y, min_drop_px)
    t, y = t[keep], y[keep]
    if len(t) < min_points:
        return []
    impacts, _ = find_peaks(y, prominence=min_drop_px)  # lowest points on screen
    # A landing comes after a real drop; a wobble of an object still held is not one.
    impacts = impacts[[i >= 2 and y[i] - y[i - 2] > min_drop_px for i in impacts]].astype(int)
    bounds = [0, *impacts.tolist(), len(y)]
    arcs = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        # Drop the contact frame itself: the ball is squashed and its centre lags.
        seg = np.arange(a + (a > 0), b)
        if len(seg) < min_points:
            continue
        for r0, r1 in _moving_runs(y[seg], min_step_px):
            # Trim one frame at each end: at release the mask still merges with the holder,
            # at landing the object is squashed or sinks out of view. On a 30 fps clip of
            # a short drop these two frames alone biased g by about 20%.
            idx = seg[r0 + 1:r1 - 1]
            if len(idx) < min_points or np.ptp(y[idx]) < min_drop_px:
                continue
            tt = t[idx] - t[idx].mean()
            coef, cov = np.polyfit(tt, y[idx], 2, cov=True)
            resid = y[idx] - np.polyval(coef, tt)
            arcs.append(Arc(t[idx[0]], t[idx[-1]], len(idx), 2 * coef[0], 2 * np.sqrt(cov[0, 0]), float(np.sqrt(np.mean(resid**2)))))
    return arcs


def object_size_px(track: Track, round_object: bool = True) -> float:
    """Typical width of the object in pixels.

    Width, not height, because motion blur stretches a falling object vertically. For
    round objects, frames whose box is far from square are skipped: those are frames
    where the mask merged with something else or was cut off.
    """
    width = track.width[track.valid]
    if round_object:
        aspect = width / np.maximum(track.height[track.valid], 1)
        square = width[(aspect > 0.8) & (aspect < 1.25)]
        if len(square) >= 3:
            width = square
    return float(np.median(width))


def gravity_clock(track: Track, size_m: tuple[float, float], arcs: list[Arc] | None = None,
                  round_object: bool = True) -> Clock | None:
    """Read k from a falling object whose real size is known to lie in `size_m`.

    The object is its own ruler: its width in pixels gives the scale at its depth.
    """
    arcs = flight_arcs(track) if arcs is None else arcs
    arcs = [a for a in arcs if a.accel_px > 0]
    if not arcs:
        return None
    w = np.array([1 / a.accel_err**2 for a in arcs])
    accel = float(np.sum(w * [a.accel_px for a in arcs]) / w.sum())
    err = float(1 / np.sqrt(w.sum()))
    size_px = object_size_px(track, round_object)
    # a_m = a_px * size_m / size_px ; k = sqrt(a_m / G)
    to_k = lambda a_px, s_m: np.sqrt(max(a_px, 1e-9) * s_m / size_px / G)
    lo = to_k(accel - 2 * err, size_m[0])
    hi = to_k(accel + 2 * err, size_m[1])
    k = to_k(accel, float(np.sqrt(size_m[0] * size_m[1])))
    g_apparent = accel * float(np.sqrt(size_m[0] * size_m[1])) / size_px
    return Clock("gravity", float(k), float(lo), float(hi), {
        "accel_px_s2": accel, "accel_err": err, "size_px": size_px, "arcs": len(arcs),
        "g_apparent_m_s2": g_apparent,
    })


def periodic_clock(name: str, times: np.ndarray, signal: np.ndarray, normal_hz: tuple[float, float],
                   fmin: float = 0.1, fmax: float = 8.0, min_power: float = 0.3) -> Clock | None:
    """Read k from a rhythm with a known natural frequency range, e.g. walking cadence.

    Lomb-Scargle works on uneven timestamps, so variable frame rate needs no resampling.
    """
    ok = ~np.isnan(signal)
    t, s = np.asarray(times)[ok], np.asarray(signal)[ok]
    if len(t) < 12:
        return None
    s = s - np.polyval(np.polyfit(t, s, 2), t)  # remove drift and perspective change
    span = t[-1] - t[0]
    freqs = np.linspace(max(fmin, 1.5 / span), fmax, 4000)
    power = lombscargle(t, s, 2 * np.pi * freqs)
    power = power / (np.sum(s**2) / 2 + 1e-12)  # fraction of variance at each frequency
    i = int(np.argmax(power))
    if power[i] < min_power:
        return None
    f, df = freqs[i], 0.5 / span
    k = f / float(np.mean(normal_hz))
    return Clock(name, float(k), float((f - df) / normal_hz[1]), float((f + df) / normal_hz[0]),
                 {"freq_hz": float(f), "power": float(power[i]), "span_s": float(span)})


def gait_clock(track: Track) -> Clock | None:
    """Walking cadence from the vertical bob of a person's bounding box."""
    return periodic_clock("gait", track.times, track.cy, WALK_CADENCE_HZ)
