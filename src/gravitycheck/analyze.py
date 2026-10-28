"""End to end: video in, clocks and verdict out.

A probe says what to look for and which law it obeys. For now probes are given by
hand; Muse Spark will propose them from the video.
"""

from dataclasses import dataclass, field

from .clocks import Arc, Clock, flight_arcs, gait_clock, gravity_clock
from .segment import main_track, sam3_tracks
from .track import Track
from .verdict import Verdict, judge
from .video import Clip, read_video


@dataclass
class Probe:
    prompt: str  # SAM noun phrase, e.g. "tennis ball"
    kind: str  # "fall" or "gait"
    size_m: tuple[float, float] | None = None  # real size range, for "fall"


@dataclass
class ProbeResult:
    probe: Probe
    track: Track | None
    clock: Clock | None
    arcs: list[Arc] = field(default_factory=list)


@dataclass
class Analysis:
    clip: Clip
    results: list[ProbeResult]
    verdict: Verdict


def run_probe(clip: Clip, probe: Probe) -> ProbeResult:
    track = main_track(sam3_tracks(clip, probe.prompt))
    if track is None:
        return ProbeResult(probe, None, None)
    if probe.kind == "fall":
        arcs = flight_arcs(track)
        return ProbeResult(probe, track, gravity_clock(track, probe.size_m, arcs), arcs)
    if probe.kind == "gait":
        return ProbeResult(probe, track, gait_clock(track))
    raise ValueError(f"unknown probe kind: {probe.kind}")


def analyze(path: str, probes: list[Probe], max_side: int = 960) -> Analysis:
    clip = read_video(path, max_side=max_side)
    results = [run_probe(clip, p) for p in probes]
    return Analysis(clip, results, judge([r.clock for r in results]))
