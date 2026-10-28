"""Draw the evidence onto the video: tracks, a real-gravity ghost, and the clock panel."""

import cv2
import numpy as np

from .analyze import Analysis, ProbeResult
from .clocks import G

GREEN, ORANGE, BLUE, WHITE, GREY, RED = (60, 220, 90), (255, 160, 40), (80, 160, 255), (240, 240, 240), (120, 120, 120), (235, 60, 60)
PANEL_H = 170
LOG_MIN, LOG_MAX = np.log(0.1), np.log(3.0)


def _reference_k(analysis: Analysis, skip: ProbeResult) -> float:
    """Speed the ghost should follow: what the other clocks say, or real time."""
    others = [r.clock.k for r in analysis.results if r is not skip and r.clock]
    return float(np.median(others)) if others else 1.0


def _ghost_paths(analysis: Analysis, r: ProbeResult) -> list[tuple[np.ndarray, np.ndarray]]:
    """For each flight: where the object would be if it fell with real gravity."""
    if not r.clock or not r.probe.size_m:
        return []
    px_per_m = r.clock.detail["size_px"] / np.sqrt(r.probe.size_m[0] * r.probe.size_m[1])
    g_px = G * _reference_k(analysis, r) ** 2 * px_per_m
    t, y, v = r.track.times, r.track.cy, r.track.valid
    paths = []
    for arc in r.arcs:
        m = v & (t >= arc.t0 - 1e-6) & (t <= arc.t1 + 1e-6)
        c = np.polyfit(t[m] - arc.t0, y[m], 2)  # measured start position and speed
        tt = t[(t >= arc.t0) & (t <= arc.t1 + 0.5 * (arc.t1 - arc.t0))]
        dt = tt - arc.t0
        paths.append((tt, c[2] + c[1] * dt + 0.5 * g_px * dt**2))
    return paths


def _panel(width: int, analysis: Analysis) -> np.ndarray:
    p = np.full((PANEL_H, width, 3), 25, np.uint8)
    v = analysis.verdict
    score = "--" if v.ai_score is None else f"{v.ai_score:.0f}"
    color = GREY if v.ai_score is None else (RED if v.ai_score >= 70 else ORANGE if v.ai_score >= 30 else GREEN)
    cv2.putText(p, "AI score", (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 1, cv2.LINE_AA)
    cv2.putText(p, score, (16, 100), cv2.FONT_HERSHEY_DUPLEX, 2.2, color, 3, cv2.LINE_AA)
    cv2.putText(p, v.label.replace("_", " "), (16, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)
    # Clock bars on a log axis of playback speed.
    x0, x1 = 190, width - 20
    to_x = lambda k: int(x0 + (np.log(np.clip(k, 0.1, 3.0)) - LOG_MIN) / (LOG_MAX - LOG_MIN) * (x1 - x0))
    for k in (0.25, 0.5, 1.0, 2.0):
        cv2.line(p, (to_x(k), 14), (to_x(k), PANEL_H - 24), (70, 70, 70) if k != 1 else (110, 110, 110), 1)
        cv2.putText(p, f"{k:g}x", (to_x(k) - 12, PANEL_H - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, GREY, 1, cv2.LINE_AA)
    clocks = [c for c in v.clocks]
    for i, c in enumerate(clocks):
        y = 34 + i * 36
        col = GREEN if c.name == "gait" else BLUE
        cv2.line(p, (to_x(c.lo), y), (to_x(c.hi), y), col, 6, cv2.LINE_AA)
        cv2.circle(p, (to_x(c.k), y), 7, WHITE, -1, cv2.LINE_AA)
        cv2.putText(p, f"{c.name} clock {c.k:.2f}x", (x0, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
    return p


def render(analysis: Analysis, out_path: str) -> None:
    from .video import write_video

    clip = analysis.clip
    ghosts = {id(r): _ghost_paths(analysis, r) for r in analysis.results}
    panel = _panel(clip.size[0], analysis)
    frames = []
    for i, (img, t) in enumerate(zip(clip.frames, clip.times)):
        img = img.copy()
        for r in analysis.results:
            tr = r.track
            if tr is None or not tr.valid[i]:
                continue
            if r.probe.kind == "fall":
                past = tr.valid & (tr.times <= t) & (tr.times > t - 1.0)
                pts = np.stack([tr.cx[past], tr.cy[past]], 1).astype(np.int32)
                cv2.polylines(img, [pts], False, BLUE, 2, cv2.LINE_AA)
                rad = int(tr.width[i] / 2) + 3
                cv2.circle(img, (int(tr.cx[i]), int(tr.cy[i])), rad, BLUE, 2, cv2.LINE_AA)
                for tt, yy in ghosts[id(r)]:
                    j = np.searchsorted(tt, t)
                    if j < len(tt) and abs(tt[j] - t) < 1e-3 and yy[j] < img.shape[0]:
                        gx, gy = int(tr.cx[i]), int(yy[j])
                        cv2.circle(img, (gx, gy), rad, ORANGE, 2, cv2.LINE_AA)
                        cv2.putText(img, "real gravity", (gx + rad + 4, gy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, ORANGE, 1, cv2.LINE_AA)
            elif r.clock:  # only show a person when they gave a usable clock
                w, h = tr.width[i], tr.height[i]
                cv2.rectangle(img, (int(tr.cx[i] - w / 2), int(tr.cy[i] - h / 2)), (int(tr.cx[i] + w / 2), int(tr.cy[i] + h / 2)), GREEN, 2)
                cv2.putText(img, f"{r.clock.detail['freq_hz']:.1f} steps/s", (int(tr.cx[i] - w / 2), int(tr.cy[i] - h / 2) - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREEN, 1, cv2.LINE_AA)
        frames.append(np.vstack([img, panel]))
    write_video(out_path, frames, fps=clip.nominal_fps, times=clip.times)
