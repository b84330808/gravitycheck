"""Per-frame object measurements.

`Track` is what every probe consumes. In production the masks come from SAM 3.1;
`track_color` is a stand-in for synthetic scenes with flat-coloured objects.
"""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Track:
    times: np.ndarray  # seconds
    cx: np.ndarray  # centroid, pixels (NaN where the object is not visible)
    cy: np.ndarray  # pixels, image coordinates: y grows downward
    width: np.ndarray  # bounding-box size, pixels
    height: np.ndarray

    @property
    def valid(self) -> np.ndarray:
        return ~np.isnan(self.cy)

    def subset(self, mask: np.ndarray) -> "Track":
        return Track(self.times[mask], self.cx[mask], self.cy[mask], self.width[mask], self.height[mask])


def load_tracks(path) -> dict[int, Track]:
    """Read tracks cached by scripts/track_clips.py."""
    data = np.load(path)
    ids = sorted({int(k.split("_")[0]) for k in data.files})
    return {i: Track(*(data[f"{i}_{f}"] for f in ("times", "cx", "cy", "width", "height"))) for i in ids}


def track_from_masks(times: np.ndarray, masks: list[np.ndarray | None]) -> Track:
    """Turn a boolean mask per frame (None when absent) into a Track."""
    rows = []
    for m in masks:
        if m is None or not m.any():
            rows.append((np.nan,) * 4)
            continue
        ys, xs = np.nonzero(m)
        rows.append((xs.mean(), ys.mean(), xs.max() - xs.min() + 1, ys.max() - ys.min() + 1))
    a = np.asarray(rows, dtype=float).reshape(-1, 4)
    return Track(np.asarray(times, float), a[:, 0], a[:, 1], a[:, 2], a[:, 3])


def track_motion(frames: list[np.ndarray], times: np.ndarray, thresh: int = 30, min_px: int = 40, max_jump: float = 0.15) -> Track:
    """Follow the main moving object against a static camera's median background.

    A stand-in for SAM 3.1 on lab footage. Each frame keeps the moving blob nearest
    the previous position (the largest one to start), so a second mover is ignored.
    """
    gray = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames]
    bg = np.median(np.stack(gray[:: max(1, len(gray) // 40)]), axis=0).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    jump = max_jump * max(gray[0].shape)
    masks, prev = [], None
    for g in gray:
        fg = cv2.morphologyEx((cv2.absdiff(g, bg) > thresh).astype(np.uint8), cv2.MORPH_OPEN, kernel)
        n, labels, stats, cents = cv2.connectedComponentsWithStats(fg)
        best, best_score = None, None
        for i in range(1, n):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < min_px:
                continue
            if prev is None:
                score = -area
            else:
                dist = np.hypot(*(cents[i] - prev))
                if dist > jump:
                    continue
                score = dist
            if best_score is None or score < best_score:
                best, best_score = i, score
        if best is None:
            masks.append(None)
            continue
        masks.append(labels == best)
        prev = cents[best]
    return track_from_masks(times, masks)


def track_color(frames: list[np.ndarray], times: np.ndarray, rgb: tuple[int, int, int], tol: int = 60, min_px: int = 30) -> Track:
    target = np.asarray(rgb, np.int16)
    masks = []
    for img in frames:
        m = np.abs(img.astype(np.int16) - target).sum(axis=2) < tol
        masks.append(m if m.sum() >= min_px else None)
    return track_from_masks(times, masks)
