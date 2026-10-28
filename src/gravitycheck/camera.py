"""Undo camera motion so tracked positions are measured against the scene, not the frame.

A handheld or panning camera adds its own motion to every object. Background
features (tracked objects masked out) give a similarity transform per frame pair;
chaining them maps every frame back to the first frame's coordinates.
"""

import cv2
import numpy as np

from .track import Track


def camera_transforms(frames: list[np.ndarray], exclude: list[Track] = (), pad: float = 0.25) -> list[np.ndarray]:
    """2x3 matrices taking each frame's pixels into frame 0's coordinates."""
    h, w = frames[0].shape[:2]
    to_first = [np.eye(2, 3)]
    prev = cv2.cvtColor(frames[0], cv2.COLOR_RGB2GRAY)
    for i in range(1, len(frames)):
        cur = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)
        mask = np.full((h, w), 255, np.uint8)
        for tr in exclude:
            for j in (i - 1, i):
                if tr.valid[j]:
                    bw, bh = tr.width[j] * (1 + pad), tr.height[j] * (1 + pad)
                    x0, y0 = int(tr.cx[j] - bw / 2), int(tr.cy[j] - bh / 2)
                    mask[max(0, y0):y0 + int(bh), max(0, x0):x0 + int(bw)] = 0
        pts = cv2.goodFeaturesToTrack(prev, 400, 0.01, 8, mask=mask)
        m = None
        if pts is not None and len(pts) >= 8:
            nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev, cur, pts, None)
            ok = st.ravel() == 1
            if ok.sum() >= 8:
                # cur -> prev, so that chaining maps back toward frame 0
                m, _ = cv2.estimateAffinePartial2D(nxt[ok], pts[ok], method=cv2.RANSAC, ransacReprojThreshold=2.0)
        if m is None:
            m = np.eye(2, 3)
        a = np.vstack([to_first[-1], [0, 0, 1]]) @ np.vstack([m, [0, 0, 1]])
        to_first.append(a[:2])
        prev = cur
    return to_first


def stabilize(track: Track, transforms: list[np.ndarray]) -> Track:
    """Express a track in frame-0 coordinates; sizes scale with the camera zoom."""
    cx, cy, w, h = track.cx.copy(), track.cy.copy(), track.width.copy(), track.height.copy()
    for i, a in enumerate(transforms):
        if np.isnan(cx[i]):
            continue
        cx[i], cy[i] = a @ np.array([cx[i], cy[i], 1.0])
        s = np.hypot(a[0, 0], a[1, 0])
        w[i], h[i] = w[i] * s, h[i] * s
    return Track(track.times, cx, cy, w, h)
