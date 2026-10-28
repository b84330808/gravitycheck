"""Render synthetic test scenes with known physics.

A red ball is dropped and bounces; a blue "walker" crosses the frame with a head bob
at its step cadence. Three knobs reproduce the cases the detector must separate:

* `speed`         playback speed; 0.25 means 4x slow motion. Both clocks slow down.
* `gravity_scale` multiplies g for the ball only, the way generators render
                  objects falling too slowly while people still move normally.
* `times`         custom frame timestamps, to mimic variable-frame-rate phone video.
"""

from dataclasses import dataclass, field

import cv2
import numpy as np

G = 9.81
BALL_RGB = (220, 30, 30)
WALKER_RGB = (30, 60, 220)


@dataclass
class Scene:
    duration: float = 8.0  # seconds of video
    fps: float = 30.0
    width: int = 640
    height: int = 480
    px_per_m: float = 180.0
    speed: float = 1.0
    gravity_scale: float = 1.0
    ball_diameter_m: float = 0.22
    drop_height_m: float = 1.6
    restitution: float = 0.7
    release_s: float = 0.5  # video time when the ball is let go
    cadence_hz: float = 1.8  # walker steps per second
    bob_m: float = 0.05  # head-bob amplitude
    walk_speed_mps: float = 0.6
    noise: float = 4.0
    seed: int = 0
    times: np.ndarray | None = field(default=None, repr=False)

    def frame_times(self) -> np.ndarray:
        if self.times is not None:
            return np.asarray(self.times, dtype=float)
        return np.arange(int(self.duration * self.fps)) / self.fps


def ball_height(t: float, h0: float, g: float, e: float, rest_speed: float = 0.15) -> float:
    """Height above the floor at physical time t after release, with bounces."""
    h, v, t0 = h0, 0.0, 0.0
    while True:
        tau = (v + np.sqrt(v * v + 2 * g * h)) / g  # time until the floor
        if t <= t0 + tau:
            dt = t - t0
            return h + v * dt - 0.5 * g * dt * dt
        t0, h, v = t0 + tau, 0.0, -e * (v - g * tau)
        if v < rest_speed:
            return 0.0


def render(scene: Scene) -> tuple[list[np.ndarray], np.ndarray]:
    rng = np.random.default_rng(scene.seed)
    times = scene.frame_times()
    floor_y = scene.height - 40
    r = scene.ball_diameter_m * scene.px_per_m / 2
    g = G * scene.gravity_scale
    base = np.full((scene.height, scene.width, 3), 170, np.uint8)
    cv2.line(base, (0, floor_y), (scene.width, floor_y), (90, 90, 90), 2)
    frames = []
    for t_video in times:
        img = base.copy()
        # Ball: the ball's own clock is physical time, scaled by playback speed.
        t_ball = max(0.0, (t_video - scene.release_s) * scene.speed)
        h = ball_height(t_ball, scene.drop_height_m, g, scene.restitution)
        cy = floor_y - r - h * scene.px_per_m
        cv2.circle(img, (int(scene.width * 0.3), int(round(cy))), int(r), BALL_RGB, -1, cv2.LINE_AA)
        # Walker: always real physics, slowed only by playback speed.
        t_walk = t_video * scene.speed
        x = 80 + scene.walk_speed_mps * t_walk * scene.px_per_m
        bob = scene.bob_m * scene.px_per_m * np.sin(2 * np.pi * scene.cadence_hz * t_walk)
        top = floor_y - 1.7 * scene.px_per_m + bob
        cv2.rectangle(img, (int(x), int(round(top))), (int(x + 0.4 * scene.px_per_m), int(round(top + 0.6 * scene.px_per_m))), WALKER_RGB, -1)
        noisy = img.astype(np.float32) + rng.normal(0, scene.noise, img.shape)
        frames.append(np.clip(noisy, 0, 255).astype(np.uint8))
    return frames, times
