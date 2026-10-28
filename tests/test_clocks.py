import numpy as np
import pytest

from gravitycheck.clocks import flight_arcs, gait_clock, gravity_clock
from gravitycheck.synth import BALL_RGB, WALKER_RGB, Scene, render
from gravitycheck.track import track_color
from gravitycheck.verdict import judge
from gravitycheck.video import read_video, write_video

BALL_SIZE = (0.20, 0.24)  # the synthetic ball is 0.22 m


def analyse(frames, times):
    ball = track_color(frames, times, BALL_RGB)
    walker = track_color(frames, times, WALKER_RGB)
    return judge([gravity_clock(ball, BALL_SIZE), gait_clock(walker)])


def test_real_time_clip_reads_normal_speed():
    v = analyse(*render(Scene()))
    assert v.label == "normal_speed", v.explanation
    gravity = v.clocks[0]
    assert gravity.detail["g_apparent_m_s2"] == pytest.approx(9.81, rel=0.08)


@pytest.mark.parametrize("speed", [0.5, 0.25])
def test_slow_motion_is_consistent_not_fake(speed):
    v = analyse(*render(Scene(speed=speed, duration=8 / speed ** 0.5)))
    assert v.label == "slow_motion", v.explanation
    assert v.k == pytest.approx(speed, rel=0.15)


def test_generated_style_weak_gravity_is_inconsistent():
    # Wan 5B's measured effective gravity: 1.81 m/s^2 (arXiv 2512.02016).
    v = analyse(*render(Scene(gravity_scale=1.81 / 9.81)))
    assert v.label == "inconsistent", v.explanation
    assert v.tension > 1.5


def test_ai_score_low_for_real_and_slow_motion_high_for_fake():
    real = analyse(*render(Scene()))
    slow = analyse(*render(Scene(speed=0.25, duration=16)))
    fake = analyse(*render(Scene(gravity_scale=1.81 / 9.81)))
    assert real.ai_score < 20 and slow.ai_score < 20
    assert fake.ai_score > 90


def test_ai_score_rises_with_how_wrong_gravity_is():
    scores = [analyse(*render(Scene(gravity_scale=s))).ai_score for s in (1.0, 0.8, 0.6, 0.4)]
    assert scores == sorted(scores)


def test_gravity_alone_cannot_separate_slow_motion_from_fake():
    """The ablation: a single gravity clock reads ~0.43 for both cases."""
    fake = gravity_clock(track_color(*render(Scene(gravity_scale=1.81 / 9.81)), BALL_RGB), BALL_SIZE)
    slow = gravity_clock(track_color(*render(Scene(speed=0.43, duration=12)), BALL_RGB), BALL_SIZE)
    assert fake.k == pytest.approx(slow.k, rel=0.15)
    assert judge([fake]).label == judge([slow]).label == "insufficient"


def test_variable_frame_rate_timestamps():
    rng = np.random.default_rng(1)
    dt = 1 / 30 * rng.uniform(0.6, 1.4, 240)  # phone-style jittery frame intervals
    times = np.r_[0, np.cumsum(dt)[:-1]]
    frames, times = render(Scene(times=times))
    arcs = flight_arcs(track_color(frames, times, BALL_RGB))
    assert arcs
    # Assuming a constant 30 fps here would bias every acceleration; real timestamps must not.
    v = analyse(frames, times)
    assert v.label == "normal_speed", v.explanation


def test_video_roundtrip_keeps_variable_timestamps(tmp_path):
    rng = np.random.default_rng(2)
    times = np.r_[0, np.cumsum(1 / 30 * rng.uniform(0.6, 1.4, 59))]
    frames, _ = render(Scene(times=times))
    path = tmp_path / "vfr.mp4"
    write_video(path, frames, fps=30, times=times)
    clip = read_video(path)
    assert len(clip.frames) == len(times)
    assert np.max(np.abs(clip.times - times)) < 2e-3
