"""Read and write video while keeping each frame's real timestamp.

Phone footage is often variable frame rate, so nothing downstream may assume
`t = i / fps`. Every frame carries the time PyAV decodes from its pts.
"""

from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import av
import numpy as np


@dataclass
class Clip:
    frames: list[np.ndarray]  # RGB uint8, H x W x 3
    times: np.ndarray  # seconds since the first frame
    nominal_fps: float

    @property
    def size(self) -> tuple[int, int]:
        h, w = self.frames[0].shape[:2]
        return w, h


def read_video(path: str | Path, max_side: int = 960) -> Clip:
    """Decode a video, downscaling so the longer side is at most `max_side` pixels."""
    frames, times = [], []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        nominal = float(stream.average_rate or stream.guessed_rate or 30)
        scale = min(1.0, max_side / max(stream.codec_context.width, stream.codec_context.height))
        width = int(round(stream.codec_context.width * scale / 2)) * 2
        height = int(round(stream.codec_context.height * scale / 2)) * 2
        for frame in container.decode(stream):
            if frame.time is None:
                continue
            frames.append(frame.to_ndarray(format="rgb24", width=width, height=height))
            times.append(frame.time)
    t = np.asarray(times, dtype=float)
    order = np.argsort(t, kind="stable")  # decode order can differ from presentation order
    return Clip([frames[i] for i in order], t[order] - t[order][0], nominal)


def write_video(path: str | Path, frames: list[np.ndarray], fps: float = 30.0, times: np.ndarray | None = None) -> None:
    """Encode RGB frames to mp4. Passing `times` writes a variable-frame-rate file."""
    tb = Fraction(1, 90_000)
    h, w = frames[0].shape[:2]
    with av.open(str(path), "w") as container:
        try:
            stream = container.add_stream("libx264", rate=int(round(fps)))
            stream.options = {"crf": "18", "preset": "veryfast"}
        except (av.FFmpegError, ValueError):
            stream = container.add_stream("mpeg4", rate=int(round(fps)))
        stream.width, stream.height, stream.pix_fmt = w, h, "yuv420p"
        stream.codec_context.time_base = tb
        for i, img in enumerate(frames):
            frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(img), format="rgb24")
            t = times[i] if times is not None else i / fps
            frame.pts, frame.time_base = int(round(t / tb)), tb
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
