"""Text-prompted object tracking with SAM 3 running locally (open weights, facebook/sam3).

The Meta Model API's SAM 3.1 will slot in behind the same function; both return a
Track per object found for the prompt.
"""

from functools import lru_cache

import numpy as np
import torch

from .track import Track, track_from_masks
from .video import Clip

MODEL_ID = "facebook/sam3"


@lru_cache(maxsize=1)
def _load(image_size: int | None):
    from transformers import Sam3VideoConfig, Sam3VideoModel, Sam3VideoProcessor

    config = Sam3VideoConfig.from_pretrained(MODEL_ID)
    kwargs = {}
    if image_size:
        config.image_size = image_size
        kwargs["size"] = {"height": image_size, "width": image_size}
    model = Sam3VideoModel.from_pretrained(MODEL_ID, config=config, dtype=torch.bfloat16).to("cuda").eval()
    processor = Sam3VideoProcessor.from_pretrained(MODEL_ID, **kwargs)
    return model, processor


def sam3_tracks(clip: Clip, prompt: str, start_s: float = 0.0, end_s: float | None = None,
                image_size: int | None = None) -> dict[int, Track]:
    """Segment and track every instance of `prompt` between start_s and end_s."""
    model, processor = _load(image_size)
    sel = np.nonzero((clip.times >= start_s) & (clip.times <= (end_s if end_s is not None else np.inf)))[0]
    frames = [clip.frames[i] for i in sel]
    session = processor.init_video_session(video=frames, inference_device="cuda", processing_device="cpu",
                                           video_storage_device="cpu", dtype=torch.bfloat16)
    processor.add_text_prompt(session, prompt)
    per_obj: dict[int, list] = {}
    with torch.inference_mode():
        for out in model.propagate_in_video_iterator(inference_session=session):
            res = processor.postprocess_outputs(session, out)
            for oid, mask in zip(res["object_ids"].tolist(), res["masks"]):
                per_obj.setdefault(oid, [None] * len(frames))[out.frame_idx] = mask.cpu().numpy().astype(bool)
    times = clip.times[sel]
    return {oid: track_from_masks(times, masks) for oid, masks in per_obj.items()}


def main_track(tracks: dict[int, Track]) -> Track | None:
    """The object visible in the most frames; ties go to the one that moves most."""
    if not tracks:
        return None
    return max(tracks.values(), key=lambda t: (t.valid.sum(), np.nanmax(t.cy) - np.nanmin(t.cy) if t.valid.any() else 0))
