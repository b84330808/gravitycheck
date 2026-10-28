"""Run SAM 3 once per clip and cache the tracks, so the physics can be re-run cheaply.

Usage: uv run scripts/track_clips.py "tennis ball" 2.3 4.3 ../data/physics-iq/000[123]*.mp4
Writes ../data/tracks/<clip>__<prompt>.npz next to the data, one array set per object.
"""

import sys
from pathlib import Path

import numpy as np

from gravitycheck.segment import sam3_tracks
from gravitycheck.video import read_video


def main() -> None:
    prompt, start, end, *paths = sys.argv[1:]
    out_dir = Path(paths[0]).parent.parent / "tracks"
    out_dir.mkdir(exist_ok=True)
    for p in map(Path, paths):
        out = out_dir / f"{p.stem}__{prompt.replace(' ', '_')}.npz"
        if out.exists():
            continue
        tracks = sam3_tracks(read_video(p), prompt, float(start), float(end))
        arrays = {}
        for oid, tr in tracks.items():
            for field in ("times", "cx", "cy", "width", "height"):
                arrays[f"{oid}_{field}"] = getattr(tr, field)
        np.savez(out, **arrays)
        print(f"{p.name}: {len(tracks)} objects -> {out.name}", flush=True)


if __name__ == "__main__":
    main()
