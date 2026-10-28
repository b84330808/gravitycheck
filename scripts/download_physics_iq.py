"""Download the real videos of the Physics-IQ benchmark (CC-BY 4.0, Google DeepMind).

Only the 30 FPS, take-1 full videos are fetched (198 clips, about 1.8 GB). They come
from the public bucket gs://physics-iq-benchmark, so no account is needed.

Usage: uv run scripts/download_physics_iq.py ../data/physics-iq
"""

import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

BUCKET = "physics-iq-benchmark"
PREFIX = "full-videos/take-1/30FPS/"


def list_objects(prefix: str) -> list[dict]:
    items, token = [], None
    while True:
        query = {"prefix": prefix, "maxResults": "1000", "fields": "items(name,size),nextPageToken"}
        if token:
            query["pageToken"] = token
        url = f"https://storage.googleapis.com/storage/v1/b/{BUCKET}/o?" + urllib.parse.urlencode(query)
        with urllib.request.urlopen(url) as resp:
            page = json.load(resp)
        items += [it for it in page.get("items", []) if it["name"].endswith(".mp4")]
        token = page.get("nextPageToken")
        if not token:
            return items


def main() -> None:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "data/physics-iq")
    out.mkdir(parents=True, exist_ok=True)
    items = list_objects(PREFIX)
    print(f"{len(items)} videos, {sum(int(i['size']) for i in items) / 1e9:.2f} GB -> {out}")
    for n, item in enumerate(items, 1):
        dest = out / Path(item["name"]).name
        if dest.exists() and dest.stat().st_size == int(item["size"]):
            continue
        url = f"https://storage.googleapis.com/{BUCKET}/" + urllib.parse.quote(item["name"])
        tmp = dest.with_suffix(".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(dest)
        print(f"[{n}/{len(items)}] {dest.name}", flush=True)
    print("done")


if __name__ == "__main__":
    main()
