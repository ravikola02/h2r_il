#!/usr/bin/env python
"""Convert a local LeRobot v2.1 dataset to v3.0 without touching the source.

The h2r_collection exports are v2.1 datasets that lack ``meta/episodes_stats.jsonl``
(they only carry the aggregated ``meta/stats.json``), while lerobot's official
``convert_dataset_v21_to_v30`` requires per-episode stats. This driver:

1. Stages a copy of the source dataset in a work directory.
2. Generates ``meta/episodes_stats.jsonl``: numeric features from the episode
   parquet files, video features from decoded subsampled frames (same sampling
   and per-channel/255 conventions as ``compute_episode_stats``).
3. Runs the upstream v2.1 -> v3.0 converter on the staged copy (local only).
4. Re-injects the custom ``h2r`` provenance block that ``DatasetInfo`` drops.
5. Moves the converted dataset to ``--dst``.

Usage:
  uv run python scripts/convert_v21_to_v30_local.py \
      --src ~/dev/h2r_collection/data/output/gear_left/lerobot_v2 \
      --dst ~/dev/h2r_collection/data/output/gear_left/lerobot_v3
"""

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import tqdm

from lerobot.datasets.compute_stats import (
    DEFAULT_QUANTILES,
    auto_downsample_height_width,
    compute_episode_stats,
    get_feature_stats,
    sample_indices,
)
from lerobot.datasets.video_utils import decode_video_frames
from lerobot.scripts.convert_dataset_v21_to_v30 import convert_dataset


def load_jsonlines(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def stats_to_jsonable(stats: dict) -> dict:
    return {
        key: {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in ft_stats.items()}
        for key, ft_stats in stats.items()
    }


def episode_video_stats(video_path: Path, num_frames: int, fps: float) -> dict:
    """Per-channel stats of an episode video, matching compute_episode_stats conventions."""
    idxs = sample_indices(num_frames)
    timestamps = [i / fps for i in idxs]
    frames = decode_video_frames(video_path, timestamps, tolerance_s=1 / fps, return_uint8=True)
    arr = np.stack([auto_downsample_height_width(f) for f in frames.numpy()])
    stats = get_feature_stats(arr, axis=(0, 2, 3), keepdims=True, quantile_list=DEFAULT_QUANTILES)
    return {k: v if k == "count" else np.squeeze(v / 255.0, axis=0) for k, v in stats.items()}


def generate_episodes_stats(root: Path) -> None:
    info = json.loads((root / "meta" / "info.json").read_text())
    fps = info["fps"]
    features = info["features"]
    video_keys = [k for k, ft in features.items() if ft["dtype"] == "video"]
    numeric_features = {k: ft for k, ft in features.items() if ft["dtype"] not in ("video", "image", "string")}
    episodes = sorted(load_jsonlines(root / "meta" / "episodes.jsonl"), key=lambda e: e["episode_index"])

    out_path = root / "meta" / "episodes_stats.jsonl"
    with open(out_path, "w") as f:
        for ep in tqdm.tqdm(episodes, desc="generate episodes_stats"):
            ep_idx = ep["episode_index"]
            chunk = ep_idx // info["chunks_size"]
            parquet_path = root / info["data_path"].format(episode_chunk=chunk, episode_index=ep_idx)
            df = pd.read_parquet(parquet_path)

            episode_data = {}
            for key in numeric_features:
                col = df[key].to_numpy()
                episode_data[key] = np.stack(col) if col.dtype == object else col
            ep_stats = compute_episode_stats(episode_data, numeric_features)

            for key in video_keys:
                video_path = root / info["video_path"].format(
                    episode_chunk=chunk, video_key=key, episode_index=ep_idx
                )
                ep_stats[key] = episode_video_stats(video_path, ep["length"], fps)

            f.write(json.dumps({"episode_index": ep_idx, "stats": stats_to_jsonable(ep_stats)}) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, required=True, help="v2.1 dataset dir (meta/, data/, videos/)")
    parser.add_argument("--dst", type=Path, required=True, help="output dir for the v3.0 dataset")
    parser.add_argument("--repo-id", type=str, default="local/converted", help="nominal repo id (local only)")
    parser.add_argument("--work-dir", type=Path, default=None, help="staging dir (default: temp dir)")
    args = parser.parse_args()

    src, dst = args.src.expanduser().resolve(), args.dst.expanduser().resolve()
    if not (src / "meta" / "info.json").exists():
        raise FileNotFoundError(f"{src} is not a LeRobot dataset (no meta/info.json)")
    if dst.exists():
        raise FileExistsError(f"{dst} already exists, refusing to overwrite")

    work_dir = args.work_dir or Path(tempfile.mkdtemp(prefix="lerobot_v21_to_v30_"))
    staged = work_dir / src.name
    print(f"Staging copy of {src} -> {staged}")
    shutil.copytree(src, staged)

    h2r_block = json.loads((staged / "meta" / "info.json").read_text()).get("h2r")

    generate_episodes_stats(staged)
    convert_dataset(repo_id=args.repo_id, root=staged, push_to_hub=False)

    if h2r_block is not None:
        info_path = staged / "meta" / "info.json"
        info = json.loads(info_path.read_text())
        info["h2r"] = h2r_block
        info_path.write_text(json.dumps(info, indent=4) + "\n")

    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(staged), str(dst))
    shutil.rmtree(work_dir, ignore_errors=True)
    print(f"Done: v3.0 dataset at {dst}")


if __name__ == "__main__":
    main()
