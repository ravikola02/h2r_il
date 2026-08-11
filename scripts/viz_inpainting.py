#!/usr/bin/env python
"""Test / visualize h2r_il inpainting methods (arm inpainting, ...).

For each requested method and a handful of sampled frames, dumps a labeled panel
— original next to the method's intermediates and final result (e.g. detected
boxes | mask overlay | inpainted) — plus a combined contact sheet, to ``--out``.
Nothing is ever written back into the dataset.

Because every :class:`~h2r_il.inpainting.InpaintingMethod` exposes ``visualize()``,
this tool works for any method you add later: register it, then

    uv run python scripts/viz_inpainting.py --methods all ...

Frames come from a LeRobot dataset (default) or from ``--images`` files.

Examples
--------
    # arm inpainting on 6 frames of a dataset (all cameras)
    uv run python scripts/viz_inpainting.py \\
        --dataset <hf-repo-id> --dataset-root <local dataset dir> \\
        --methods arm_inpaint --num-frames 6

    # every registered method, on explicit image files
    uv run python scripts/viz_inpainting.py --images a.png b.png --methods all

    # warm the on-disk cache for a whole dataset so training reads it (no models
    # loaded at train time). Uses H2R_INPAINTING_CACHE if set.
    uv run python scripts/viz_inpainting.py \\
        --dataset <hf-repo-id> --dataset-root ... \\
        --methods arm_inpaint --num-frames -1 --fill-cache --no-panels
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

import inspect

from h2r_il.inpainting import available_methods, build_inpainting_method, inpainting_method_class

REPO_ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- #
# frame sources
# --------------------------------------------------------------------------- #
def frames_from_images(paths: list[str]) -> tuple[Iterator[tuple[str, np.ndarray, str]], int]:
    # yields (name, image, group); group buckets frames into videos
    def _gen() -> Iterator[tuple[str, np.ndarray, str]]:
        for p in paths:
            yield Path(p).stem, np.asarray(Image.open(p).convert("RGB")), "images"

    return _gen(), len(paths)


def _episode_range(ds, episode: int) -> tuple[int, int]:
    """``[from, to)`` global frame indices of ``episode`` (LeRobot v3 metadata)."""
    eps = ds.meta.episodes
    n_eps = len(eps["dataset_from_index"])
    if not 0 <= episode < n_eps:
        raise SystemExit(f"--episode {episode} out of range: dataset has {n_eps} episode(s)")
    return int(eps["dataset_from_index"][episode]), int(eps["dataset_to_index"][episode])


def frames_from_dataset(
    repo_id: str, root: str | None, cameras: list[str] | None, num_frames: int,
    shard: int = 0, num_shards: int = 1, consecutive: bool = False, start: int = 0,
    episode: int | None = None,
) -> tuple[Iterator[tuple[str, np.ndarray, str]], int]:
    """Stream ``(name, HxWx3 uint8, camera)`` frames from a LeRobot dataset.

    Returns a *lazy* generator plus the total frame count. Lazy matters: warming
    the whole-dataset cache (``--num-frames -1``) is ~20k frames — materializing
    them all would need tens of GB of RAM.

    ``consecutive`` samples a contiguous run from ``start`` (a real temporal clip,
    for video) instead of frames spread evenly across the dataset.
    ``episode`` restricts the run to a single episode, resolving its global frame
    range from the dataset metadata; ``num_frames`` then truncates that episode
    (-1 / unset takes the whole thing) and ``start`` is an offset *within* it.
    ``shard``/``num_shards`` slice the frame indices so N processes (e.g. one per
    GPU, each with its own CUDA_VISIBLE_DEVICES) can warm the shared cache in
    parallel.
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(repo_id, root=root, return_uint8=True)
    cam_keys = cameras or list(ds.meta.camera_keys)
    n = len(ds)
    if episode is not None:
        ep0, ep1 = _episode_range(ds, episode)
        lo = ep0 + max(0, start)
        idxs = list(range(min(lo, ep1), ep1))
        if num_frames is not None and 0 <= num_frames < len(idxs):
            idxs = idxs[:num_frames]
    elif num_frames is None or num_frames < 0 or num_frames >= n:
        idxs = list(range(n))
    elif consecutive:
        idxs = list(range(start, min(start + num_frames, n)))
    else:
        idxs = np.linspace(0, n - 1, num_frames).round().astype(int).tolist()
    if num_shards > 1:
        idxs = idxs[shard::num_shards]

    def _gen() -> Iterator[tuple[str, np.ndarray, str]]:
        for i in idxs:
            item = ds[i]
            ep = int(item["episode_index"].item())
            fr = int(item["frame_index"].item())
            for cam in cam_keys:
                if cam not in item:
                    continue
                t = item[cam]
                if t.ndim == 4:  # (T,C,H,W) -> last frame
                    t = t[-1]
                arr = t.permute(1, 2, 0).cpu().numpy()  # HWC uint8
                short = cam.replace("observation.images.", "")
                yield f"ep{ep:03d}_f{fr:04d}_{short}", np.ascontiguousarray(arr), short

    return _gen(), len(idxs) * len(cam_keys)


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #
def _tile(img: np.ndarray, label: str, width: int) -> np.ndarray:
    """Resize ``img`` to ``width`` (keep aspect) and add a label bar on top."""
    h, w = img.shape[:2]
    new_h = max(1, round(h * width / w))
    im = Image.fromarray(img).resize((width, new_h), Image.BILINEAR)
    bar = 22
    canvas = Image.new("RGB", (width, new_h + bar), (20, 20, 20))
    canvas.paste(im, (0, bar))
    ImageDraw.Draw(canvas).text((5, 4), label, fill=(240, 240, 240))
    return np.asarray(canvas)


def _hconcat(tiles: list[np.ndarray]) -> np.ndarray:
    h = max(t.shape[0] for t in tiles)
    padded = [np.pad(t, ((0, h - t.shape[0]), (0, 0), (0, 0)), constant_values=20) for t in tiles]
    return np.concatenate(padded, axis=1)


def _vconcat(rows: list[np.ndarray]) -> np.ndarray:
    w = max(r.shape[1] for r in rows)
    padded = [np.pad(r, ((0, 0), (0, w - r.shape[1]), (0, 0)), constant_values=20) for r in rows]
    return np.concatenate(padded, axis=0)


def build_panel(original: np.ndarray, viz: dict[str, np.ndarray], title: str, tile_w: int) -> np.ndarray:
    tiles = [_tile(original, "original", tile_w)]
    # keep "result" last, drop any intermediate identical to original
    keys = [k for k in viz if k != "result"] + (["result"] if "result" in viz else [])
    for k in keys:
        tiles.append(_tile(viz[k], f"{title}: {k}", tile_w))
    return _hconcat(tiles)


def write_video(path: Path, frames_rgb: list[np.ndarray], fps: float) -> None:
    """Encode a list of equal-size RGB frames to an mp4 (cv2, no subprocess)."""
    import cv2

    if not frames_rgb:
        return
    h, w = frames_rgb[0].shape[:2]
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames_rgb:
        if f.shape[:2] != (h, w):
            f = cv2.resize(f, (w, h))
        vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    vw.release()


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_argument_group("frame source (choose one)")
    src.add_argument("--dataset", help="LeRobot dataset repo_id")
    src.add_argument("--dataset-root", help="local dataset root (offline)")
    src.add_argument("--images", nargs="+", help="explicit image files instead of a dataset")
    ap.add_argument("--cameras", nargs="+", help="camera keys (default: all in the dataset)")
    ap.add_argument("--num-frames", type=int, default=6, help="frames to sample (-1 = all)")

    ap.add_argument("--methods", nargs="+", default=["arm_inpaint"],
                    help='method names, or "all"')
    ap.add_argument("--spec", help="JSON spec fully overriding --methods, e.g. "
                    '\'[{"name":"arm_inpaint","dilation":21}]\'')

    # arm_inpaint tuning knobs (default=None -> only override when the user sets
    # them; they are forwarded to any method whose constructor accepts them).
    tune = ap.add_argument_group("arm_inpaint tuning (override method defaults)")
    tune.add_argument("--prompt", help="Grounding DINO text query, e.g. 'arm. hand.'")
    tune.add_argument("--box-threshold", type=float, dest="box_threshold",
                      help="detection confidence to keep a box (lower -> also keeps the "
                           "lower-scoring full-arm box, not just the hand). default 0.2")
    tune.add_argument("--text-threshold", type=float, dest="text_threshold",
                      help="how strongly a box must match the prompt words. default 0.2")
    tune.add_argument("--dilation", type=int,
                      help="px to grow the mask before inpainting (cleaner edges). default 15")
    tune.add_argument("--max-boxes", type=int, dest="max_boxes",
                      help="cap on boxes (highest-scoring) fed to SAM2. default 8")
    tune.add_argument("--min-box-area-frac", type=float, dest="min_box_area_frac",
                      help="drop boxes smaller than this fraction of the frame. default 0.0008")
    tune.add_argument("--max-box-area-frac", type=float, dest="max_box_area_frac",
                      help="drop boxes larger than this fraction (kills whole-frame false "
                           "positives at low box-threshold). default 0.6")
    tune.add_argument("--device", help="torch device for the models, e.g. cuda / cuda:1 / cpu")
    ap.add_argument("--cache", action="store_true", help="enable the on-disk cache")
    ap.add_argument("--fill-cache", action="store_true",
                    help="run each method through the cache (implies --cache); use to warm it")
    ap.add_argument("--no-panels", action="store_true", help="skip image output (e.g. cache-only)")
    ap.add_argument("--video", action="store_true",
                    help="write an mp4 (original|result over consecutive frames) per method+camera")
    ap.add_argument("--fps", type=float, default=10.0, help="frames/sec for --video")
    ap.add_argument("--start-index", type=int, default=0, help="first dataset index for --video clip")
    ap.add_argument("--episode", type=int,
                    help="restrict to one episode (its frame range is looked up in the "
                         "dataset metadata); --start-index is then an offset within it")
    ap.add_argument("--tile-width", type=int, default=560)
    ap.add_argument("--out", default=str(REPO_ROOT / "outputs" / "inpainting"))
    ap.add_argument("--num-shards", type=int, default=1,
                    help="split frames across N parallel processes (e.g. one per GPU)")
    ap.add_argument("--shard", type=int, default=0, help="this process's shard index [0, num-shards)")
    args = ap.parse_args()

    # resolve methods
    if args.spec:
        specs = json.loads(args.spec)
        specs = [s if isinstance(s, dict) else {"name": s} for s in specs]
    else:
        names = available_methods() if args.methods == ["all"] else args.methods
        specs = [{"name": n} for n in names]

    # CLI tuning overrides: only those the user actually set (default None).
    tunables = ["prompt", "box_threshold", "text_threshold", "dilation",
                "max_boxes", "min_box_area_frac", "max_box_area_frac", "device"]
    overrides = {k: getattr(args, k) for k in tunables if getattr(args, k) is not None}

    use_cache = args.cache or args.fill_cache
    methods = []
    for s in specs:
        s = dict(s)
        name = s.pop("name")
        # forward each override only to methods whose constructor accepts it;
        # CLI overrides win over --spec values.
        accepted = inspect.signature(inpainting_method_class(name).__init__).parameters
        s.update({k: v for k, v in overrides.items() if k in accepted})
        methods.append(build_inpainting_method(name, cache=use_cache, **s))

    # resolve frames (lazy: streamed one at a time, never all held in RAM).
    # --video wants a real temporal clip, so sample consecutive frames.
    if args.images:
        frame_iter, total = frames_from_images(args.images)
    elif args.dataset:
        frame_iter, total = frames_from_dataset(
            args.dataset, args.dataset_root, args.cameras, args.num_frames,
            shard=args.shard, num_shards=args.num_shards,
            consecutive=args.video, start=args.start_index, episode=args.episode,
        )
    else:
        ap.error("provide --images or --dataset")

    save_panels = not args.no_panels and not args.video  # --video -> video is the output
    need_result = save_panels or args.video               # else: cache-only warm

    jobs = total * len(methods)
    print(f"{total} frame(s) x {len(methods)} method(s) = {jobs} jobs; "
          f"methods: {[m.method_name for m in methods]}")
    if jobs > 200 and save_panels:
        print(f"NOTE: {jobs} panels will be written. For a whole-dataset cache "
              "warm, add --fill-cache --no-panels.")

    out_dir = Path(args.out)
    if save_panels or args.video:
        out_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[np.ndarray] = []
    videos: dict[tuple[str, str], list[np.ndarray]] = {}
    done = 0
    start = time.time()
    for fname, img, group in frame_iter:
        for m in methods:
            t0 = time.time()
            if not need_result:
                m(img[None].transpose(0, 3, 1, 2))  # exercise __call__/cache, discard
            elif args.video:
                # A video shows only original|result, so go through the cached
                # __call__ rather than visualize(): the latter recomputes the
                # boxes/mask intermediates the video never renders, and skips the
                # cache entirely (~0.65 s/frame even when the frame is warm).
                result = m(img)
                vframe = _hconcat([_tile(img, "original", args.tile_width),
                                   _tile(result, m.method_name, args.tile_width)])
                videos.setdefault((m.method_name, group), []).append(vframe)
            else:
                viz = m.visualize(img)
                panel = build_panel(img, viz, m.method_name, args.tile_width)
                Image.fromarray(panel).save(out_dir / f"{fname}__{m.method_name}.png")
                all_rows.append(panel)
            done += 1
            # progress with rolling ETA (per-frame when producing output; every 25
            # for a big cache-only warm)
            every = 1 if need_result else 25
            if done % every == 0 or done == jobs:
                rate = done / max(1e-6, time.time() - start)
                eta = (jobs - done) / max(1e-6, rate)
                print(f"  [{done}/{jobs}] {m.method_name} {fname}: "
                      f"{time.time() - t0:.2f}s  ({rate:.1f} frame/s, ETA {eta/60:.1f} min)",
                      flush=True)

    if all_rows and save_panels:
        Image.fromarray(_vconcat(all_rows)).save(out_dir / "contact_sheet.png")
        print(f"wrote {len(all_rows)} panel(s) + contact sheet -> {out_dir}")
    if videos:
        for (mname, group), vframes in videos.items():
            vpath = out_dir / f"{mname}__{group}.mp4"
            write_video(vpath, vframes, args.fps)
            print(f"wrote video ({len(vframes)} frames @ {args.fps}fps) -> {vpath}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
