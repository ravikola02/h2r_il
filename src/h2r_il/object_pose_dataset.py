#!/usr/bin/env python
"""Object pose for a whole LeRobot dataset, from one pass over one camera video.

The per-clip module (:mod:`h2r_il.object_pose`) tracks one video. This one points
it at a *dataset* instead, and the trick that makes that cheap is a property of
LeRobot v3 storage: a camera's episodes are concatenated, in episode order, into
one mp4 per chunk. When every episode of a camera lives in a single file, that
file's frame index **is** the dataset's global frame index -- so there is nothing
to cut, nothing to re-align, and no per-episode bookkeeping to get wrong.

That buys three things a per-episode sweep cannot:

  * **One annotation pass.** You tag the object in v2d's SAM2 UI on the whole
    video, re-anchoring wherever it drifts, instead of annotating N clips.
  * **One mesh and one metric scale.** SAM3D builds the object mesh once and MoGe
    solves scale once, so poses are directly comparable *across* episodes. A
    per-episode sweep re-estimates focal length per clip, which lets absolute
    scale drift between episodes -- exactly what an aux loss must not see.
  * **One crop.** Identical framing for every frame in the dataset.

What this module does NOT do is trust any of that silently. ``prepare`` proves the
concatenation holds -- packet count against ``total_frames``, and every episode's
start timestamp against the running sum of episode lengths -- and refuses to stage
anything if the mapping is off by even one frame. A silent one-frame shift here
would mislabel every pose in the dataset, and nothing downstream could detect it.

Usage -- `build` runs the whole chain and skips whatever is already on disk, so
it is both the start command and the resume command:

    # See what it would do and roughly how long, without doing it
    uv run --no-sync python -m h2r_il.object_pose_dataset build \
        --dataset-root /path/to/kitting --dry-run

    # Run it. Stops at the one manual step (tagging in v2d's SAM2 UI) with the
    # command to run; re-run this afterwards and it picks up from there.
    uv run --no-sync python -m h2r_il.object_pose_dataset build \
        --dataset-root /path/to/kitting

    # What exists so far
    uv run --no-sync python -m h2r_il.object_pose_dataset status \
        --dataset-root /path/to/kitting

The individual stages (`prepare`, `track`, `store`) remain available and are what
`build` shells out to; reach for them when re-running one step with different
arguments, and then let `build` rebuild only what went stale.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from h2r_il.object_pose import (
    DEFAULT_CROP,
    DEFAULT_CROP_FOR,
    REPO_DIR,
    WEIGHTS,
    probe_size,
    read_local_env,
    resolve_v2d_root,
)

DEFAULT_CAMERA = "observation.images.head"

# Frame-count agreement is the whole safety argument, so the tolerance is zero.
# Timestamps are floats, though, so the episode-start check rounds to the nearest
# frame and then demands an exact integer match.
TIMESTAMP_ROUNDING_TOLERANCE = 0.51


# --------------------------------------------------------------------------------
# Dataset metadata
# --------------------------------------------------------------------------------

def load_info(dataset_root: Path) -> dict:
    """meta/info.json -- feature list, fps, totals and the path templates."""
    path = dataset_root / "meta" / "info.json"
    if not path.is_file():
        sys.exit(f"not a LeRobot dataset (no {path}). Pass --dataset-root.")
    return json.loads(path.read_text())


def load_episodes(dataset_root: Path) -> pd.DataFrame:
    """meta/episodes/**.parquet as one frame, in episode order.

    v3 shards this table, so read every shard and sort -- relying on glob order
    would silently scramble the episode-start arithmetic below.
    """
    shards = sorted((dataset_root / "meta" / "episodes").glob("**/*.parquet"))
    if not shards:
        sys.exit(f"no episode metadata under {dataset_root / 'meta' / 'episodes'}")
    table = pd.concat([pd.read_parquet(shard) for shard in shards], ignore_index=True)
    return table.sort_values("episode_index").reset_index(drop=True)


def camera_keys(info: dict) -> list[str]:
    """Every video feature in the dataset, for error messages and --camera checks."""
    return [key for key, spec in info["features"].items()
            if spec.get("dtype") == "video"]


def resolve_camera_video(dataset_root: Path, info: dict, episodes: pd.DataFrame,
                         camera: str) -> Path:
    """The single mp4 holding every episode of ``camera``.

    Multiple files would break the identity this module rests on -- frame index
    equals global index only while one file holds the whole dataset -- so that
    case is refused rather than handled. Splitting the store per chunk is a real
    feature, but it is not this step, and guessing would produce poses silently
    misattributed to the wrong episodes.
    """
    if camera not in camera_keys(info):
        sys.exit(f"{camera!r} is not a video feature. Available: "
                 f"{', '.join(camera_keys(info)) or '(none)'}")

    chunks = episodes[f"videos/{camera}/chunk_index"].unique()
    files = episodes[f"videos/{camera}/file_index"].unique()
    if len(chunks) != 1 or len(files) != 1:
        sys.exit(
            f"{camera} spans {len(chunks)} chunk(s) x {len(files)} file(s). This "
            "module needs all episodes in one file, because it identifies the "
            "video's frame index with the dataset's global index. Per-file "
            "handling is not implemented; use a single-file export."
        )

    relative = info["video_path"].format(video_key=camera,
                                         chunk_index=int(chunks[0]),
                                         file_index=int(files[0]))
    video = dataset_root / relative
    if not video.is_file():
        sys.exit(f"missing video: {video}")
    return video


# --------------------------------------------------------------------------------
# The alignment proof
# --------------------------------------------------------------------------------

def count_frames(video: Path) -> int:
    """Decoded packet count. Slower than reading a header, and that is the point:
    a container's declared ``nb_frames`` is metadata and can lie."""
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
         "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", str(video)],
        check=True, capture_output=True, text=True,
    )
    return int(result.stdout.strip().split(",")[0])


def verify_alignment(video: Path, info: dict, episodes: pd.DataFrame,
                     camera: str) -> dict:
    """Prove video frame index == dataset global index, or exit.

    Two independent checks, because either alone can pass on a broken mapping:

    1. Total frames. The video must hold exactly ``total_frames`` frames -- no
       padding, no dropped tail.
    2. Every episode start. Episode e must begin at video frame
       ``sum(lengths[:e])``, cross-checked against its recorded
       ``from_timestamp``. This is the check that catches a shift *inside* the
       file, which check 1 cannot see.
    """
    fps = float(info["fps"])
    total_frames = int(info["total_frames"])

    counted = count_frames(video)
    if counted != total_frames:
        sys.exit(f"ALIGNMENT FAILED: {video.name} holds {counted} frames but the "
                 f"dataset records {total_frames}. Frame index cannot be the "
                 "global index; refusing to stage.")

    lengths = episodes["length"].to_numpy()
    starts = lengths.cumsum() - lengths                      # exclusive prefix sum
    from_ts = episodes[f"videos/{camera}/from_timestamp"].to_numpy()
    expected_frame = from_ts * fps

    drift = abs(expected_frame - starts)
    worst = int(drift.argmax())
    if drift[worst] > TIMESTAMP_ROUNDING_TOLERANCE:
        sys.exit(
            f"ALIGNMENT FAILED at episode {int(episodes['episode_index'][worst])}: "
            f"from_timestamp {from_ts[worst]:.6f}s is video frame "
            f"{expected_frame[worst]:.2f}, but the episode lengths put it at frame "
            f"{starts[worst]}. The episodes are not stored back to back; refusing "
            "to stage."
        )

    return {
        "video": str(video),
        "camera": camera,
        "fps": fps,
        "total_frames": total_frames,
        "counted_frames": counted,
        "episodes": int(len(episodes)),
        "max_episode_start_drift_frames": float(drift.max()),
        "identity": "video frame index == dataset global index",
    }


# --------------------------------------------------------------------------------
# Tagging guide
# --------------------------------------------------------------------------------

def tagging_guide(episodes: pd.DataFrame, camera: str) -> dict:
    """Where to re-anchor while tagging: every episode boundary is a hard cut.

    Concatenation makes one long video out of N unrelated takes, so the picture
    jumps discontinuously at each episode start. SAM2 propagates a mask forward
    from an annotated frame and has no reason to survive a cut, so these frames
    are where a new box is worth the most. They are a starting point, not a
    prescription -- add more wherever the object is occluded or moves fast.
    """
    lengths = episodes["length"].to_numpy()
    starts = lengths.cumsum() - lengths
    return {
        "camera": camera,
        "note": ("Each listed frame is the first frame of an episode, i.e. a hard "
                 "scene cut. Tag at least these; add more where the object is "
                 "occluded, small, or moving fast."),
        "episode_start_frames": [
            {"episode_index": int(e), "start_frame": int(s), "length": int(n)}
            for e, s, n in zip(episodes["episode_index"], starts, lengths)
        ],
    }


# --------------------------------------------------------------------------------
# Staging
# --------------------------------------------------------------------------------

def stage_video(video: Path, work_dir: Path, stem: str) -> Path:
    """Hard-link the dataset video into the work dir under a readable name.

    :mod:`h2r_il.object_pose` writes ``<stem>_crop.mp4`` and the prompts JSON
    *next to the video it is given*, so it must be given a path in our own
    directory rather than one inside the dataset.

    A hard link, specifically, and this is the whole point: object_pose resolves
    ``--video`` with ``Path.resolve()``, which follows a **symlink** back to the
    dataset and writes there anyway -- silently defeating the staging. This is not
    hypothetical; a 152 MB crop and the prompts JSON landed in the dataset's video
    directory that way. ``resolve()`` has nothing to follow on a hard link: the
    path stays in the work dir while the bytes stay shared, so staging costs no
    disk and the dataset stays untouched.

    Hard links cannot cross filesystems. Outputs and datasets share one mount
    here, so the fallback should not trigger, but a copy is better than quietly
    writing into the dataset again.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    staged = work_dir / f"{stem}.mp4"
    if staged.is_symlink() or staged.exists():
        staged.unlink()
    try:
        os.link(video, staged)
    except OSError:
        print(f"[warn] cannot hard-link across filesystems; copying "
              f"{video.stat().st_size / 1e6:.0f} MB instead.")
        shutil.copy2(video, staged)
    return staged


# --------------------------------------------------------------------------------
# The pose store
# --------------------------------------------------------------------------------

def build_store(trajectory: Path, info: dict, episodes: pd.DataFrame,
                offset: int) -> tuple[dict, dict]:
    """Place a trajectory onto the dataset's global frame axis.

    ``offset`` is the dataset frame the trajectory's frame 0 corresponds to. For a
    whole-video run that is 0 and the mapping is the identity; for a trajectory
    tracked from a single episode's clip it is that episode's start frame. Keeping
    the offset explicit is what lets per-episode runs be stitched into the same
    store if whole-video tracking turns out not to survive the scene cuts.

    Dense, not sparse. 20613 frames of 4x4 float32 is 1.3 MB -- small enough that
    the simplest possible layout wins, and a DataLoader gets O(1) lookup by the
    global index it already has. Frames with no pose stay zero and are marked in
    ``valid``; that mask has to reach the loss, or uncovered frames would train
    the model towards a pose of all zeros.
    """
    data = np.load(trajectory)
    total_frames = int(info["total_frames"])

    frames = data["frame_index"].astype(np.int64) + offset
    if frames.min() < 0 or frames.max() >= total_frames:
        sys.exit(f"trajectory maps to dataset frames {frames.min()}..{frames.max()}, "
                 f"outside 0..{total_frames - 1}. Wrong --offset/--episode?")

    valid = np.zeros(total_frames, dtype=bool)
    valid[frames] = True

    store = {
        "T_cam_obj": np.zeros((total_frames, 4, 4), dtype=np.float32),
        "position": np.zeros((total_frames, 3), dtype=np.float32),
        "quat_xyzw": np.zeros((total_frames, 4), dtype=np.float32),
        "valid": valid,
    }
    store["T_cam_obj"][frames] = data["T_cam_obj"].astype(np.float32)
    store["position"][frames] = data["position"].astype(np.float32)
    store["quat_xyzw"][frames] = data["quat_xyzw"].astype(np.float32)

    # Carry the episode axis so a consumer can group, split or report by episode
    # without re-reading the dataset's own metadata.
    lengths = episodes["length"].to_numpy()
    starts = lengths.cumsum() - lengths
    episode_of = np.repeat(episodes["episode_index"].to_numpy(), lengths)
    frame_in_episode = np.arange(total_frames) - np.repeat(starts, lengths)
    store["episode_index"] = episode_of.astype(np.int32)
    store["frame_in_episode"] = frame_in_episode.astype(np.int32)

    store["intrinsics"] = data["intrinsics"].astype(np.float32)
    store["image_size"] = data["image_size"].astype(np.int32)

    covered = [int(valid[s:s + n].sum()) for s, n in zip(starts, lengths)]
    coverage = {
        "frames_with_pose": int(valid.sum()),
        "frames_total": total_frames,
        "episodes_fully_covered": int(sum(c == n for c, n in zip(covered, lengths))),
        "episodes_partially_covered": int(sum(0 < c < n for c, n in zip(covered, lengths))),
        "episodes_uncovered": int(sum(c == 0 for c in covered)),
        "per_episode": [
            {"episode_index": int(e), "covered": int(c), "length": int(n)}
            for e, c, n in zip(episodes["episode_index"], covered, lengths)
        ],
    }
    return store, coverage


def write_store(store: dict, coverage: dict, out_dir: Path, provenance: dict) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / "object_pose.npz"
    np.savez(npz_path, **store)
    meta = dict(provenance)
    meta["coverage"] = coverage
    (out_dir / "object_pose.json").write_text(json.dumps(meta, indent=2))
    return npz_path


def dataset_name(dataset_root: Path) -> str:
    """A short label for this dataset, used to namespace outputs.

    Export layouts bury the dataset under a format directory -- here the real path
    is ``<...>/kitting/lerobot_v3``, reached through a ``datasets/kitting``
    symlink. Taking the basename blindly labels every dataset ``lerobot_v3`` and
    quietly lands gear_left's poses and kitting's in the same directory, so a
    version-looking basename defers to its parent.
    """
    name = dataset_root.name
    if name.startswith("lerobot"):
        return dataset_root.parent.name
    return name


def default_work_dir(dataset_root: Path, camera: str) -> Path:
    """``outputs/object_pose/<dataset>/<camera>``.

    Deliberately not inside the dataset. The pose store is a derived artifact of
    *this* project, and the dataset stays exactly as exported -- the same rule the
    inpainting cache follows.
    """
    short = camera.rsplit(".", 1)[-1]
    return REPO_DIR / "outputs" / "object_pose" / dataset_name(dataset_root) / short


def resolve_dataset_root(argument: str | None) -> Path:
    """--dataset-root, else <DATASETS_ROOT>/kitting from the gitignored local.env.

    ``absolute()`` rather than ``resolve()``: the dataset directories here are
    symlinks, and following them replaces the name the user typed with the export
    layout's own. Absolute-but-unresolved keeps paths in the user's terms, and
    reads through the symlink work either way.
    """
    if argument:
        root = Path(argument).expanduser().absolute()
    else:
        datasets_root = read_local_env("DATASETS_ROOT")
        if not datasets_root:
            sys.exit("no --dataset-root, and DATASETS_ROOT is not in configs/local.env")
        root = Path(datasets_root).expanduser().absolute() / "kitting"
        print(f"No --dataset-root given; using {root}")
    if not root.is_dir():
        sys.exit(f"no such dataset directory: {root}")
    return root


# --------------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------------

def command_prepare(args: argparse.Namespace) -> int:
    dataset_root = resolve_dataset_root(args.dataset_root)
    info = load_info(dataset_root)
    episodes = load_episodes(dataset_root)
    video = resolve_camera_video(dataset_root, info, episodes, args.camera)

    print(f"dataset   {dataset_root}")
    print(f"camera    {args.camera}")
    print(f"video     {video.relative_to(dataset_root)}  "
          f"({video.stat().st_size / 1e6:.0f} MB)")
    print("\nVerifying that the video's frame index is the dataset's global index...")
    alignment = verify_alignment(video, info, episodes, args.camera)
    print(f"  OK  {alignment['counted_frames']} frames == total_frames, "
          f"{alignment['episodes']} episodes back to back "
          f"(worst start drift {alignment['max_episode_start_drift_frames']:.3f} frames)")

    width, height = probe_size(video)
    if (width, height) != DEFAULT_CROP_FOR:
        print(f"\n[warn] this video is {width}x{height}, but object_pose's default "
              f"crop {DEFAULT_CROP} was tuned for "
              f"{DEFAULT_CROP_FOR[0]}x{DEFAULT_CROP_FOR[1]}. Pass explicit bounds "
              "or --crop none when you track.")

    work_dir = Path(args.work_dir).expanduser().resolve() if args.work_dir \
        else default_work_dir(dataset_root, args.camera)
    stem = f"{dataset_root.name}_{args.camera.rsplit('.', 1)[-1]}"
    staged = stage_video(video, work_dir, stem)

    (work_dir / "alignment.json").write_text(json.dumps(alignment, indent=2))
    guide = tagging_guide(episodes, args.camera)
    (work_dir / "tagging_guide.json").write_text(json.dumps(guide, indent=2))

    run_dir = work_dir / "run"
    print(f"\nstaged    {staged}  (symlink; the dataset itself is untouched)")
    print(f"written   alignment.json, tagging_guide.json "
          f"({len(guide['episode_start_frames'])} episode-start frames to tag)")
    print("\nNext -- tag the object yourself, then track. The first run opens the")
    print("SAM2 UI because no prompts exist yet; Ctrl-C there when you are done")
    print("and it asks before spending GPU time:\n")
    print(f"    uv run --no-sync python -m h2r_il.object_pose \\\n"
          f"        --video {staged} \\\n"
          f"        --out {run_dir}\n")
    print("While tagging, re-anchor at the episode-start frames listed in "
          "tagging_guide.json -- each one is a hard scene cut.")
    return 0


def command_track(args: argparse.Namespace) -> int:
    """FoundationPose over the whole camera video, re-registering on a schedule.

    v2d's own step 8 registers once and tracks everything after it, which is
    wrong for a concatenated dataset: at each episode cut the pose prior belongs
    to a different take. On kitting that lost the object just after the first cut
    and held a pose 12x too close for 98.7% of the dataset.

    Two changes, both measured against MoGe depth (independent of FoundationPose):
      * register at every episode boundary -- the cut is known, no need to infer it;
      * register every N frames within an episode -- error grows between
        registrations, and N=15 cut median error ~3x on 2 of 3 test episodes and
        won p90 on all 3, at ~50% more wall time.

    Sparse schedules are worse than none: N=60 left 15.4% of frames >10 cm off,
    because a failed registration then stands for 60 frames. Either register
    often or not at all.
    """
    dataset_root = resolve_dataset_root(args.dataset_root)
    info = load_info(dataset_root)
    episodes = load_episodes(dataset_root)
    resolve_camera_video(dataset_root, info, episodes, args.camera)   # re-validate

    work_dir = Path(args.work_dir).expanduser().absolute() if args.work_dir \
        else default_work_dir(dataset_root, args.camera)
    run_dir = Path(args.out).expanduser().absolute() if args.out else work_dir / "run"
    stem = f"{dataset_name(dataset_root)}_{args.camera.rsplit('.', 1)[-1]}"
    video = work_dir / f"{stem}_crop.mp4"
    if not video.is_file():
        sys.exit(f"no cropped video at {video}. Run `prepare`, then tag and track "
                 "once with h2r_il.object_pose to produce the crop and prompts.")
    for required in ("depth", "masks", "intrinsics", "scaled_mesh.glb"):
        if not (run_dir / required).exists():
            sys.exit(f"{run_dir / required} missing; this step reuses the earlier "
                     "stages' outputs and cannot rebuild them.")

    lengths = episodes["length"].to_numpy()
    starts = (lengths.cumsum() - lengths)[1:]        # frame 0 is the reference
    v2d_root = resolve_v2d_root()

    module_src = Path(__file__).resolve().parent / "v2d_ext" / "run_episodes.py"
    container_module = "/workspace/v2d_foundation_pose/lib/run_episodes.py"

    sys.path.insert(0, str(v2d_root / "reconstruction" / "modules"))
    sys.path.insert(0, str(v2d_root / "reconstruction"))
    os.chdir(v2d_root / "reconstruction")
    from v2d.docker.container import run_in_container   # noqa: E402

    weights = WEIGHTS["foundation_pose"]
    weights_container = f"/data/weights_dir/{os.path.basename(os.path.abspath(weights))}"

    print(f"tracking {video.name}: {len(starts)} episode boundaries"
          + (f", plus every {args.register_every} frames" if args.register_every else ""))
    run_in_container(
        image="v2d_foundation_pose",
        module="v2d.foundation_pose.lib.run_episodes",
        inputs={
            "video_path": str(video),
            "depth_folder": str(run_dir / "depth"),
            "masks_folder": str(run_dir / "masks" / str(args.object_id)),
            "camera_intrinsics_path": str(run_dir / "intrinsics" / "000000.json"),
            "mesh_path": str(run_dir / "scaled_mesh.glb"),
            "weights_dir": weights,
        },
        outputs={"poses_dir": str(run_dir / "poses")},
        extra_args={
            "reference_frame": 0,
            "register_frames": ",".join(str(int(s)) for s in starts),
            "register_every": args.register_every,
            "register_iteration": 10,
            "track_iteration": 5,
            # Measured worse: constraining rotation pushes orientation error into
            # translation (median 2-3x worse on every test episode). For a
            # position-only target, drop rotation when building the target
            # instead -- the store keeps raw poses precisely so that stays a
            # downstream choice.
            "fix_rotation": args.fix_rotation,
        },
        gpus=True,
        env={"FOUNDATIONPOSE_WEIGHTS_DIR": weights_container},
        # Single-file mount: the image's compiled FoundationPose CUDA extensions
        # live under /workspace and a full dev mount would hide them.
        extra_volumes=[f"{module_src}:{container_module}:ro"],
    )
    print(f"\nposes -> {run_dir / 'poses'}")
    print("Next: rebuild the trajectory, then place it on the dataset's frame axis:")
    print(f"    python -m h2r_il.object_pose --video {video} --crop none "
          f"--reduce-only --out {run_dir}")
    print(f"    python -m h2r_il.object_pose_dataset store "
          f"--dataset-root {dataset_root} --camera {args.camera}")
    return 0


def command_store(args: argparse.Namespace) -> int:
    dataset_root = resolve_dataset_root(args.dataset_root)
    info = load_info(dataset_root)
    episodes = load_episodes(dataset_root)
    work_dir = Path(args.work_dir).expanduser().absolute() if args.work_dir \
        else default_work_dir(dataset_root, args.camera)

    trajectory = (Path(args.trajectory).expanduser().absolute() if args.trajectory
                  else work_dir / "run" / "trajectory.npz")
    if not trajectory.is_file():
        sys.exit(f"no trajectory at {trajectory}. Track first, or pass --trajectory.")

    # --episode is the ergonomic form of --offset: a trajectory tracked from one
    # episode's clip starts at that episode's first dataset frame.
    if args.episode is not None:
        lengths = episodes["length"].to_numpy()
        starts = lengths.cumsum() - lengths
        matches = episodes.index[episodes["episode_index"] == args.episode]
        if len(matches) == 0:
            sys.exit(f"no episode {args.episode} in this dataset.")
        offset = int(starts[matches[0]])
    else:
        offset = args.offset

    store, coverage = build_store(trajectory, info, episodes, offset)

    run_manifest = trajectory.parent / "run_manifest.json"
    provenance = {
        "dataset": str(dataset_root),
        "dataset_name": dataset_name(dataset_root),
        "camera": args.camera,
        "trajectory": str(trajectory),
        "frame_offset": offset,
        "frame_convention": "index into the dataset's global frame axis",
        "pose_convention": ("object->camera 4x4 in the OpenCV optical frame, as "
                            "written by h2r_il.object_pose"),
        "run_manifest": (json.loads(run_manifest.read_text())
                         if run_manifest.is_file() else None),
    }
    out_dir = Path(args.out).expanduser().absolute() if args.out else work_dir / "store"
    npz_path = write_store(store, coverage, out_dir, provenance)

    print(f"trajectory  {trajectory}")
    print(f"offset      {offset}  (dataset frame of the trajectory's frame 0)")
    print(f"store       {npz_path}  ({npz_path.stat().st_size / 1e6:.1f} MB)")
    print(f"\ncoverage    {coverage['frames_with_pose']}/{coverage['frames_total']} "
          f"frames ({100 * coverage['frames_with_pose'] / coverage['frames_total']:.1f}%)")
    print(f"            episodes: {coverage['episodes_fully_covered']} full, "
          f"{coverage['episodes_partially_covered']} partial, "
          f"{coverage['episodes_uncovered']} uncovered")
    if coverage["episodes_uncovered"]:
        print("\nUncovered episodes carry valid=False for every frame. Whatever "
              "consumes this store must mask them out of the loss.")
    return 0


# --------------------------------------------------------------------------------
# The whole chain, as one command
# --------------------------------------------------------------------------------

# In dependency order: each stage consumes the previous one's output. That is why
# completion is decided by what is on disk and its mtimes (see _stage_done) rather
# than by a state file -- a state file can disagree with reality after a manual
# rerun of one step, and the failure mode is a store built from stale poses.
STAGE_ORDER = ("prepare", "reconstruct", "track", "reduce", "store")

STAGE_HELP = {
    "prepare": "verify frame alignment, stage the camera video",
    "reconstruct": "v2d stock pipeline: crop, frames, SAM2 masks, MoGe depth, "
                   "SAM3D mesh, metric scale",
    "track": "FoundationPose with the re-registration schedule",
    "reduce": "poses -> trajectory.npz (+ overlay video)",
    "store": "place the trajectory on the dataset's global frame axis",
}

# Measured wall time on kitting's head camera, one GPU, for the plan printout.
# `reconstruct` is the padded one: v2d's run_pipeline is monolithic, so it also
# runs a register-once FoundationPose pass whose poses `track` then replaces.
# ~1.7 h of that stage is thrown away and there is no stage selector to skip it --
# worth knowing before you start, not worth forking v2d over.
BASELINE_FRAMES = 20613
STAGE_MINUTES = {"prepare": 1, "reconstruct": 220, "track": 190, "reduce": 5, "store": 1}
# Which of those scale with the video length. prepare and store are metadata work
# on any dataset; the three GPU stages are per-frame, so quoting kitting's numbers
# for a dataset a third the size would overstate it threefold.
FRAME_SCALED_STAGES = ("reconstruct", "track", "reduce")


def _stage_budget_minutes(stage: str, frames: int) -> float:
    """Rough minutes for one stage at this dataset's size. Deliberately coarse --
    it exists to make 'overnight or over lunch?' answerable, not to be a forecast."""
    minutes = STAGE_MINUTES[stage]
    if stage in FRAME_SCALED_STAGES and frames > 0:
        return minutes * frames / BASELINE_FRAMES
    return minutes


def _stage_paths(dataset_root: Path, work_dir: Path, camera: str) -> dict:
    stem = f"{dataset_name(dataset_root)}_{camera.rsplit('.', 1)[-1]}"
    run_dir = work_dir / "run"
    return {
        "stem": stem,
        "run": run_dir,
        "alignment": work_dir / "alignment.json",
        "staged": work_dir / f"{stem}.mp4",
        "crop": work_dir / f"{stem}_crop.mp4",
        "prompts": work_dir / f"{stem}_crop_prompts.json",
        "poses": run_dir / "poses",
        "trajectory": run_dir / "trajectory.npz",
        "store": work_dir / "store" / "object_pose.npz",
    }


def _mtime(path: Path) -> float:
    """0.0 for anything missing, so a missing input never looks newer."""
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def _fresher_than(target: Path, source: Path) -> bool:
    """Is ``target`` present and at least as new as ``source``?

    ``>=`` rather than ``>``: reduce writes trajectory.npz seconds after the last
    pose file, and on a coarse filesystem clock the two can land on the same
    stamp. Treating equal as stale would rerun both tail stages on every call.
    """
    return target.exists() and _mtime(target) >= _mtime(source)


def _stage_done(stage: str, paths: dict, expected_frames: int) -> bool:
    if stage == "prepare":
        return paths["alignment"].is_file() and paths["staged"].exists()
    if stage == "reconstruct":
        return paths["crop"].is_file() and all(
            (paths["run"] / name).exists()
            for name in ("depth", "masks", "intrinsics", "scaled_mesh.glb")
        )
    if stage == "track":
        poses = paths["poses"]
        if not poses.is_dir():
            return False
        # A crashed track leaves a partial directory that looks plausible. One
        # pose per dataset frame is the only check that catches that.
        return sum(1 for _ in poses.iterdir()) >= expected_frames > 0
    if stage == "reduce":
        return _fresher_than(paths["trajectory"], paths["poses"])
    if stage == "store":
        return _fresher_than(paths["store"], paths["trajectory"])
    raise ValueError(f"unknown stage {stage!r}")


def _stage_command(stage: str, args, dataset_root: Path, work_dir: Path,
                   paths: dict) -> list[str]:
    common = ["--dataset-root", str(dataset_root), "--camera", args.camera,
              "--work-dir", str(work_dir)]
    run_dir = paths["run"]

    if stage == "prepare":
        return [sys.executable, "-m", "h2r_il.object_pose_dataset", "prepare", *common]

    if stage == "reconstruct":
        cmd = [sys.executable, "-m", "h2r_il.object_pose",
               "--video", str(paths["staged"]), "--out", str(run_dir)]
        if args.object_id is not None:
            cmd += ["--object-id", str(args.object_id)]
        return cmd

    if stage == "track":
        cmd = [sys.executable, "-m", "h2r_il.object_pose_dataset", "track", *common,
               "--out", str(run_dir), "--register-every", str(args.register_every)]
        if args.object_id is not None:
            cmd += ["--object-id", str(args.object_id)]
        if args.fix_rotation:
            cmd += ["--fix-rotation"]
        return cmd

    if stage == "reduce":
        # --crop none against the ALREADY cropped video is mandatory, not tidiness:
        # the default crop box would re-encode <stem>_crop.mp4 from itself, cropping
        # a crop and clobbering the file every later stage reads.
        cmd = [sys.executable, "-m", "h2r_il.object_pose",
               "--video", str(paths["crop"]), "--crop", "none",
               "--reduce-only", "--out", str(run_dir)]
        if args.no_overlay:
            cmd += ["--no-overlay"]
        return cmd

    if stage == "store":
        return [sys.executable, "-m", "h2r_il.object_pose_dataset", "store", *common,
                "--trajectory", str(paths["trajectory"])]

    raise ValueError(f"unknown stage {stage!r}")


def _child_env() -> dict:
    """Make the h2r_il package importable in the child regardless of our launcher.

    The documented invocations are a mix of ``uv run`` and ``PYTHONPATH=src
    .venv/bin/python``. Rather than depend on which one started us, put this
    package's parent on the child's path explicitly.
    """
    env = os.environ.copy()
    package_parent = str(Path(__file__).resolve().parent.parent)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (package_parent + os.pathsep + existing) if existing \
        else package_parent
    return env


def _format_duration(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f} s"
    if seconds < 5400:
        return f"{seconds / 60:.1f} min"
    return f"{seconds / 3600:.2f} h"


def command_build(args: argparse.Namespace) -> int:
    """Run the whole chain from dataset to lookup table, resuming where it left off.

    The five stages are separate processes on purpose. ``track`` chdirs into the
    v2d checkout and prepends to ``sys.path`` to import the container helper, so
    running the stages in one process would leave the cwd and import state of one
    stage sitting under the next -- and ``reduce`` resolves relative paths.

    Stages already satisfied on disk are skipped, so this is the resume command as
    well as the start command: after a crash, or after re-running ``track`` with a
    different schedule by hand, calling it again does exactly the outstanding work.
    """
    dataset_root = resolve_dataset_root(args.dataset_root)
    info = load_info(dataset_root)
    expected_frames = int(info.get("total_frames", 0))
    work_dir = Path(args.work_dir).expanduser().absolute() if args.work_dir \
        else default_work_dir(dataset_root, args.camera)
    paths = _stage_paths(dataset_root, work_dir, args.camera)

    if args.only:
        stages, forced_from = [args.only], 0
    else:
        begin = args.start_from or ("prepare" if args.force else None)
        stages = list(STAGE_ORDER[STAGE_ORDER.index(begin):] if begin else STAGE_ORDER)
        # With no explicit start, nothing is forced: every stage runs only if the
        # disk says it has not been done.
        forced_from = 0 if begin else len(stages)

    plan = []
    for position, stage in enumerate(stages):
        done = _stage_done(stage, paths, expected_frames)
        plan.append((stage, (position >= forced_from) or not done, done))

    print(f"dataset   {dataset_root}")
    print(f"camera    {args.camera}")
    print(f"work dir  {work_dir}")
    print(f"frames    {expected_frames}")
    if args.fix_rotation:
        print("\n[warn] --fix-rotation forces rotation to identity while tracking. "
              "Measured on\n       episodes 5/6/8 it made translation 2-3x worse "
              "(FoundationPose absorbs the\n       denied orientation into position). "
              "Rotation is unvalidated either way.")
    print("\nplan:")
    for stage, will_run, done in plan:
        mark = "run " if will_run else "skip"
        note = "already done" if done else "not done"
        estimate = _stage_budget_minutes(stage, expected_frames)
        budget = f"~{estimate:.0f} min" if will_run else ""
        print(f"  [{mark}] {stage:<11} {note:<12} {budget:<9} {STAGE_HELP[stage]}")

    budget = sum(_stage_budget_minutes(s, expected_frames)
                 for s, will_run, _ in plan if will_run)
    if budget:
        print(f"\n  rough budget {_format_duration(budget * 60)} of GPU time")
    if args.dry_run:
        return 0

    env = _child_env()
    timings: list[tuple[str, float]] = []
    for stage, will_run, _ in plan:
        if not will_run:
            continue

        # The one human step. object_pose opens the SAM2 UI itself on a terminal,
        # so only a non-interactive run has to stop here -- and it must stop before
        # burning GPU time on a pipeline that would exit at the same check.
        if stage == "reconstruct" and not paths["prompts"].is_file() \
                and not sys.stdin.isatty():
            print(f"\nPAUSED: no prompts at {paths['prompts']}.")
            print("Tagging is manual and needs a terminal. Tag the object, then "
                  "re-run this command:\n")
            print(f"    python -m h2r_il.object_pose --video {paths['staged']} \\")
            print(f"        --out {paths['run']}\n")
            print("Re-anchor at the episode-start frames in tagging_guide.json -- "
                  "each is a hard cut.")
            return 2

        command = _stage_command(stage, args, dataset_root, work_dir, paths)
        print(f"\n{'=' * 78}\n== {stage}: {STAGE_HELP[stage]}\n{'=' * 78}")
        print("$ " + " ".join(command))
        began = time.monotonic()
        result = subprocess.run(command, env=env)
        elapsed = time.monotonic() - began
        timings.append((stage, elapsed))
        if result.returncode != 0:
            print(f"\n{stage} failed after {_format_duration(elapsed)} "
                  f"(exit {result.returncode}). Nothing after it ran; fix the cause "
                  f"and re-run to resume here.")
            return result.returncode
        print(f"\n-- {stage} done in {_format_duration(elapsed)}")

        if not _stage_done(stage, paths, expected_frames):
            # Exit 0 with the output missing means an assumption above is wrong.
            # Continuing would build the store out of whatever stale thing is there.
            print(f"\n{stage} exited 0 but its output is still missing or "
                  f"incomplete. Stopping rather than feeding the next stage "
                  f"something stale.")
            return 1

    if timings:
        print(f"\n{'=' * 78}")
        for stage, elapsed in timings:
            print(f"  {stage:<11} {_format_duration(elapsed)}")
        print(f"  {'total':<11} {_format_duration(sum(t for _, t in timings))}")

    store_meta = paths["store"].with_suffix(".json")
    if store_meta.is_file():
        coverage = json.loads(store_meta.read_text())["coverage"]
        print(f"\nlookup table  {paths['store']}")
        print(f"              {coverage['frames_with_pose']}/"
              f"{coverage['frames_total']} frames, "
              f"{coverage['episodes_fully_covered']} episodes fully covered")
        print(f"\nTrain against it with:\n"
              f"    CONFIG=configs/{dataset_name(dataset_root)}.env "
              f"OBJECT_POSE={paths['store'].parent} ./scripts/ft_groot.sh")
    return 0


def command_status(args: argparse.Namespace) -> int:
    dataset_root = resolve_dataset_root(args.dataset_root)
    work_dir = Path(args.work_dir).expanduser().resolve() if args.work_dir \
        else default_work_dir(dataset_root, args.camera)
    if not work_dir.is_dir():
        print(f"nothing staged yet at {work_dir}; run `prepare` first.")
        return 0

    print(f"work dir  {work_dir}")
    stem = f"{dataset_root.name}_{args.camera.rsplit('.', 1)[-1]}"
    for label, path in (
        ("staged video", work_dir / f"{stem}.mp4"),
        ("cropped clip", work_dir / f"{stem}_crop.mp4"),
        ("prompts     ", work_dir / f"{stem}_crop_prompts.json"),
        ("run manifest", work_dir / "run" / "run_manifest.json"),
        ("trajectory  ", work_dir / "run" / "trajectory.npz"),
        ("overlay     ", work_dir / "run" / "overlay.mp4"),
        ("pose store  ", work_dir / "store" / "object_pose.npz"),
    ):
        mark = "x" if path.exists() else " "
        print(f"  [{mark}] {label}  {path.name}")

    store_meta = work_dir / "store" / "object_pose.json"
    if store_meta.is_file():
        coverage = json.loads(store_meta.read_text())["coverage"]
        print(f"\n  store covers {coverage['frames_with_pose']}/"
              f"{coverage['frames_total']} frames; episodes "
              f"{coverage['episodes_fully_covered']} full, "
              f"{coverage['episodes_partially_covered']} partial, "
              f"{coverage['episodes_uncovered']} uncovered")

    prompts = work_dir / f"{stem}_crop_prompts.json"
    if prompts.is_file():
        entries = json.loads(prompts.read_text())
        entries = entries.get("prompts", entries) if isinstance(entries, dict) else entries
        frames = sorted({int(entry["frame_index"]) for entry in entries})
        print(f"\n  {len(entries)} prompt(s) on {len(frames)} frame(s): "
              f"{frames[:12]}{' ...' if len(frames) > 12 else ''}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m h2r_il.object_pose_dataset",
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    for name, handler, help_text in (
        ("prepare", command_prepare,
         "Verify frame alignment and stage the camera video for tagging."),
        ("track", command_track,
         "FoundationPose over the camera video, re-registering at episode "
         "boundaries and on a periodic schedule."),
        ("store", command_store,
         "Place a tracked trajectory onto the dataset's global frame axis."),
        ("build", command_build,
         "Run the whole chain -- prepare, reconstruct, track, reduce, store -- "
         "skipping whatever is already on disk."),
        ("status", command_status, "What has been produced so far."),
    ):
        sub = subparsers.add_parser(name, help=help_text, description=help_text)
        sub.add_argument("--dataset-root", help="LeRobot dataset directory. "
                                                "Default: <DATASETS_ROOT>/kitting")
        sub.add_argument("--camera", default=DEFAULT_CAMERA,
                         help=f"Video feature to track. Default: {DEFAULT_CAMERA}")
        sub.add_argument("--work-dir", help="Where to stage. Default: "
                                            "outputs/object_pose/<dataset>/<camera>")
        if name in ("track", "build"):
            sub.add_argument("--out", help="Run directory holding depth/masks/"
                                           "intrinsics/mesh. Default: <work-dir>/run")
            sub.add_argument("--object-id", type=int,
                             default=None if name == "build" else 0,
                             help="Which mask subdirectory to track. Default 0 "
                                  "(build: let each stage pick its own default).")
            sub.add_argument("--register-every", type=int, default=15,
                             help="Re-register every N frames within an episode, on "
                                  "top of the episode boundaries. 0 disables. "
                                  "Default 15 (measured; see the command help).")
            sub.add_argument("--fix-rotation", action="store_true",
                             help="Force rotation to identity while tracking. "
                                  "Measured WORSE (translation error 2-3x): prefer "
                                  "dropping rotation when building the target.")
        if name == "build":
            sub.add_argument("--from", dest="start_from", choices=STAGE_ORDER,
                             help="Start here and run everything after it, even if "
                                  "those stages already look done.")
            sub.add_argument("--only", choices=STAGE_ORDER,
                             help="Run exactly this one stage.")
            sub.add_argument("--force", action="store_true",
                             help="Redo every stage from the start.")
            sub.add_argument("--dry-run", action="store_true",
                             help="Print which stages would run, then stop.")
            sub.add_argument("--no-overlay", action="store_true",
                             help="Skip the review overlay video in `reduce`. It is "
                                  "how you check the pose tracks the object, so skip "
                                  "it only when re-running a trajectory you trust.")
        if name == "store":
            sub.add_argument("--trajectory", help="trajectory.npz to place. "
                                                  "Default: <work-dir>/run/trajectory.npz")
            placement = sub.add_mutually_exclusive_group()
            placement.add_argument("--offset", type=int, default=0,
                                   help="Dataset frame that the trajectory's frame 0 "
                                        "is. Default 0, i.e. a whole-video run.")
            placement.add_argument("--episode", type=int,
                                   help="Trajectory came from this episode's clip; "
                                        "its start frame becomes the offset.")
            sub.add_argument("--out", help="Store directory. Default: <work-dir>/store")
        sub.set_defaults(handler=handler)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
