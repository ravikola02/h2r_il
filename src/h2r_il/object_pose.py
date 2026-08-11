#!/usr/bin/env python
"""Monocular 6-DoF object pose from a demo video, in the camera optical frame.

The pipeline itself is **not implemented here**. It is sourced from the
video_to_data (v2d) reconstruction modules -- `v2d.pipelines.run_video_object_tracking`
-- which chains SAM2 masks -> MoGe depth -> SAM3D mesh -> metric scale ->
FoundationPose tracking -> mesh-overlay renders. No CAD model and no depth sensor:
the mesh comes from SAM3D and the metric scale from MoGe depth.

v2d is NOT vendored into this repo. It stays a separate checkout on the shared mount
(`V2D_ROOT`, see LOCAL_SETUP.md) whose host packages are installed editable into this
env, so `import v2d` resolves into that checkout. Only orchestration runs here; every
model runs inside v2d's own docker images. This module adds what the raw pipeline does
not give you: run identity, a single trajectory array, and a reviewable overlay video.

Two outputs, both first-class:
  * 6-DoF pose  -> trajectory.npz (+ trajectory.json summary)
  * overlay     -> overlay.mp4, v2d's mesh renders with the pose axes drawn on top

Output frame: object -> camera in the OpenCV **optical** frame (x right, y down,
z forward, metres), which is what FoundationPose and MoGe natively produce -- no
conversion is applied by default. `--frame ros` re-expresses the saved trajectory in
REP-103 body axes (x forward, y left, z up); the overlay is always drawn from the
optical-frame poses, since that is the frame the intrinsics project in.

Usage (runs in this repo's uv env -- v2d's host packages are installed into it):

    # 1. First run on a clip: no prompts JSON yet, so the SAM2 annotation UI opens
    #    on localhost:8080, then it asks before spending GPU time
    uv run --no-sync python -m h2r_il.object_pose --video clip.mp4

    # 2. Later runs reuse <video_dir>/<stem>_prompts.json and go straight to tracking;
    #    pass --annotate to reopen the UI and edit them
    uv run --no-sync python -m h2r_il.object_pose --video clip.mp4 --out outputs/object_pose/clip

    # 3. Rebuild trajectory + overlay from an existing run, no GPU work
    uv run --no-sync python -m h2r_il.object_pose --video clip.mp4 --reduce-only
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

REPO_DIR = Path(__file__).resolve().parents[2]

# Subdirectories v2d writes under the run's output directory.
POSES_SUBDIR = "poses"
INTRINSICS_SUBDIR = "intrinsics"
RENDERS_SUBDIR = "renders"
FRAMES_SUBDIR = "frames"

# Weights live in the v2d checkout (~10 GB, shared), addressed relative to
# <V2D_ROOT>/reconstruction the same way v2d's own scripts do.
WEIGHTS = {
    "sam2": "data/weights/sam2",
    "moge": "data/weights/moge",
    "sam3d": "data/weights/sam3d",
    "foundation_pose": "data/weights/foundation_pose",
}

# OpenCV optical (x right, y down, z forward) -> REP-103 body (x fwd, y left, z up).
R_ROS_FROM_OPTICAL = np.array([[0.0, 0.0, 1.0],
                               [-1.0, 0.0, 0.0],
                               [0.0, -1.0, 0.0]])

# Default crop, derived from the object's travel across the kitting head-camera
# clips: it roughly doubles the object's share of the frame (0.49% -> 0.96%), which
# is what kept FoundationPose locked on 65 frames longer. It is specific to that
# framing -- hence DEFAULT_CROP_FOR, which triggers a warning on any other size.
# Pass `--crop none` to disable.
DEFAULT_CROP = "108,61,1136,518"
DEFAULT_CROP_FOR = (1280, 720)

# Overlay axis triad: BGR, so x=red, y=green, z=blue.
AXIS_COLORS = ((0, 0, 255), (0, 255, 0), (255, 0, 0))
AXIS_LENGTH_M = 0.05


# --------------------------------------------------------------------------------
# Locating v2d
# --------------------------------------------------------------------------------

def read_local_env(key: str) -> str | None:
    """Pull one key out of the gitignored configs/local.env.

    That file is bash, but every entry is a plain assignment or the `${VAR-default}`
    form, so a targeted regex beats spawning a shell. Machine paths stay out of
    tracked files (LOCAL_SETUP.md); this only reads them back.
    """
    local_env = REPO_DIR / "configs" / "local.env"
    if not local_env.is_file():
        return None
    pattern = re.compile(rf"^\s*{re.escape(key)}=(?:\$\{{{re.escape(key)}-)?([^}}\n#]*)")
    for line in local_env.read_text().splitlines():
        match = pattern.match(line)
        if match:
            return match.group(1).strip().strip('"').strip("'") or None
    return None


def resolve_v2d_root() -> Path:
    """Find the v2d checkout: $V2D_ROOT, else configs/local.env, else give up loudly."""
    raw = os.environ.get("V2D_ROOT") or read_local_env("V2D_ROOT")
    if not raw:
        sys.exit(
            "V2D_ROOT is not set.\n"
            "Add it to the gitignored configs/local.env, e.g.\n"
            "    V2D_ROOT=${V2D_ROOT-/path/to/video_to_data}\n"
            "It must point at a video_to_data checkout, not a copy inside this repo."
        )
    root = Path(raw).expanduser().resolve()
    if not (root / "reconstruction" / "modules").is_dir():
        sys.exit(f"V2D_ROOT={root} does not look like a video_to_data checkout.")
    return root


def import_v2d(v2d_root: Path):
    """Import the v2d entry points and prove they came from V2D_ROOT.

    The host packages are installed editable, so `import v2d` normally resolves
    straight into the shared checkout. Checking it here turns the confusing failure
    modes -- a `uv sync` having dropped the editables, or a stray second copy --
    into one line at startup instead of a docker error twenty minutes in.
    """
    try:
        import v2d.common.datatypes as datatypes
        from v2d.pipelines.run_video_object_tracking import run_video_object_tracking
    except ImportError as exc:
        sys.exit(
            f"Cannot import v2d ({exc}).\n"
            "The v2d host packages are missing from this env. Reinstall them "
            "(additive, --no-deps keeps this env's own pins untouched):\n"
            f"    V=$V2D_ROOT/reconstruction/modules\n"
            "    uv pip install --no-deps -e $V/v2d_common -e $V/v2d_docker \\\n"
            "        -e $V/v2d_pipelines -e $V/v2d_sam2/docker -e $V/v2d_moge/docker \\\n"
            "        -e $V/v2d_sam3d/docker -e $V/v2d_mesh/docker \\\n"
            "        -e $V/v2d_foundation_pose/docker\n"
            "Note a plain `uv sync` drops them again; this repo's scripts use --no-sync."
        )
    resolved = Path(datatypes.__file__).resolve()
    if v2d_root not in resolved.parents:
        sys.exit(
            f"v2d resolves to {resolved}, which is outside V2D_ROOT={v2d_root}.\n"
            "Two checkouts are in play; fix the env or V2D_ROOT before trusting a run."
        )
    return datatypes.Transform3d, run_video_object_tracking


# --------------------------------------------------------------------------------
# Run identity
# --------------------------------------------------------------------------------

def fingerprint(video: Path, prompts: Path, object_id: int, reference_frame: int,
                simplify_factor: float) -> dict:
    """Everything that changes the poses. Stored so a stale reuse can be detected.

    v2d's pipeline is a straight line that overwrites per-stage artifacts in place and
    keeps no record of what produced them, so re-running with different prompts leaves
    an output directory holding a *mix* of two runs -- new masks and mesh beside old
    poses. That reprojects wrong while looking entirely plausible. Hence the manifest.
    """
    stat = video.stat()
    return {
        "video": str(video),
        "video_size": stat.st_size,
        "video_mtime": int(stat.st_mtime),
        "prompts_sha1": hashlib.sha1(prompts.read_bytes()).hexdigest(),
        "object_id": object_id,
        "reference_frame": reference_frame,
        "simplify_factor": simplify_factor,
    }


def v2d_revision(v2d_root: Path) -> str:
    """Short git description of the v2d checkout, for the manifest. Best effort."""
    try:
        out = subprocess.run(
            ["git", "-C", str(v2d_root), "describe", "--always", "--dirty"],
            capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip() if out.returncode == 0 else "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def check_manifest(out_dir: Path, current: dict, force: bool) -> None:
    """Refuse to reuse an output directory that was produced by a different run."""
    manifest_path = out_dir / "run_manifest.json"
    if not manifest_path.is_file():
        return
    previous = json.loads(manifest_path.read_text()).get("fingerprint", {})
    if previous == current:
        return
    changed = [k for k in current if previous.get(k) != current.get(k)]
    message = (
        f"{out_dir} holds a previous run that differs in: {', '.join(changed)}.\n"
        "Re-running would mix new masks/mesh with old poses. Use a fresh --out, "
        "or pass --force to overwrite in place."
    )
    if not force:
        sys.exit(message)
    print(f"[warn] {message}\n[warn] --force given; overwriting.")


# --------------------------------------------------------------------------------
# Prompts
# --------------------------------------------------------------------------------

def load_prompts(prompts_path: Path) -> list[dict]:
    return json.loads(prompts_path.read_text())["prompts"]


def describe_prompt(prompt: dict) -> str:
    """One-line summary of a single SAM2 prompt, for the confirmation listing."""
    box = prompt.get("box")
    shape = (f"box ({box['x0']:.0f},{box['y0']:.0f})-({box['x1']:.0f},{box['y1']:.0f})"
             if box else "points only")
    return f"object_id={prompt['object_id']}  frame={prompt['frame_index']}  {shape}"


def _ffmpeg() -> str:
    """A real ffmpeg, never the snap one.

    `/usr/local/bin/ffmpeg` on this box symlinks to `/snap/bin/ffmpeg`, which is
    confined and cannot write to `/mnt` -- and it comes first on PATH. Resolving the
    symlink and rejecting anything under /snap avoids a failure that looks like a
    permissions bug in this script.
    """
    for candidate in ("/usr/bin/ffmpeg", shutil.which("ffmpeg") or ""):
        if candidate and os.path.exists(candidate):
            if "snap" not in os.path.realpath(candidate):
                return candidate
    sys.exit("no non-snap ffmpeg found; install it with `apt install ffmpeg`.")


def crop_from_prompts(prompts: list[dict], margin: float,
                      width: int, height: int) -> tuple[int, int, int, int]:
    """Union of every annotated box, padded by `margin` of its own size."""
    boxes = [p["box"] for p in prompts if p.get("box")]
    if not boxes:
        sys.exit("--crop auto needs boxes in the prompts file; found none.")
    x0 = min(b["x0"] for b in boxes)
    y0 = min(b["y0"] for b in boxes)
    x1 = max(b["x1"] for b in boxes)
    y1 = max(b["y1"] for b in boxes)
    pad_x, pad_y = (x1 - x0) * margin, (y1 - y0) * margin
    return clamp_crop((x0 - pad_x, y0 - pad_y, x1 + pad_x, y1 + pad_y), width, height)


def clamp_crop(box: tuple[float, float, float, float],
               width: int, height: int) -> tuple[int, int, int, int]:
    """Clip to the frame and force even width/height (yuv420p needs even dims)."""
    x0 = max(0, int(box[0]))
    y0 = max(0, int(box[1]))
    x1 = min(width, int(box[2]))
    y1 = min(height, int(box[3]))
    if x1 - x0 < 16 or y1 - y0 < 16:
        sys.exit(f"crop {(x0, y0, x1, y1)} is degenerate for a {width}x{height} video.")
    return x0, y0, x0 + (x1 - x0) // 2 * 2, y0 + (y1 - y0) // 2 * 2


def make_cropped_clip(video: Path, box: tuple[int, int, int, int], dest: Path) -> Path:
    """Re-encode the crop to its own clip. Cheap and idempotent enough to just redo.

    Re-encoding rather than stream-copying keeps frame indices aligned with the
    source: a copy would snap the start to the nearest keyframe.
    """
    x0, y0, x1, y1 = box
    cmd = [_ffmpeg(), "-y", "-loglevel", "error", "-i", str(video),
           "-vf", f"crop={x1 - x0}:{y1 - y0}:{x0}:{y0}",
           "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", "-an", str(dest)]
    print(f"Cropping to {x1 - x0}x{y1 - y0} at ({x0},{y0}) -> {dest.name}")
    subprocess.run(cmd, check=True)
    return dest


def _host_address() -> str:
    """This host's address as the SSH client sees it, for the port-forward hint.

    SSH_CONNECTION is "<client ip> <client port> <server ip> <server port>", so the
    third field is the address the client already reached us on -- better than
    guessing from the hostname, which may not resolve on the client's side.
    """
    parts = os.environ.get("SSH_CONNECTION", "").split()
    return parts[2] if len(parts) >= 3 else "<this-host>"


def annotate(video: Path, prompts_path: Path, port: int) -> None:
    """Open v2d's SAM2 annotation UI and block until you stop it with Ctrl-C.

    The UI has no 'done' button: it writes the JSON on every add and delete, so the
    file on disk is always current and Ctrl-C loses nothing. Ctrl-C reaches both the
    container (docker run then exits non-zero) and this process, so the resulting
    CalledProcessError and KeyboardInterrupt are the normal way out, not failures.
    """
    from v2d.sam2.docker.run_annotate import run_annotate

    action = "Editing" if prompts_path.is_file() else "Creating"
    print(f"{action} {prompts_path}")
    print(f"Annotation UI -> http://localhost:{port}")
    # Over SSH, "localhost" is the *client's* localhost, where nothing is listening:
    # the server binds 0.0.0.0 on this host. Say so, because the failure is silent --
    # the browser simply never reaches it and the server log stays empty.
    if os.environ.get("SSH_CONNECTION"):
        print(f"  You are on SSH. From your machine, forward the port first:\n"
              f"      ssh -L {port}:localhost:{port} {os.environ.get('USER', 'user')}@{_host_address()}\n"
              f"  and open http://localhost:{port} there. Use a SECOND terminal --\n"
              f"  closing this session kills the annotator.")
    print("Box the object on a frame where it is unoccluded and sharp.")
    print("Every edit is saved immediately. Press Ctrl-C here when done.\n")
    try:
        run_annotate(video_path=str(video), prompts_path=str(prompts_path), port=port)
    except KeyboardInterrupt:
        pass
    except subprocess.CalledProcessError as exc:
        # 130 SIGINT, 143 SIGTERM, 137 SIGKILL: the server was stopped, not broken.
        if exc.returncode not in (130, 137, 143):
            raise
    print("\nAnnotation server stopped.")


def confirm_continue(prompts_path: Path) -> bool:
    """Summarize what was annotated and confirm before spending GPU time.

    Ctrl-C out of the UI means both 'I am finished' and 'abort everything', so ask
    rather than guess. Non-interactive callers proceed -- they cannot answer, and
    they only reach this point by passing --annotate deliberately.
    """
    if not prompts_path.is_file():
        print(f"No prompts written to {prompts_path}; nothing to run.")
        return False

    prompts = load_prompts(prompts_path)
    print(f"Saved {len(prompts)} prompt(s) to {prompts_path}:")
    for prompt in prompts:
        print(f"  {describe_prompt(prompt)}")

    if not sys.stdin.isatty():
        return True
    return input("\nContinue to the tracking pipeline? [y/N] ").strip().lower() in ("y", "yes")


def resolve_object_id(prompts: list[dict], requested: int | None) -> int:
    """Explicit id wins; auto-select only when the prompts name exactly one object.

    Resolved before the pipeline starts and never allowed to stay None: the pipeline
    interpolates it into a mask path, so a None would surface as a missing
    masks/None/ directory only after segmentation and depth have already run.
    """
    ids = sorted({p["object_id"] for p in prompts})
    if requested is not None:
        if requested not in ids:
            sys.exit(f"--object-id {requested} is not in the prompts (have: {ids}).")
        return requested
    if len(ids) != 1:
        sys.exit(f"prompts contain {len(ids)} object ids {ids}; pass --object-id.")
    return ids[0]


def resolve_reference_frame(prompts: list[dict], object_id: int,
                            requested: int | None) -> int:
    """Explicit frame wins, else the first annotated frame for this object.

    The reference frame drives the SAM3D mesh, the metric-scale solve and the
    FoundationPose registration, so it wants to be a frame where the object is
    unoccluded -- which is exactly the frame you chose to annotate.
    """
    if requested is not None:
        return requested
    frames = [p["frame_index"] for p in prompts if p["object_id"] == object_id]
    if not frames:
        sys.exit(f"no prompts for object_id={object_id}; cannot pick a reference frame.")
    return min(frames)


# --------------------------------------------------------------------------------
# Reduction: per-frame pose JSONs -> one trajectory
# --------------------------------------------------------------------------------

def load_intrinsics(out_dir: Path, reference_frame: int) -> dict:
    """The reference frame's intrinsics -- the ones FoundationPose actually tracked with.

    MoGe estimates intrinsics per frame, but the pipeline hands FP only the reference
    frame's, so that is the single camera model the poses are consistent with.
    """
    path = out_dir / INTRINSICS_SUBDIR / f"{reference_frame:06d}.json"
    if not path.is_file():
        sys.exit(f"missing intrinsics for the reference frame: {path}")
    return json.loads(path.read_text())


def load_poses(out_dir: Path, transform3d_cls) -> tuple[np.ndarray, np.ndarray]:
    """Read poses/*.json in frame order -> (frame_index[N], T_cam_obj[N,4,4]).

    Parsing goes through v2d's own Transform3d rather than reading the quaternion
    directly: the JSON stores it **wxyz**, which is the opposite of scipy's default,
    and sourcing the conversion keeps this file correct if v2d ever changes it.
    """
    pose_dir = out_dir / POSES_SUBDIR
    pose_files = sorted(pose_dir.glob("*.json"), key=lambda p: int(p.stem))
    if not pose_files:
        sys.exit(f"no pose JSONs in {pose_dir}; did the pipeline finish?")

    frames, matrices = [], []
    for path in pose_files:
        transform = transform3d_cls.load(str(path))
        # to_matrix() folds scale into the rotation block. FoundationPose emits unit
        # scale (the metric scale is already baked into the mesh), but if that ever
        # changes, the 3x3 stops being a rotation and every rpy below is garbage.
        scale = np.asarray(transform.scale, dtype=float)
        if not np.allclose(scale, 1.0, atol=1e-6):
            sys.exit(f"{path} has non-unit scale {scale.tolist()}; refusing to "
                     "extract a rotation from a scaled matrix.")
        frames.append(int(path.stem))
        matrices.append(transform.to_matrix())
    return np.asarray(frames, dtype=np.int64), np.asarray(matrices, dtype=np.float64)


def _ema_forward(matrices: np.ndarray, alpha: float) -> np.ndarray:
    """One causal EMA pass: s_t = alpha*x_t + (1-alpha)*s_{t-1}.

    Position is a plain EMA. Rotation cannot be: averaging quaternions
    componentwise is not a rotation, and the sign ambiguity of the double cover
    makes it worse. Instead each step slerps from the running estimate toward the
    measurement, by scaling the rotation vector of the delta -- which is geodesic
    interpolation and takes the shortest path automatically.
    """
    positions = matrices[:, :3, 3]
    rotations = Rotation.from_matrix(matrices[:, :3, :3])

    out_pos = np.empty_like(positions)
    out_rot = [rotations[0]]
    out_pos[0] = positions[0]

    for i in range(1, len(matrices)):
        out_pos[i] = alpha * positions[i] + (1.0 - alpha) * out_pos[i - 1]
        delta = rotations[i] * out_rot[i - 1].inv()
        out_rot.append(Rotation.from_rotvec(delta.as_rotvec() * alpha) * out_rot[i - 1])

    smoothed = np.repeat(np.eye(4)[None], len(matrices), axis=0)
    smoothed[:, :3, :3] = Rotation.concatenate(out_rot).as_matrix()
    smoothed[:, :3, 3] = out_pos
    return smoothed


def smooth_poses(matrices: np.ndarray, strength: float, zero_phase: bool) -> np.ndarray:
    """Exponentially smooth a pose trajectory.

    `strength` runs 0 to 1: **0 is no smoothing, 1 is maximum**. It maps to the
    EMA's usual alpha (the weight on each new measurement) as ``alpha = 1 -
    strength``, so the knob reads the way its name does. strength=1 gives
    alpha=0, which holds the first pose forever -- mathematically the limit of
    infinite smoothing, and useless in practice; stay well below it.

    Causal (default) only ever pulls a frame toward its past, which matters when a
    diverged tail would otherwise be dragged backwards into the good frames.
    `zero_phase` runs the filter forwards then backwards to cancel the lag a
    single pass introduces -- measured on this data it kept the reprojection
    error at raw levels (10.1 -> 11.0 px) while halving jitter, where a causal
    pass at the same strength tripled the error, because the object moves fast
    enough (~47 cm/s) that lag becomes error.
    """
    if not 0.0 <= strength <= 1.0:
        sys.exit(f"--smooth must be in [0, 1] (0 = off, 1 = maximum), got {strength}")
    alpha = 1.0 - strength
    if strength == 0.0 or len(matrices) < 2:
        return matrices
    smoothed = _ema_forward(matrices, alpha)
    if zero_phase:
        smoothed = _ema_forward(smoothed[::-1], alpha)[::-1]
    return smoothed


def to_ros_frame(matrices: np.ndarray) -> np.ndarray:
    """Re-express object->camera poses in REP-103 body axes.

    Left-multiplication only: this rotates the frame the pose is *expressed in*.
    The object's own body frame is whatever SAM3D produced and is left untouched.
    """
    change = np.eye(4)
    change[:3, :3] = R_ROS_FROM_OPTICAL
    return change @ matrices


def write_trajectory(out_dir: Path, frames: np.ndarray, matrices: np.ndarray,
                     intrinsics: dict, manifest: dict, frame_name: str) -> Path:
    """Write trajectory.npz (machine) + trajectory.json (human)."""
    rotations = Rotation.from_matrix(matrices[:, :3, :3])
    quat_xyzw = rotations.as_quat()
    position = matrices[:, :3, 3]

    npz_path = out_dir / "trajectory.npz"
    np.savez(
        npz_path,
        frame_index=frames,
        T_cam_obj=matrices,
        position=position,
        quat_xyzw=quat_xyzw,
        quat_wxyz=quat_xyzw[:, [3, 0, 1, 2]],
        # Intrinsic XYZ euler, matching the convention used in eval_openloop.py.
        rpy=rotations.as_euler("XYZ"),
        intrinsics=np.array([intrinsics["fx"], intrinsics["fy"],
                             intrinsics["cx"], intrinsics["cy"]]),
        image_size=np.array([intrinsics["width"], intrinsics["height"]]),
    )

    # Gaps mean FoundationPose dropped frames; the caller needs to know before
    # treating frame_index as a dense 0..N-1 range.
    expected = np.arange(frames[0], frames[-1] + 1)
    missing = sorted(set(expected.tolist()) - set(frames.tolist()))
    summary = {
        "frame_convention": frame_name,
        "pose_convention": "object -> camera, metres",
        "pipeline": "v2d.pipelines.run_video_object_tracking (sourced from V2D_ROOT)",
        "n_poses": int(frames.size),
        "frame_range": [int(frames[0]), int(frames[-1])],
        "missing_frames": missing,
        "intrinsics": intrinsics,
        "position_min": position.min(axis=0).round(4).tolist(),
        "position_max": position.max(axis=0).round(4).tolist(),
        "path_length_m": round(float(np.linalg.norm(np.diff(position, axis=0), axis=1).sum()), 4),
        **manifest,
    }
    (out_dir / "trajectory.json").write_text(json.dumps(summary, indent=2))
    return npz_path


# --------------------------------------------------------------------------------
# Overlay
# --------------------------------------------------------------------------------

def project(points_cam: np.ndarray, intrinsics: dict) -> np.ndarray:
    """Pinhole-project camera-frame points to pixels. Optical frame: +z is forward."""
    z = np.clip(points_cam[:, 2], 1e-6, None)
    u = intrinsics["fx"] * points_cam[:, 0] / z + intrinsics["cx"]
    v = intrinsics["fy"] * points_cam[:, 1] / z + intrinsics["cy"]
    return np.stack([u, v], axis=1)


def draw_pose_axes(image: np.ndarray, matrix: np.ndarray, intrinsics: dict) -> None:
    """Draw the object's body axes at its tracked pose, in place.

    Takes an **optical-frame** object->camera matrix, because that is the frame the
    intrinsics project in; a --frame ros trajectory would land somewhere arbitrary.
    """
    origin_and_axes = np.array([[0, 0, 0],
                                [AXIS_LENGTH_M, 0, 0],
                                [0, AXIS_LENGTH_M, 0],
                                [0, 0, AXIS_LENGTH_M]], dtype=np.float64)
    points_cam = (matrix[:3, :3] @ origin_and_axes.T).T + matrix[:3, 3]
    if points_cam[0, 2] <= 0:  # object behind the camera: nothing sensible to draw
        return
    pixels = project(points_cam, intrinsics).astype(int)
    origin = tuple(pixels[0])
    for axis_pixel, color in zip(pixels[1:], AXIS_COLORS):
        cv2.line(image, origin, tuple(axis_pixel), color, 3, cv2.LINE_AA)
    cv2.circle(image, origin, 5, (255, 255, 255), -1, cv2.LINE_AA)


def draw_label(image: np.ndarray, text: str) -> None:
    """Draw the pose readout on a filled box, in place.

    A single putText pass over a solid box, not an outline pass plus a fill pass:
    OpenCV's Hershey glyph advance widens with `thickness`, so two passes at
    different thicknesses drift apart across the string and the label reads doubled.
    """
    font, scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
    (width, height), baseline = cv2.getTextSize(text, font, scale, thickness)
    origin = (12, 16 + height)
    cv2.rectangle(image,
                  (origin[0] - 8, origin[1] - height - 8),
                  (origin[0] + width + 8, origin[1] + baseline + 4),
                  (0, 0, 0), -1)
    cv2.putText(image, text, origin, font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def source_frames(out_dir: Path) -> tuple[Path, str]:
    """Prefer v2d's mesh-overlay renders; fall back to the raw extracted frames.

    The renders already composite the tracked mesh over the video, which is the
    strongest visual check that the pose is locked on. If step 9 did not run, the
    axes alone still show the trajectory.
    """
    renders = out_dir / RENDERS_SUBDIR
    if renders.is_dir() and any(renders.glob("*.png")):
        return renders, "mesh renders + pose axes"
    frames = out_dir / FRAMES_SUBDIR
    if frames.is_dir() and any(frames.glob("*.png")):
        return frames, "raw frames + pose axes (no mesh renders found)"
    sys.exit(f"no renders/ or frames/ under {out_dir}; nothing to overlay.")


def write_overlay_video(out_dir: Path, frames: np.ndarray, matrices_optical: np.ndarray,
                        intrinsics: dict, fps: float) -> tuple[Path, str]:
    """Compose an overlay.mp4: per-frame image + pose axes + a numeric readout."""
    image_dir, description = source_frames(out_dir)
    pose_by_frame = {int(f): m for f, m in zip(frames, matrices_optical)}

    image_paths = sorted(image_dir.glob("*.png"), key=lambda p: int(p.stem))
    first = cv2.imread(str(image_paths[0]))
    height, width = first.shape[:2]

    video_path = out_dir / "overlay.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (width, height))
    if not writer.isOpened():
        sys.exit(f"cannot open {video_path} for writing (missing mp4v codec?)")

    try:
        for path in image_paths:
            index = int(path.stem)
            image = cv2.imread(str(path))
            matrix = pose_by_frame.get(index)
            if matrix is not None:
                draw_pose_axes(image, matrix, intrinsics)
                x, y, z = matrix[:3, 3]
                roll, pitch, yaw = Rotation.from_matrix(matrix[:3, :3]).as_euler("XYZ", degrees=True)
                label = (f"f{index:05d}  xyz {x:+.3f} {y:+.3f} {z:+.3f} m  "
                         f"rpy {roll:+6.1f} {pitch:+6.1f} {yaw:+6.1f} deg")
            else:
                label = f"f{index:05d}  (no pose)"
            draw_label(image, label)
            writer.write(image)
    finally:
        writer.release()
    return video_path, description


def probe_size(video: Path) -> tuple[int, int]:
    """(width, height) of the source video, for clamping a crop to the frame."""
    capture = cv2.VideoCapture(str(video))
    try:
        return (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    finally:
        capture.release()


def probe_fps(video: Path) -> float:
    """Frame rate of the source video, so the overlay plays at real speed."""
    capture = cv2.VideoCapture(str(video))
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
    finally:
        capture.release()
    return fps if fps and fps > 0 else 30.0


# --------------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m h2r_il.object_pose",
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--video", required=True, help="Input demo video.")
    parser.add_argument("--prompts", help="SAM2 prompts JSON. "
                                          "Default: <video_dir>/<stem>_prompts.json")
    parser.add_argument("--out", help="Output dir. Default: outputs/object_pose/<stem>")
    parser.add_argument("--annotate", action="store_true",
                        help="Open the SAM2 annotation UI first, then track. Implied "
                             "when the prompts JSON does not exist yet; pass it "
                             "explicitly to edit prompts that already do.")
    parser.add_argument("--port", type=int, default=8080, help="Annotation UI port.")
    parser.add_argument("--object-id", type=int, default=None,
                        help="Which object_id to track. Inferred if the prompts hold one.")
    parser.add_argument("--reference-frame", type=int, default=None,
                        help="Frame for the mesh, scale solve and registration. "
                             "Default: first annotated frame for this object.")
    parser.add_argument("--simplify-factor", type=float, default=0.5,
                        help="Fraction of mesh faces to keep (0.0-1.0).")
    parser.add_argument("--crop", metavar="X0,Y0,X1,Y1", default=DEFAULT_CROP,
                        help="Crop the video before annotating and tracking, so a small "
                             "object survives each model's input downscale. Explicit pixel "
                             "bounds, 'auto' for the union of an existing prompts file's "
                             f"boxes, or 'none' to disable. Default: {DEFAULT_CROP} "
                             f"(tuned for {DEFAULT_CROP_FOR[0]}x{DEFAULT_CROP_FOR[1]} "
                             "head-camera clips).")
    parser.add_argument("--crop-from", metavar="PROMPTS.json",
                        help="Prompts file for --crop auto. Default: the source video's.")
    parser.add_argument("--crop-margin", type=float, default=0.15,
                        help="Padding around the box union for --crop auto (default 0.15).")
    parser.add_argument("--frame", choices=("optical", "ros"), default="optical",
                        help="Axes for the saved trajectory: OpenCV optical (default) "
                             "or REP-103 body. The overlay is always optical.")
    parser.add_argument("--reduce-only", action="store_true",
                        help="Skip the pipeline; rebuild trajectory + overlay from poses/.")
    parser.add_argument("--smooth", type=float, metavar="STRENGTH", default=0.0,
                        help="Exponentially smooth the poses. STRENGTH is 0 to 1: "
                             "0 = off (default), 0.5 = moderate, 1 = maximum (degenerate, "
                             "holds a single pose). Position by EMA, rotation by geodesic "
                             "slerp.")
    parser.add_argument("--smooth-zero-phase", action="store_true",
                        help="Run --smooth forwards then backwards to cancel lag. Prefer "
                             "it when the object moves fast; skip it when a diverged "
                             "segment could bleed backwards into good frames.")
    parser.add_argument("--no-overlay", action="store_true",
                        help="Skip overlay.mp4 (the pose outputs are still written).")
    parser.add_argument("--force", action="store_true",
                        help="Reuse an output dir whose manifest does not match.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    v2d_root = resolve_v2d_root()
    transform3d_cls, run_pipeline = import_v2d(v2d_root)

    # Resolve every user path to absolute *before* the chdir below, so relative
    # arguments keep meaning what they meant in the shell that typed them.
    video = Path(args.video).expanduser().resolve()
    if not video.is_file():
        sys.exit(f"no such video: {video}")
    # Crop before anything else, so the annotator, every model and the poses all see
    # the same frame. Cropping does not add pixels to the object, but MoGe, SAM3D and
    # FoundationPose each resize the *whole* frame to their input resolution -- a
    # tighter frame is what stops a small object being downscaled into nothing.
    crop_box = None
    if args.crop and args.crop.strip().lower() not in ("none", "off", ""):
        source_width, source_height = probe_size(video)
        # The default box is tuned to one framing. Applied to a different resolution
        # it would crop a geometrically meaningless region, and clamping would hide
        # that -- so say it out loud rather than producing quietly wrong poses.
        if args.crop == DEFAULT_CROP and (source_width, source_height) != DEFAULT_CROP_FOR:
            print(f"[warn] default crop {DEFAULT_CROP} was tuned for "
                  f"{DEFAULT_CROP_FOR[0]}x{DEFAULT_CROP_FOR[1]}, but this video is "
                  f"{source_width}x{source_height}. Pass explicit bounds, '--crop auto', "
                  "or '--crop none'.")
        if args.crop == "auto":
            auto_from = (Path(args.crop_from).expanduser().resolve() if args.crop_from
                         else video.parent / f"{video.stem}_prompts.json")
            if not auto_from.is_file():
                sys.exit(f"--crop auto needs an existing prompts file; {auto_from} not "
                         "found. Pass --crop-from, or give explicit X0,Y0,X1,Y1.")
            crop_box = crop_from_prompts(load_prompts(auto_from), args.crop_margin,
                                         source_width, source_height)
            print(f"Crop derived from {auto_from.name}")
        else:
            try:
                values = [float(v) for v in args.crop.split(",")]
                if len(values) != 4:
                    raise ValueError
            except ValueError:
                sys.exit(f"--crop expects 'X0,Y0,X1,Y1' or 'auto', got {args.crop!r}")
            crop_box = clamp_crop(tuple(values), source_width, source_height)

        cropped = video.parent / f"{video.stem}_crop.mp4"
        video = make_cropped_clip(video, crop_box, cropped)

    prompts_path = (Path(args.prompts).expanduser().resolve() if args.prompts
                    else video.parent / f"{video.stem}_prompts.json")
    out_dir = (Path(args.out).expanduser().resolve() if args.out
               else REPO_DIR / "outputs" / "object_pose" / video.stem)
    out_dir.mkdir(parents=True, exist_ok=True)

    # v2d resolves its weights relative to reconstruction/ and mounts them into the
    # containers from there, so the pipeline has to be driven from that directory.
    # Everything above is already absolute, so this only affects v2d's own paths.
    os.chdir(v2d_root / "reconstruction")

    # The pipeline cannot start without prompts, so a missing JSON is a request to
    # annotate, not an error -- but only when someone is there to drive the UI and
    # Ctrl-C out of it. Headless, that would block on a web server forever, so say
    # what is missing instead. --annotate forces the UI either way, which is also
    # how you edit prompts that already exist.
    needs_prompts = not prompts_path.is_file()
    if not args.reduce_only and (args.annotate or needs_prompts):
        if needs_prompts and not args.annotate:
            if not sys.stdin.isatty():
                sys.exit(f"no prompts at {prompts_path}, and stdin is not a terminal "
                         "so the annotation UI cannot be driven. Annotate first with "
                         "--annotate from a terminal, or pass --prompts.")
            print(f"No prompts at {prompts_path} -- opening the annotator first.\n")
        annotate(video, prompts_path, args.port)
        if not confirm_continue(prompts_path):
            return 0

    # Only reachable under --reduce-only, which skips the annotator above: the
    # prompts still decide which object_id and reference frame the existing poses
    # belong to, so they are required even when no GPU work will run.
    if not prompts_path.is_file():
        sys.exit(f"no prompts at {prompts_path}; --reduce-only still needs them to "
                 "resolve the object id and reference frame. Pass --prompts.")

    prompts = load_prompts(prompts_path)
    object_id = resolve_object_id(prompts, args.object_id)
    reference_frame = resolve_reference_frame(prompts, object_id, args.reference_frame)

    manifest = {
        "fingerprint": fingerprint(video, prompts_path, object_id, reference_frame,
                                   args.simplify_factor),
        "v2d_root": str(v2d_root),
        "v2d_revision": v2d_revision(v2d_root),
    }
    if crop_box is not None:
        # Poses are in the cropped camera's optical frame. The camera did not move, so
        # the object's 3D position is unchanged in principle -- but MoGe re-estimates
        # the focal length from the cropped image, so absolute metric scale can differ
        # from an uncropped run. Record the offset so pixel coordinates can be mapped
        # back onto the original frames: u_orig = u_crop + crop_x0.
        manifest["crop"] = {"x0": crop_box[0], "y0": crop_box[1],
                            "x1": crop_box[2], "y1": crop_box[3],
                            "note": "intrinsics are the crop's; MoGe re-estimates focal"}

    if args.reduce_only:
        # Describe what actually produced the poses, not what the current arguments
        # would produce. Recomputing here would stamp trajectory.json with a
        # fingerprint that never ran -- exactly the provenance the manifest exists
        # to protect. A directory with no manifest predates it and is unverifiable.
        manifest_path = out_dir / "run_manifest.json"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text())
        else:
            manifest = {"provenance": "unknown: no run_manifest.json in this output dir"}
            print(f"[warn] {out_dir} has no run_manifest.json; the poses there may not "
                  "match the arguments given. Recorded as provenance: unknown.")
    else:
        check_manifest(out_dir, manifest["fingerprint"], args.force)
        print(f"Tracking object_id={object_id} from reference frame {reference_frame}")
        print(f"  video  {video}\n  out    {out_dir}\n")
        run_pipeline(
            video_path=str(video),
            prompts_path=str(prompts_path),
            object_id=object_id,
            output_dir=str(out_dir),
            sam2_weights=WEIGHTS["sam2"],
            moge_weights=WEIGHTS["moge"],
            sam3d_weights=WEIGHTS["sam3d"],
            fp_weights=WEIGHTS["foundation_pose"],
            reference_frame=reference_frame,
            simplify_factor=args.simplify_factor,
        )
        # Written only on success, so a crashed run cannot be mistaken for a good one.
        (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))

    frames, matrices_optical = load_poses(out_dir, transform3d_cls)
    intrinsics = load_intrinsics(out_dir, reference_frame)

    if args.smooth > 0.0:
        raw = matrices_optical
        matrices_optical = smooth_poses(raw, args.smooth, args.smooth_zero_phase)
        moved = np.linalg.norm(matrices_optical[:, :3, 3] - raw[:, :3, 3], axis=1)
        mode = "zero-phase" if args.smooth_zero_phase else "causal"
        print(f"Smoothed ({mode}, strength={args.smooth}): positions moved "
              f"p50 {np.median(moved) * 100:.2f} cm, max {moved.max() * 100:.2f} cm")
        # Record both: strength is the knob, alpha is what the filter actually used,
        # so a reader does not have to know the convention to reproduce the run.
        manifest["smoothing"] = {"strength": args.smooth, "alpha": 1.0 - args.smooth,
                                 "mode": mode, "applies_to": "trajectory and overlay"}

    matrices = (to_ros_frame(matrices_optical) if args.frame == "ros"
                else matrices_optical)
    npz_path = write_trajectory(out_dir, frames, matrices, intrinsics, manifest,
                                args.frame)
    print(f"\n6-DoF pose  {frames.size} frames ({args.frame} frame) -> {npz_path}")
    print(f"            summary -> {out_dir / 'trajectory.json'}")

    if not args.no_overlay:
        video_path, description = write_overlay_video(
            out_dir, frames, matrices_optical, intrinsics, probe_fps(video)
        )
        print(f"Overlay     {description} -> {video_path}")
        print("            check the mesh tracks the object before trusting the poses.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
