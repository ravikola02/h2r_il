#!/usr/bin/env python
"""FoundationPose tracking that re-registers at episode boundaries.

v2d's `run_video_to_poses` registers once at a reference frame and then tracks
every remaining frame. That is correct for one continuous take, but a LeRobot
camera video is N independent episodes concatenated end to end: at each cut the
pose prior is the previous take's last frame, which is meaningless. With
recovery disabled a single bad prior propagates to the end of the dataset --
observed on kitting, where tracking collapsed just after the first cut (frame
268) into a pose 12x too close and held it for the remaining 98.7% of frames.

This module is identical to v2d's except that it takes `--register_frames`, a
comma-separated list of frame indices at which to re-register from the observed
mask instead of tracking forward. Pass the episode start frames.

It lives outside the v2d checkout and is bind-mounted into the container as a
single file, so the image's compiled FoundationPose extensions under /workspace
are left untouched.
"""
import argparse
import logging
import os

import cv2
import numpy as np

from v2d.common.datatypes import CameraIntrinsics, DepthImage, Mask, Transform3d
from v2d.common.datatypes import Image as V2dImage
from v2d.foundation_pose.lib.foundation_pose_tracker import FoundationPoseTracker
from v2d.mesh.lib.mesh import Mesh

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _identity_rotation(pose):
    """Same translation, rotation forced to identity.

    FoundationPose always solves full 6-DoF; there is no way to constrain its
    optimiser. So the constraint is applied after the fact and fed back through
    reset_to_pose, which makes the *next* frame track from an identity-rotation
    prior rather than merely relabelling the output.

    Worth knowing before trusting this: the object's body frame comes from SAM3D
    and is arbitrary, so "identity" means aligned with the camera optical axes,
    not with any meaningful axis of the part.
    """
    m = pose.to_matrix()
    m[:3, :3] = np.eye(3)
    return Transform3d.from_matrix(m)


def run(video_path, depth_folder, masks_folder, camera_intrinsics_path, mesh_path,
        poses_dir, weights_dir, reference_frame=0, register_frames="",
        register_iteration=10, track_iteration=5, reregister_iou_thresh=None,
        max_frames=0, register_every=0, fix_rotation=False):
    mesh = Mesh.load(mesh_path)
    tracker = FoundationPoseTracker(mesh, weights_dir)

    intrinsics = CameraIntrinsics.load(camera_intrinsics_path)
    cap = cv2.VideoCapture(video_path)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if max_frames:
        num_frames = min(num_frames, reference_frame + max_frames)

    boundaries = {int(x) for x in register_frames.split(",") if x.strip()}
    if register_every:
        # Periodic registration *in addition to* the episode cuts. Drift measured
        # against MoGe grows with frames-since-registration (0.6 cm at age <30,
        # 8 cm at age 150+), so this bounds it. Anchored to each episode start so
        # the phase does not wander across episodes.
        starts = sorted(boundaries | {reference_frame})
        for i, s in enumerate(starts):
            end = starts[i + 1] if i + 1 < len(starts) else num_frames
            boundaries.update(range(s + register_every, end, register_every))

    logger.info(f"{len(boundaries)} registration frames; {num_frames} frames total")

    def load(idx):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            return None, None, None
        rgb = V2dImage(data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        dp = os.path.join(depth_folder, f"{idx:06d}.png")
        mp = os.path.join(masks_folder, f"{idx:06d}.png")
        depth = DepthImage.load(dp) if os.path.exists(dp) else None
        mask = Mask.load(mp) if os.path.exists(mp) else None
        return rgb, depth, mask

    os.makedirs(poses_dir, exist_ok=True)

    def save(idx, pose):
        pose.save(os.path.join(poses_dir, f"{idx:06d}.json"))

    rgb, depth, mask = load(reference_frame)
    if rgb is None or depth is None or mask is None:
        raise RuntimeError(f"cannot load reference frame {reference_frame}")
    pose = tracker.register(rgb, depth, mask, intrinsics, iteration=register_iteration)
    if fix_rotation:
        pose = _identity_rotation(pose)
    save(reference_frame, pose)
    tracker.reset_to_pose(pose)
    if fix_rotation:
        logger.info("fix_rotation: rotation forced to identity every frame")

    registered = 0
    recovered = 0
    for idx in range(reference_frame + 1, num_frames):
        rgb, depth, mask = load(idx)
        if rgb is None or depth is None:
            break

        if idx in boundaries and mask is not None:
            # Episode cut: the previous frame is a different take, so the tracked
            # prior carries no information. Register from this frame's mask.
            pose = tracker.register(rgb, depth, mask, intrinsics,
                                    iteration=register_iteration)
            tracker.reset_to_pose(pose)
            registered += 1
            logger.info(f"Forward frame {idx}/{num_frames} — re-registered (episode start)")
        elif reregister_iou_thresh is not None and mask is not None:
            pose, did = tracker.track_one_with_recovery(
                rgb, depth, mask, intrinsics, iteration=track_iteration,
                iou_thresh=reregister_iou_thresh, recovery_iteration=register_iteration)
            recovered += did
            if idx % 500 == 0:
                logger.info(f"Forward frame {idx}/{num_frames}")
        else:
            pose = tracker.track_one(rgb, depth, intrinsics, iteration=track_iteration)
            if idx % 500 == 0:
                logger.info(f"Forward frame {idx}/{num_frames}")

        if fix_rotation:
            # Feed the constrained pose back, so the next frame's prior carries
            # identity rotation too rather than the tracker drifting freely
            # underneath and only the saved copy being flattened.
            pose = _identity_rotation(pose)
            tracker.reset_to_pose(pose)
        save(idx, pose)

    cap.release()
    logger.info(f"Completed {num_frames} frames; {registered} boundary registrations, "
                f"{recovered} IoU recoveries")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--video_path", required=True)
    p.add_argument("--depth_folder", required=True)
    p.add_argument("--masks_folder", required=True)
    p.add_argument("--camera_intrinsics_path", required=True)
    p.add_argument("--mesh_path", required=True)
    p.add_argument("--poses_dir", required=True)
    p.add_argument("--weights_dir", required=True)
    p.add_argument("--reference_frame", type=int, default=0)
    p.add_argument("--register_frames", default="")
    p.add_argument("--register_iteration", type=int, default=10)
    p.add_argument("--track_iteration", type=int, default=5)
    p.add_argument("--reregister_iou_thresh", type=float, default=None)
    p.add_argument("--max_frames", type=int, default=0)
    p.add_argument("--register_every", type=int, default=0)
    p.add_argument("--fix_rotation", action="store_true",
                   help="Force rotation to identity every frame, tracking "
                        "translation only.")
    a = p.parse_args()
    run(a.video_path, a.depth_folder, a.masks_folder, a.camera_intrinsics_path,
        a.mesh_path, a.poses_dir, a.weights_dir, a.reference_frame,
        a.register_frames, a.register_iteration, a.track_iteration,
        a.reregister_iou_thresh, a.max_frames, a.register_every, a.fix_rotation)
