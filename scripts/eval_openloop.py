#!/usr/bin/env python
"""Open-loop validation of a trained GR00T policy against recorded episodes.

At each sampled start frame t the policy sees the recorded observation and
predicts a chunk of `chunk_size` actions, which is compared against the
recorded actions[t : t+chunk_size]. Nothing is executed: the observations
always come from the recording, so errors do not compound the way they would
in a rollout. This measures action prediction, not task success.

The val export (pick_cube_eef_optical) stores state/action as 20-d
[L pos 3, L rot6d 6, R pos 3, R rot6d 6, L grip, R grip], while the policy was
trained on the 14-d [L pos 3, L rpy 3, R pos 3, R rpy 3, L grip, R grip]
layout. The rot6d -> rpy mapping used here (rows convention + intrinsic XYZ
Euler) was verified exactly against gear_left, which ships both layouts for the
same frames: it round-trips to ~1e-7 rad.

Usage:
  uv run python scripts/eval_openloop.py \
      --checkpoint outputs/<job>/checkpoints/020000/pretrained_model \
      --dataset <path to the validation export> \
      --episodes 14 15 19 --out outputs/validation/<job>__pick_cube
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.spatial.transform import Rotation

from lerobot.datasets.video_utils import decode_video_frames
from lerobot.policies.groot.modeling_groot import GrootPolicy
from lerobot.policies.groot.configuration_groot import GrootConfig
from lerobot.processor import make_default_processors
from lerobot.policies.factory import make_pre_post_processors

DIM_NAMES = [
    "left.wrist.x", "left.wrist.y", "left.wrist.z",
    "left.wrist.roll", "left.wrist.pitch", "left.wrist.yaw",
    "right.wrist.x", "right.wrist.y", "right.wrist.z",
    "right.wrist.roll", "right.wrist.pitch", "right.wrist.yaw",
    "left.gripper", "right.gripper",
]
CAMERAS = ["head", "wrist_left", "wrist_right"]


def rot6d_to_rpy(d6: np.ndarray) -> np.ndarray:
    """(N,6) rot6d -> (N,3) intrinsic-XYZ Euler. Rows convention, Gram-Schmidt."""
    a1, a2 = d6[:, :3], d6[:, 3:]
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
    b2 = a2 - (b1 * a2).sum(-1, keepdims=True) * b1
    b2 = b2 / np.linalg.norm(b2, axis=-1, keepdims=True)
    b3 = np.cross(b1, b2)
    mat = np.stack([b1, b2, b3], axis=-2)  # rows
    return Rotation.from_matrix(mat).as_euler("XYZ")


def to14(v20: np.ndarray) -> np.ndarray:
    """(N,20) val layout -> (N,14) trained layout."""
    return np.concatenate(
        [
            v20[:, 0:3],                  # L pos
            rot6d_to_rpy(v20[:, 3:9]),    # L rpy
            v20[:, 9:12],                 # R pos
            rot6d_to_rpy(v20[:, 12:18]),  # R rpy
            v20[:, 18:20],                # grippers
        ],
        axis=1,
    ).astype(np.float32)


def angdiff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Wrapped a-b for angle dims, plain difference elsewhere is handled by caller."""
    d = a - b
    return np.arctan2(np.sin(d), np.cos(d))


ANGLE_DIMS = [3, 4, 5, 9, 10, 11]


def signed_err(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Per-dim signed error, wrapping the Euler dims so a +-pi flip is not a huge error."""
    err = pred - gt
    err[..., ANGLE_DIMS] = angdiff(pred[..., ANGLE_DIMS], gt[..., ANGLE_DIMS])
    return err


def gt_chunk(action: np.ndarray, s: int, chunk: int) -> np.ndarray:
    """The recorded action chunk at start frame s, as-is.

    Past the end of the episode the last recorded action is repeated, so a chunk
    that runs off the end still scores against something meaningful.
    """
    return action[np.clip(s + np.arange(chunk), 0, len(action) - 1)]


def load_episode(root: Path, ep: int, layout: str):
    pq = root / "data" / "chunk-000" / f"episode_{ep:06d}.parquet"
    df = pd.read_parquet(pq)
    conv = to14 if layout == "rot6d20" else (lambda v: v.astype(np.float32))
    state = conv(np.stack(df["observation.state"].values))
    action = conv(np.stack(df["action"].values))
    return df, state, action


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--episodes", type=int, nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--stride", type=int, default=20, help="frames between evaluated start points")
    p.add_argument(
        "--n-action-steps",
        type=int,
        default=None,
        help="Keep only the first N steps of each predicted chunk (receding horizon), the way a "
        "deployment would re-plan rather than run all chunk_size steps. Defaults to the full "
        "chunk_size. The policy is unchanged: it still predicts the whole chunk, the tail is "
        "just dropped before scoring.",
    )
    p.add_argument("--task", default=None, help="override language instruction")
    p.add_argument(
        "--state-layout",
        choices=["rot6d20", "rpy14"],
        default="rot6d20",
        help="rot6d20: the 20-d val export, converted to the trained 14-d layout. "
        "rpy14: already in the trained layout (e.g. gear_left lerobot_v2).",
    )
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    root = Path(args.dataset)
    out = Path(args.out)
    (out / "plots").mkdir(parents=True, exist_ok=True)

    info = json.loads((root / "meta" / "info.json").read_text())
    fps = float(info["fps"])
    tasks = {
        json.loads(l)["task_index"]: json.loads(l)["task"]
        for l in (root / "meta" / "tasks.jsonl").read_text().splitlines()
    }

    print(f"loading policy from {args.checkpoint}")
    policy = GrootPolicy.from_pretrained(args.checkpoint)
    policy.to(args.device)
    policy.eval()
    cfg: GrootConfig = policy.config
    chunk = cfg.chunk_size
    n_steps = min(args.n_action_steps or chunk, chunk)

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg,
        pretrained_path=args.checkpoint,
        preprocessor_overrides={"device_processor": {"device": args.device}},
    )

    summary = {}
    for ep in args.episodes:
        df, state, action = load_episode(root, ep, args.state_layout)
        T = len(df)
        task = args.task if args.task is not None else tasks[int(df["task_index"].iloc[0])]
        # the kept prefix spans n_steps-1 frames of the demo, so start points
        # only need to stop that far from the end.
        span = n_steps - 1
        starts = list(range(0, max(T - span, 1), args.stride))
        print(f"\nepisode {ep}: {T} frames, task={task!r}, {len(starts)} eval points")

        # decode only the frames we actually need, per camera
        ts = [s / fps for s in starts]
        frames = {}
        for cam in CAMERAS:
            vp = root / "videos" / "chunk-000" / f"observation.images.{cam}" / f"episode_{ep:06d}.mp4"
            frames[cam] = decode_video_frames(vp, ts, tolerance_s=1.0 / fps + 1e-4)

        preds, gts = [], []
        for i, s in enumerate(starts):
            batch = {
                "observation.state": torch.from_numpy(state[s : s + 1]),
                "task": [task],
            }
            for cam in CAMERAS:
                batch[f"observation.images.{cam}"] = frames[cam][i : i + 1]
            pb = preprocessor(batch)
            a = policy.predict_action_chunk(pb)
            a = postprocessor(a)
            # policy still predicts the full chunk; keep only the prefix we score.
            a = a.float().cpu().numpy()[0][:n_steps]  # (n_steps, 14)

            gt = gt_chunk(action, s, n_steps)
            n = min(len(gt), len(a))
            preds.append(a[:n])
            gts.append(gt[:n])
            if i % 5 == 0:
                print(f"  {i+1}/{len(starts)}", end="\r", flush=True)

        P = np.concatenate(preds)
        G = np.concatenate(gts)
        E = signed_err(P, G)

        per_dim_mae = np.abs(E).mean(0)
        per_dim_rmse = np.sqrt((E**2).mean(0))
        # A constant per-dim bias means the policy has the trajectory shape but sits
        # in a shifted frame; debiased MAE is what is left once that shift is removed.
        # If debiased MAE ~ MAE, the shift is not the problem and there is no signal.
        per_dim_bias = E.mean(0)
        per_dim_mae_debiased = np.abs(E - per_dim_bias).mean(0)
        # Same debiasing for the frozen-state baseline, so the two stay comparable.
        base_debiased = None
        # a baseline any useful policy must beat: predict the current state, frozen
        base = np.concatenate(
            [np.repeat(state[s : s + 1], len(g), axis=0) for s, g in zip(starts, gts)]
        )
        BE = signed_err(base, G)
        base_mae = np.abs(BE).mean(0)
        base_debiased = np.abs(BE - BE.mean(0)).mean(0)

        summary[ep] = {
            "frames": T,
            "task": task,
            "eval_points": len(starts),
            "chunk": chunk,
            "n_action_steps": n_steps,
            "per_dim_mae": dict(zip(DIM_NAMES, per_dim_mae.round(4).tolist())),
            "per_dim_rmse": dict(zip(DIM_NAMES, per_dim_rmse.round(4).tolist())),
            "per_dim_bias": dict(zip(DIM_NAMES, per_dim_bias.round(4).tolist())),
            "per_dim_mae_debiased": dict(zip(DIM_NAMES, per_dim_mae_debiased.round(4).tolist())),
            "static_state_baseline_mae": dict(zip(DIM_NAMES, base_mae.round(4).tolist())),
            "static_state_baseline_mae_debiased": dict(zip(DIM_NAMES, base_debiased.round(4).tolist())),
            "pos_mae_m": float(np.abs(E[:, [0, 1, 2, 6, 7, 8]]).mean()),
            "rot_mae_rad": float(np.abs(E[:, ANGLE_DIMS]).mean()),
            "baseline_pos_mae_m": float(np.abs(BE[:, [0, 1, 2, 6, 7, 8]]).mean()),
            "baseline_rot_mae_rad": float(np.abs(BE[:, ANGLE_DIMS]).mean()),
        }
        print(f"\n  pos MAE {summary[ep]['pos_mae_m']:.4f} m "
              f"(static-state baseline {summary[ep]['baseline_pos_mae_m']:.4f} m)")
        print(f"  rot MAE {summary[ep]['rot_mae_rad']:.4f} rad "
              f"(baseline {summary[ep]['baseline_rot_mae_rad']:.4f} rad)")

        # per-dim trajectory: GT vs the chunk predicted from each start point
        fig, axes = plt.subplots(7, 2, figsize=(16, 20), sharex=True)
        for d, ax in enumerate(axes.T.flatten()):
            ax.plot(action[:, d], "k-", lw=1.5, label="ground truth", zorder=3)
            for j, s in enumerate(starts):
                seg = preds[j][:, d]
                # place the chunk on the demo timeline: step j lands at frame s + j.
                xs = s + np.arange(len(seg))
                ax.plot(xs, seg, "-", lw=0.8, alpha=0.6,
                        color="tab:red", label="predicted chunk" if j == 0 else None)
            ax.set_title(f"{DIM_NAMES[d]}  (MAE {per_dim_mae[d]:.4f})", fontsize=9)
            ax.grid(alpha=0.3)
            if d == 0:
                ax.legend(fontsize=8)
        fig.suptitle(
            f"Open-loop action prediction — episode {ep} (raw episode_{ep+1:04d})\n"
            f"{Path(args.checkpoint).parts[-4]} | task={task!r} "
            f"| first {n_steps}/{chunk} steps",
            fontsize=12,
        )
        fig.supxlabel("frame index")
        fig.tight_layout()
        fig.savefig(out / "plots" / f"episode_{ep:02d}_traj.png", dpi=110)
        plt.close(fig)

        # error growth across the chunk horizon
        H = np.stack([signed_err(pr[:n_steps], g[:n_steps])
                      for pr, g in zip(preds, gts) if len(pr) == n_steps])
        if len(H):
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(np.abs(H[:, :, [0, 1, 2, 6, 7, 8]]).mean((0, 2)), label="position (m)")
            ax.plot(np.abs(H[:, :, ANGLE_DIMS]).mean((0, 2)), label="rotation (rad)")
            ax.set_xlabel("step within predicted chunk")
            ax.set_ylabel("mean absolute error")
            ax.set_title(f"Error vs chunk horizon — episode {ep}")
            ax.grid(alpha=0.3)
            ax.legend()
            fig.tight_layout()
            fig.savefig(out / "plots" / f"episode_{ep:02d}_horizon.png", dpi=110)
            plt.close(fig)

    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {out/'summary.json'} and {out/'plots'}")


if __name__ == "__main__":
    main()
