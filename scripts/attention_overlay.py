#!/usr/bin/env python
"""Render *where the GR00T policy looked* over the recorded camera frames.

Same open-loop replay as ``scripts/eval_openloop.py`` — recorded observations in,
predicted action chunks out, nothing executed — with the DiT's action->image
cross-attention painted over each camera view as a video.

What the heatmap actually is
----------------------------
``AlternateVLDiT`` (lerobot/policies/groot/action_head/cross_attention_dit.py)
interleaves its 32 blocks::

    odd idx                        -> self-attention
    even idx, idx % 4 == 0         -> cross-attend to TEXT tokens
    even idx, otherwise            -> cross-attend to IMAGE tokens

so image cross-attention lives at blocks [2, 6, 10, 14, 18, 22, 26, 30]. On each of
those we recompute ``softmax(Q K^T / sqrt(d))`` from the block's own ``to_q``/``to_k``
(the model runs SDPA, which never materialises the probabilities), average over heads
and over the action-chunk queries, and keep the column per image token. Scores are
accumulated over every denoising step of every inference call, then averaged.

Tokens -> pixels: Qwen3VL merges ``spatial_merge_size**2`` patches into one token, so
each 256x256 model input becomes an 8x8 token grid. The heatmap is genuinely 8x8,
bilinearly upscaled to the frame — coarse regions, not fine edges. Do not read
pixel-level precision into it.

The N cameras appear as N contiguous runs of image tokens, in the order the packing
processor fed them (``video_modality_keys`` on the checkpoint's preprocessor — here
head, wrist_left, wrist_right). Runs are located from ``image_mask`` and split by the
per-image token counts derived from ``image_grid_thw``.

Caveats
-------
- Attention is allocation, not causation. A bright patch is where the model attended,
  not proof it used that information, or used it correctly.
- Averaging over 8 layers and all heads hides disagreement between them; ``--layers``
  isolates one.
- Text and state contributions are invisible here — only image-attending blocks are
  captured.
- The flow-matching loop starts from ``torch.randn``, so predictions vary run to run
  unless seeded. ``--seed`` (default 0) is applied before every inference call.

Usage
-----
    uv run python scripts/attention_overlay.py \
        --checkpoint outputs/<run>/checkpoints/020000/pretrained_model \
        --dataset <local dataset dir> \
        --episodes 0 1 2 \
        --out outputs/validation/<run>__attn
"""

import argparse
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch

from lerobot.datasets.video_utils import decode_video_frames
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.groot.configuration_groot import GrootConfig
from lerobot.policies.groot.modeling_groot import GrootPolicy

DIM_NAMES = [
    "left.wrist.x", "left.wrist.y", "left.wrist.z",
    "left.wrist.roll", "left.wrist.pitch", "left.wrist.yaw",
    "right.wrist.x", "right.wrist.y", "right.wrist.z",
    "right.wrist.roll", "right.wrist.pitch", "right.wrist.yaw",
    "left.gripper", "right.gripper",
]
ANGLE_DIMS = [3, 4, 5, 9, 10, 11]
POS_DIMS = [0, 1, 2, 6, 7, 8]


# ---------------------------------------------------------------------------------
# Attention capture
# ---------------------------------------------------------------------------------


class ImageAttentionCapture:
    """Hooks the DiT's image cross-attention blocks, accumulating per-image-token scores.

        cap = ImageAttentionCapture(policy)
        cap.attach(); cap.reset()
        policy.predict_action_chunk(batch)     # all denoising steps accumulate
        maps = cap.per_camera_maps(cap.mean_scores(), n_cameras=3)
        cap.detach()
    """

    def __init__(self, policy: GrootPolicy, layers: list[int] | None = None):
        self.policy = policy
        self.model = policy._groot_model
        self.dit = self._find_dit(self.model)
        self.image_layers = self._image_layer_indices(self.dit)
        if layers:
            unknown = sorted(set(layers) - set(self.image_layers))
            if unknown:
                raise ValueError(
                    f"--layers {unknown} are not image cross-attention layers. "
                    f"Available: {self.image_layers}"
                )
            self.layers = sorted(layers)
        else:
            self.layers = list(self.image_layers)

        self._handles: list[Any] = []
        self._sum: torch.Tensor | None = None
        self._n = 0
        # Captured from the backbone so tokens can be mapped back to cameras.
        self.image_mask: torch.Tensor | None = None
        self.image_grid_thw: torch.Tensor | None = None
        self._backbone_forward = None

    @staticmethod
    def _find_dit(model: torch.nn.Module) -> torch.nn.Module:
        for _, mod in model.named_modules():
            if type(mod).__name__ == "AlternateVLDiT":
                return mod
        raise RuntimeError("AlternateVLDiT not found; this script targets the GR00T N1.7 DiT.")

    @staticmethod
    def _image_layer_indices(dit: torch.nn.Module) -> list[int]:
        """Mirrors AlternateVLDiT.forward's block routing."""
        n = int(getattr(dit, "attend_text_every_n_blocks", 2))
        total = len(dit.transformer_blocks)
        return [i for i in range(total) if i % 2 == 0 and i % (2 * n) != 0]

    def attach(self) -> None:
        for idx in self.layers:
            attn = self.dit.transformer_blocks[idx].attn1
            self._handles.append(
                attn.register_forward_pre_hook(self._make_hook(), with_kwargs=True)
            )
        # Spy on the backbone to grab image_mask / image_grid_thw for this observation.
        bb = self.model.backbone
        self._backbone_forward = bb.forward

        def spy(vl_input):
            grid = vl_input.get("image_grid_thw") if hasattr(vl_input, "get") else None
            out = self._backbone_forward(vl_input)
            self.image_mask = out["image_mask"].detach()
            if grid is not None:
                self.image_grid_thw = grid.detach()
            return out

        bb.forward = spy

    def detach(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()
        if self._backbone_forward is not None:
            self.model.backbone.forward = self._backbone_forward
            self._backbone_forward = None

    def reset(self) -> None:
        self._sum = None
        self._n = 0

    def _make_hook(self):
        def hook(module, args, kwargs):
            hidden = kwargs.get("encoder_hidden_states")
            if hidden is None:
                return  # self-attention call; nothing to do
            query_in = args[0] if args else kwargs.get("hidden_states")
            if query_in is None:
                return
            with torch.no_grad():
                self._accumulate(module, query_in, hidden, kwargs.get("attention_mask"))

        return hook

    def _accumulate(
        self,
        attn: torch.nn.Module,
        query_in: torch.Tensor,
        encoder_hidden: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> None:
        heads = attn.heads
        q = attn.to_q(query_in)  # (B, T, inner)
        k = attn.to_k(encoder_hidden)  # (B, S, inner)
        b, t, inner = q.shape
        s = k.shape[1]
        head_dim = inner // heads
        q = q.view(b, t, heads, head_dim).transpose(1, 2).float()  # (B, H, T, d)
        k = k.view(b, s, heads, head_dim).transpose(1, 2).float()  # (B, H, S, d)

        logits = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(head_dim)  # (B,H,T,S)
        if mask is not None:
            m = mask
            while m.dim() < 4:
                m = m.unsqueeze(1)
            if m.dtype == torch.bool:
                logits = logits.masked_fill(~m, torch.finfo(logits.dtype).min)
            else:
                logits = logits + m.to(logits.dtype)

        probs = torch.softmax(logits, dim=-1)
        # Average over heads and over the action-chunk queries -> one score per token.
        scores = torch.nan_to_num(probs.mean(dim=1).mean(dim=1)[0], nan=0.0)  # (S,)

        if self._sum is None:
            self._sum = scores.detach().clone()
        else:
            self._sum += scores.detach()
        self._n += 1

    def mean_scores(self) -> np.ndarray | None:
        if self._sum is None or self._n == 0:
            return None
        return (self._sum / self._n).cpu().numpy().astype(np.float32)

    def per_camera_maps(self, scores: np.ndarray | None, n_cameras: int) -> list[np.ndarray] | None:
        """Split token scores into one (h, w) grid per camera, in packing order."""
        if scores is None or self.image_mask is None:
            return None
        idx = torch.where(self.image_mask[0])[0].cpu().numpy()
        if idx.size == 0:
            return None

        if self.image_grid_thw is not None:
            merge = self._spatial_merge_size()
            grid = self.image_grid_thw.cpu().numpy()
            shapes, counts = [], []
            for row in grid[:n_cameras]:
                _t, h, w = int(row[0]), int(row[1]), int(row[2])
                shapes.append((h // merge, w // merge))
                counts.append((h // merge) * (w // merge))
        else:  # fall back to an equal split into squares
            per = idx.size // n_cameras
            side = int(round(math.sqrt(per)))
            shapes = [(side, side)] * n_cameras
            counts = [per] * n_cameras

        maps, cursor = [], 0
        for (h, w), c in zip(shapes, counts):
            take = idx[cursor : cursor + c]
            cursor += c
            if take.size != c:
                return None
            maps.append(scores[take].reshape(h, w))
        return maps

    def _spatial_merge_size(self) -> int:
        cfg = getattr(self.model.backbone, "model", None)
        cfg = getattr(cfg, "config", None)
        vcfg = getattr(cfg, "vision_config", None)
        return int(getattr(vcfg, "spatial_merge_size", 2) or 2)


# ---------------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------------


def to_bgr(frame: torch.Tensor, width: int) -> np.ndarray:
    """(C,H,W) float [0,1] -> (h,w,3) uint8 BGR, scaled to `width`."""
    img = (frame.permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    h, w = img.shape[:2]
    if w != width:
        img = cv2.resize(img, (width, int(round(h * width / w))), interpolation=cv2.INTER_AREA)
    return img


def overlay(img: np.ndarray, amap: np.ndarray, vmin: float, vmax: float,
            alpha: float, colormap: int) -> np.ndarray:
    h, w = img.shape[:2]
    norm = (amap - vmin) / max(vmax - vmin, 1e-12)
    norm = np.clip(norm, 0.0, 1.0)
    big = cv2.resize((norm * 255).astype(np.uint8), (w, h), interpolation=cv2.INTER_LINEAR)
    heat = cv2.applyColorMap(big, colormap)
    return cv2.addWeighted(heat, alpha, img, 1 - alpha, 0)


def label(img: np.ndarray, text: str, corner: str = "tl") -> np.ndarray:
    out = img.copy()
    org = (8, 24) if corner == "tl" else (8, out.shape[0] - 10)
    cv2.putText(out, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(out, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
    return out


# ---------------------------------------------------------------------------------


def angdiff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d = a - b
    return np.arctan2(np.sin(d), np.cos(d))


def signed_err(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    err = pred - gt
    err[..., ANGLE_DIMS] = angdiff(pred[..., ANGLE_DIMS], gt[..., ANGLE_DIMS])
    return err


def gt_chunk(action: np.ndarray, s: int, chunk: int) -> np.ndarray:
    return action[np.clip(s + np.arange(chunk), 0, len(action) - 1)]


def video_path(root: Path, cam: str, ep: int) -> Path:
    return root / "videos" / "chunk-000" / f"observation.images.{cam}" / f"episode_{ep:06d}.mp4"


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--dataset", required=True, help="LeRobot v2.1 dataset root")
    p.add_argument("--episodes", type=int, nargs="+", default=[0])
    p.add_argument("--out", required=True)
    p.add_argument("--stride", type=int, default=20,
                   help="frames between rendered inferences")
    p.add_argument("--max-points", type=int, default=None,
                   help="cap on rendered inferences per episode")
    p.add_argument("--n-action-steps", type=int, default=None,
                   help="score only the first N steps of each chunk (default: full chunk)")
    p.add_argument("--task", default=None, help="override language instruction")
    p.add_argument("--layers", type=int, nargs="+", default=None,
                   help="image cross-attention blocks to average (default: all 8)")
    p.add_argument("--alpha", type=float, default=0.5, help="heatmap opacity 0-1")
    p.add_argument("--colormap", default="jet", help="jet | turbo | inferno | viridis")
    p.add_argument("--normalize", choices=["global", "frame", "camera"], default="global",
                   help="global: one colour scale per episode across all cameras (camera "
                        "brightness comparable). camera: one scale per camera (structure "
                        "within a view clearer). frame: rescale every frame independently.")
    p.add_argument("--vmax-percentile", type=float, default=99.0)
    p.add_argument("--tile-width", type=int, default=480, help="per-camera width in the video")
    p.add_argument("--fps", type=int, default=4,
                   help="output video fps (one frame per inference, not per control step)")
    p.add_argument("--no-show-raw", action="store_true", help="drop the raw-frame row")
    p.add_argument("--save-frames", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    root = Path(args.dataset)
    out = Path(args.out)
    (out / "videos").mkdir(parents=True, exist_ok=True)

    info = json.loads((root / "meta" / "info.json").read_text())
    fps = float(info["fps"])
    tasks = {
        json.loads(line)["task_index"]: json.loads(line)["task"]
        for line in (root / "meta" / "tasks.jsonl").read_text().splitlines()
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

    # Camera order must match what the packing processor feeds the backbone, since that
    # is the order the image-token runs appear in.
    cameras = None
    for step in getattr(preprocessor, "steps", []):
        keys = getattr(step, "video_modality_keys", None)
        if keys:
            cameras = list(keys)
            break
    if cameras is None:
        cameras = ["head", "wrist_left", "wrist_right"]
        print(f"warning: could not read video_modality_keys; assuming {cameras}")
    print(f"cameras (packing order): {cameras}")

    cap = ImageAttentionCapture(policy, layers=args.layers)
    print(f"image cross-attention blocks: {cap.image_layers} | capturing {cap.layers}")
    cap.attach()

    colormap = getattr(cv2, f"COLORMAP_{args.colormap.upper()}")
    summary = {}

    try:
        for ep in args.episodes:
            df = pd.read_parquet(root / "data" / "chunk-000" / f"episode_{ep:06d}.parquet")
            state = np.stack(df["observation.state"].values).astype(np.float32)
            action = np.stack(df["action"].values).astype(np.float32)
            T = len(df)
            task = args.task if args.task is not None else tasks[int(df["task_index"].iloc[0])]

            starts = list(range(0, max(T - (n_steps - 1), 1), args.stride))
            if args.max_points:
                starts = starts[: args.max_points]
            print(f"\nepisode {ep}: {T} frames, task={task!r}, {len(starts)} inferences")

            ts = [s / fps for s in starts]
            frames = {
                cam: decode_video_frames(video_path(root, cam, ep), ts,
                                         tolerance_s=1.0 / fps + 1e-4)
                for cam in cameras
            }

            tiles, maps_per_point, preds, gts = [], [], [], []
            cam_mass = np.zeros(len(cameras), dtype=np.float64)

            for i, s in enumerate(starts):
                batch = {
                    "observation.state": torch.from_numpy(state[s : s + 1]),
                    "task": [task],
                }
                for cam in cameras:
                    batch[f"observation.images.{cam}"] = frames[cam][i : i + 1]

                cap.reset()
                torch.manual_seed(args.seed)
                with torch.no_grad():
                    pb = preprocessor(batch)
                    a = policy.predict_action_chunk(pb)
                    a = postprocessor(a)
                a = a.float().cpu().numpy()[0][:n_steps]

                amaps = cap.per_camera_maps(cap.mean_scores(), len(cameras))
                if amaps is None:
                    raise RuntimeError("no attention captured — the DiT hooks did not fire")
                maps_per_point.append(amaps)
                cam_mass += [float(m.sum()) for m in amaps]

                tiles.append([to_bgr(frames[cam][i], args.tile_width) for cam in cameras])

                gt = gt_chunk(action, s, n_steps)
                n = min(len(gt), len(a))
                preds.append(a[:n])
                gts.append(gt[:n])
                print(f"  {i + 1}/{len(starts)}", end="\r", flush=True)

            # colour scale
            allv = np.concatenate([m.ravel() for pt in maps_per_point for m in pt])
            g_lo, g_hi = float(allv.min()), float(np.percentile(allv, args.vmax_percentile))
            per_cam = [
                np.concatenate([pt[c].ravel() for pt in maps_per_point])
                for c in range(len(cameras))
            ]
            c_lo = [float(v.min()) for v in per_cam]
            c_hi = [float(np.percentile(v, args.vmax_percentile)) for v in per_cam]

            writer, frame_dir = None, out / "frames" / f"episode_{ep:02d}"
            if args.save_frames:
                frame_dir.mkdir(parents=True, exist_ok=True)

            for i, (row, amaps) in enumerate(zip(tiles, maps_per_point)):
                over = []
                for c, (img, amap) in enumerate(zip(row, amaps)):
                    if args.normalize == "frame":
                        lo, hi = float(amap.min()), float(np.percentile(amap, args.vmax_percentile))
                    elif args.normalize == "camera":
                        lo, hi = c_lo[c], c_hi[c]
                    else:
                        lo, hi = g_lo, g_hi
                    share = amaps[c].sum() / max(sum(m.sum() for m in amaps), 1e-12)
                    over.append(label(overlay(img, amap, lo, hi, args.alpha, colormap),
                                      f"{cameras[c]}  {share * 100:.0f}%"))
                canvas = np.hstack(over)
                if not args.no_show_raw:
                    canvas = np.vstack([np.hstack([label(x, cameras[c])
                                                   for c, x in enumerate(row)]), canvas])
                canvas = label(canvas, f"ep {ep}  frame {starts[i]}", corner="bl")

                if writer is None:
                    h, w = canvas.shape[:2]
                    writer = cv2.VideoWriter(
                        str(out / "videos" / f"episode_{ep:02d}_attention.mp4"),
                        cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h))
                writer.write(canvas)
                if args.save_frames:
                    cv2.imwrite(str(frame_dir / f"{starts[i]:06d}.png"), canvas)
            if writer is not None:
                writer.release()

            P, G = np.concatenate(preds), np.concatenate(gts)
            E = signed_err(P, G)
            share = (cam_mass / max(cam_mass.sum(), 1e-12)).tolist()
            summary[ep] = {
                "frames": T,
                "task": task,
                "inferences": len(starts),
                "chunk": chunk,
                "n_action_steps": n_steps,
                "layers": cap.layers,
                "attention_share_by_camera": dict(zip(cameras, [round(x, 4) for x in share])),
                "attention_grid": [list(m.shape) for m in maps_per_point[0]],
                "pos_mae_m": float(np.abs(E[:, POS_DIMS]).mean()),
                "rot_mae_rad": float(np.abs(E[:, ANGLE_DIMS]).mean()),
                "per_dim_mae": dict(zip(DIM_NAMES, np.abs(E).mean(0).round(4).tolist())),
                "video": str(out / "videos" / f"episode_{ep:02d}_attention.mp4"),
            }
            print(f"\n  attention share: " +
                  "  ".join(f"{c} {s * 100:.1f}%" for c, s in zip(cameras, share)))
            print(f"  pos MAE {summary[ep]['pos_mae_m']:.4f} m  "
                  f"rot MAE {summary[ep]['rot_mae_rad']:.4f} rad")
    finally:
        cap.detach()

    (out / "attention_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {out / 'attention_summary.json'} and {out / 'videos'}")


if __name__ == "__main__":
    main()
