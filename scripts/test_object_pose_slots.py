#!/usr/bin/env python
"""Checks for the GR00T spare-action-dim pose channel. No GPU, no model weights.

Simulates GR00T's action head loss exactly as `groot_n1_7.py` computes it --
`mse(pred, velocity, reduction="none") * action_mask`, reduced by
`sum / mask.sum()` -- and drives the real mixin against it.

    PYTHONPATH=src:<lerobot>/src python scripts/test_object_pose_slots.py
"""
import sys
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch import nn

from h2r_il.object_pose_inject import POSE_KEY, VALID_KEY, PoseStore
from h2r_il.policies.object_pose_slots import ObjectPoseSlotsMixin

STORE = "/mnt/shared_data/h2r_il/outputs/object_pose/kitting/head/store"
ACTION_DIM, MAX_ACTION_DIM, HORIZON, BATCH = 14, 132, 8, 4
POSE_SLICE = slice(ACTION_DIM, ACTION_DIM + 3)
failures = []


def check(label, condition, detail=""):
    print(f"  {'ok  ' if condition else 'FAIL'}  {label}{'  ' + detail if detail else ''}")
    if not condition:
        failures.append(label)


class FakeGroot(ObjectPoseSlotsMixin, nn.Module):
    """The mixin under test, with GR00T's own loss arithmetic."""

    def __init__(self, weight=1.0, store=None):
        nn.Module.__init__(self)
        self.config = SimpleNamespace(
            object_pose_weight=weight, object_pose_target="position",
            object_pose_store=store,
        )
        self._setup_object_pose_slots(ACTION_DIM, MAX_ACTION_DIM)

    def run(self, batch, action, mask, pred, velocity):
        """One training step's worth of GR00T arithmetic."""
        self._pending_batch = batch
        action = self._write_pose_into_actions(action)
        mask = torch.maximum(mask, self._pose_weights(mask))
        # GR00T's action head, verbatim.
        per_element = F.mse_loss(pred, velocity, reduction="none") * mask
        groot_scalar = per_element.sum() / (mask.sum() + 1e-6)
        total, metrics = self._recompose_losses(per_element, mask)
        return action, mask, total, groot_scalar, metrics


def make_batch(valid_steps=HORIZON):
    valid = torch.zeros(BATCH, HORIZON, dtype=torch.bool)
    valid[:, :valid_steps] = True
    return {POSE_KEY: torch.randn(BATCH, HORIZON, 3), VALID_KEY: valid}


torch.manual_seed(0)
action = torch.zeros(BATCH, HORIZON, MAX_ACTION_DIM)
action[:, :, :ACTION_DIM] = torch.randn(BATCH, HORIZON, ACTION_DIM)
base_mask = torch.zeros(BATCH, HORIZON, MAX_ACTION_DIM)
base_mask[:, :, :ACTION_DIM] = 1.0
pred = torch.randn(BATCH, HORIZON, MAX_ACTION_DIM)
velocity = torch.randn(BATCH, HORIZON, MAX_ACTION_DIM)

print("1. no new parameters and no new state-dict entries")
policy = FakeGroot()
check("mixin adds zero parameters", sum(p.numel() for p in policy.parameters()) == 0,
      f"{sum(p.numel() for p in policy.parameters())} params")
check("statistics stay out of the state dict", len(policy.state_dict()) == 0,
      f"keys: {list(policy.state_dict())}")
check("buffers still exist and can move with the module",
      hasattr(policy, "object_pose_mean") and hasattr(policy, "object_pose_std"))

print("\n2. the pose lands in the spare dims and nothing else moves")
batch = make_batch()
written, mask, _, _, _ = policy.run(batch, action, base_mask, pred, velocity)
check("real action dims untouched",
      torch.allclose(written[:, :, :ACTION_DIM], action[:, :, :ACTION_DIM]))
check("pose slot written across the whole horizon",
      (written[:, :, POSE_SLICE] != 0).any(-1).all().item())
check("dims past the pose slot stay zero",
      torch.count_nonzero(written[:, :, ACTION_DIM + 3:]).item() == 0)
check("mask opened exactly the pose slot",
      torch.equal(mask[:, :, POSE_SLICE], torch.ones(BATCH, HORIZON, 3)))
check("mask elsewhere unchanged",
      torch.equal(mask[:, :, ACTION_DIM + 3:], base_mask[:, :, ACTION_DIM + 3:]))

print("\n3. the target is a trajectory, not a repeated constant")
step_spread = written[:, :, POSE_SLICE].std(dim=1).mean()
check("pose varies across chunk steps", float(step_spread) > 0.1,
      f"per-step std {float(step_spread):.3f}")

print("\n4. weight 0 reproduces stock GR00T's loss exactly")
stock = FakeGroot(weight=0.0)
_, mask0, total0, _, _ = stock.run(batch, action, base_mask, pred, velocity)
# what GR00T computes with the pose channel absent entirely
per_stock = F.mse_loss(pred, velocity, reduction="none") * base_mask
stock_scalar = per_stock.sum() / (base_mask.sum() + 1e-6)
check("total == stock action loss", torch.allclose(total0, stock_scalar, atol=1e-6),
      f"{float(total0):.8f} vs {float(stock_scalar):.8f}")

print("\n5. the pose term carries its own weight")
_, _, total1, groot_scalar, m1 = FakeGroot(weight=1.0).run(
    batch, action, base_mask, pred, velocity)
_, _, total2, _, m2 = FakeGroot(weight=2.0).run(batch, action, base_mask, pred, velocity)
check("weight scales only the pose term",
      abs(float(total2 - total1) - m1["object_pose_loss"]) < 1e-5,
      f"delta {float(total2 - total1):.6f}, pose {m1['object_pose_loss']:.6f}")
check("total = action + weight * pose",
      abs(float(total1) - (m1["action_loss"] + m1["object_pose_loss"])) < 1e-5)
check("not merely GR00T's own averaged scalar",
      abs(float(total1) - float(groot_scalar)) > 1e-6,
      "each term normalised over its own elements")

print("\n6. masked steps are not supervised")
half = make_batch(valid_steps=HORIZON // 2)
policy_h = FakeGroot()
written_h, mask_h, _, _, mh = policy_h.run(half, action, base_mask, pred, velocity)
check("mask closed on invalid steps",
      float(mask_h[:, HORIZON // 2:, POSE_SLICE].sum()) == 0.0)
check("supervised elements = B * valid steps * 3",
      mh["object_pose_supervised_elems"] == BATCH * (HORIZON // 2) * 3,
      f"{mh['object_pose_supervised_elems']}")
poisoned = {POSE_KEY: half[POSE_KEY].clone(), VALID_KEY: half[VALID_KEY]}
poisoned[POSE_KEY][:, HORIZON // 2:] = 1e3
_, _, total_p, _, _ = FakeGroot().run(poisoned, action, base_mask, pred, velocity)
_, _, total_c, _, _ = FakeGroot().run(half, action, base_mask, pred, velocity)
check("garbage in masked steps cannot affect the loss",
      torch.allclose(total_p, total_c), f"{float(total_p):.6f} vs {float(total_c):.6f}")

print("\n7. an all-invalid batch is zero, not NaN")
none_valid = {POSE_KEY: torch.randn(BATCH, HORIZON, 3),
              VALID_KEY: torch.zeros(BATCH, HORIZON, dtype=torch.bool)}
p2 = FakeGroot()
pred_g = pred.clone().requires_grad_(True)
_, _, total_n, _, mn = p2.run(none_valid, action, base_mask, pred_g, velocity)
total_n.backward()
check("loss finite", torch.isfinite(total_n).item(), f"{float(total_n):.6f}")
check("pose term exactly zero", mn["object_pose_loss"] == 0.0)
check("gradients finite", torch.isfinite(pred_g.grad).all().item())
check("no gradient on the pose channel",
      float(pred_g.grad[:, :, POSE_SLICE].abs().sum()) == 0.0)

print("\n8. a target shorter than the chunk masks the tail instead of failing")
short = {POSE_KEY: torch.randn(BATCH, 3), VALID_KEY: torch.ones(BATCH, dtype=torch.bool)}
p3 = FakeGroot()
_, mask_s, _, _, ms = p3.run(short, action, base_mask, pred, velocity)
check("per-frame (B, dim) target accepted", ms["object_pose_supervised_elems"] == BATCH * 3)
check("only step 0 supervised", float(mask_s[:, 1:, POSE_SLICE].sum()) == 0.0)

print("\n9. the slot must fit")
try:
    bad = SimpleNamespace(object_pose_weight=1.0, object_pose_target="pose6d",
                          object_pose_store=None)
    obj = nn.Module(); obj.config = bad
    ObjectPoseSlotsMixin._setup_object_pose_slots(obj, 130, 132)
    check("oversized slot rejected", False, "no error raised")
except ValueError as exc:
    check("oversized slot rejected", "no room" in str(exc))

print("\n10. standardisation against the real kitting store")
try:
    real = FakeGroot(store=STORE)
    store = PoseStore(STORE, "position", horizon=HORIZON)
    window = torch.stack([store.attach({"index": torch.tensor(i)})[POSE_KEY]
                          for i in (0, 500, 5000)])
    valid = torch.ones(3, HORIZON, dtype=torch.bool)
    real._pending_batch = {POSE_KEY: window, VALID_KEY: valid}
    out = real._write_pose_into_actions(torch.zeros(3, HORIZON, MAX_ACTION_DIM))
    written_pose = out[:, :, POSE_SLICE]
    check("standardised into roughly unit scale",
          0.1 < float(written_pose.std()) < 5.0, f"std {float(written_pose.std()):.3f}")
    check("raw metres would have been much smaller",
          float(window.std()) < float(written_pose.std()),
          f"raw std {float(window.std()):.3f} -> standardised {float(written_pose.std()):.3f}")
except FileNotFoundError as exc:
    print(f"     skipped (no store): {exc}")

print()
if failures:
    print(f"FAILED: {len(failures)} check(s): {failures}")
    sys.exit(1)
print("all checks passed")
