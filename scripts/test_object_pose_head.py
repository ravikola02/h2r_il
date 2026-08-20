#!/usr/bin/env python
"""Checks for the object-pose aux head that do not need a GPU or model weights.

Covers the parts that are easy to get silently wrong: whether the target survives
LeRobot's processor pipeline at all, whether the loss actually masks, and whether
the gradient reaches the trunk.

    PYTHONPATH=src:<lerobot>/src python scripts/test_object_pose_head.py
"""
import sys
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from h2r_il.losses.object_pose import masked_mean_pool, object_pose_loss, target_statistics
from h2r_il.object_pose_inject import POSE_KEY, VALID_KEY
from h2r_il.policies.object_pose_head import ObjectPoseAuxMixin, ObjectPoseHead

STORE = "/mnt/shared_data/h2r_il/outputs/object_pose/kitting/head/store"
failures = []


def check(label, condition, detail=""):
    print(f"  {'ok  ' if condition else 'FAIL'}  {label}{'  ' + detail if detail else ''}")
    if not condition:
        failures.append(label)


print("1. the target survives LeRobot's processor pipeline")
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.processor.converters import batch_to_transition, transition_to_batch
from lerobot.processor.normalize_processor import NormalizerProcessorStep

raw = {
    "observation.state": torch.ones(2, 14),
    "action": torch.ones(2, 14),
    POSE_KEY: torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]),
    VALID_KEY: torch.tensor([True, False]),
}
step = NormalizerProcessorStep(
    features={
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(14,)),
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(14,)),
    },
    norm_map={FeatureType.STATE: NormalizationMode.MEAN_STD,
              FeatureType.ACTION: NormalizationMode.MEAN_STD},
    stats={"observation.state": {"mean": torch.zeros(14), "std": torch.ones(14)},
           "action": {"mean": torch.zeros(14), "std": torch.ones(14)}},
)
out = transition_to_batch(step(batch_to_transition(raw)))
check("pose key reaches the policy", POSE_KEY in out)
check("valid flag reaches the policy", VALID_KEY in out)
check("pose value untouched by the normalizer",
      torch.allclose(out[POSE_KEY], raw[POSE_KEY]))
# The bug this replaced: an unprefixed key is silently dropped here.
unprefixed = transition_to_batch(batch_to_transition({"object_pose": torch.zeros(2, 3)}))
check("an unprefixed key would have been dropped", "object_pose" not in unprefixed)

print("\n2. pooling ignores padding")
features = torch.stack([
    torch.stack([torch.full((4,), 1.0), torch.full((4,), 3.0), torch.full((4,), 99.0)]),
    torch.stack([torch.full((4,), 2.0), torch.full((4,), 99.0), torch.full((4,), 99.0)]),
])
mask = torch.tensor([[True, True, False], [True, False, False]])
pooled = masked_mean_pool(features, mask)
check("padded tokens excluded",
      torch.allclose(pooled, torch.tensor([[2.0] * 4, [2.0] * 4])), f"got {pooled[:, 0].tolist()}")
check("no mask == plain mean",
      torch.allclose(masked_mean_pool(features, None), features.mean(1)))

print("\n3. the loss masks on validity")
torch.manual_seed(0)
mean, std = torch.zeros(3), torch.ones(3)
pred = torch.randn(4, 3)
target = torch.randn(4, 3)
valid = torch.tensor([True, False, True, False])
poisoned = target.clone()
poisoned[~valid] = 1e3          # garbage in the invalid rows
loss_a, _ = object_pose_loss(pred, target, valid, mean, std)
loss_b, _ = object_pose_loss(pred, poisoned, valid, mean, std)
check("invalid rows do not affect the loss", torch.allclose(loss_a, loss_b),
      f"{float(loss_a):.6f} vs {float(loss_b):.6f}")
subset, _ = object_pose_loss(pred[valid], target[valid], torch.ones(2, dtype=torch.bool), mean, std)
check("equals the loss over valid rows only", torch.allclose(loss_a, subset),
      f"{float(loss_a):.6f} vs {float(subset):.6f}")

print("\n4. an all-invalid batch is zero, not NaN")
pred_g = torch.randn(4, 3, requires_grad=True)
zero_loss, metrics = object_pose_loss(
    pred_g, torch.randn(4, 3), torch.zeros(4, dtype=torch.bool), mean, std)
zero_loss.backward()
check("loss is exactly zero", float(zero_loss) == 0.0)
check("loss is finite", torch.isfinite(zero_loss).item())
check("gradient is finite and zero",
      torch.isfinite(pred_g.grad).all().item() and float(pred_g.grad.abs().sum()) == 0.0)
check("valid_frac reported as 0", metrics["object_pose_valid_frac"] == 0.0)

print("\n5. error is reported in centimetres in the original units")
mean_m = torch.tensor([0.5, 0.0, 0.8])
std_m = torch.tensor([0.1, 0.1, 0.1])
target_m = torch.tensor([[0.5, 0.0, 0.8]])
# prediction is standardised; +1 std on x == +10 cm
pred_m = torch.tensor([[1.0, 0.0, 0.0]])
_, m = object_pose_loss(pred_m, target_m, torch.tensor([True]), mean_m, std_m)
check("10 cm offset reported as 10 cm", abs(m["object_pose_err_cm"] - 10.0) < 1e-3,
      f"got {m['object_pose_err_cm']:.3f}")

print("\n6. statistics come from covered frames only")
values = np.array([[1.0, 1.0, 1.0]] * 5 + [[0.0, 0.0, 0.0]] * 5, np.float32)
valid_np = np.array([True] * 5 + [False] * 5)
mu, sigma = target_statistics(values, valid_np)
check("uncovered zeros excluded from the mean", np.allclose(mu, 1.0), f"got {mu.tolist()}")
check("degenerate std floored above zero", (sigma > 0).all(), f"got {sigma.tolist()}")

print("\n7. head shape, and the gradient reaches the trunk")
head = ObjectPoseHead(feature_dim=8, target_dim=3, hidden_dim=16)
trunk = torch.randn(2, 5, 8, requires_grad=True)
prediction = head(trunk, torch.ones(2, 5, dtype=torch.bool))
check("prediction shape (B, 3)", tuple(prediction.shape) == (2, 3), str(tuple(prediction.shape)))
prediction.sum().backward()
check("gradient flows into the trunk features",
      trunk.grad is not None and float(trunk.grad.abs().sum()) > 0)

print("\n8. the mixin folds the aux term into the policy loss")


class FakePolicy(ObjectPoseAuxMixin, nn.Module):
    def __init__(self, config):
        nn.Module.__init__(self)
        self.config = config
        self._setup_object_pose_aux(8)


config = SimpleNamespace(object_pose_weight=2.0, object_pose_target="position",
                         object_pose_hidden_dim=16, object_pose_store=None,
                         object_pose_detach=False)
policy = FakePolicy(config)
check("head registered as a submodule (so the optimiser sees it)",
      any("object_pose_head" in n for n, _ in policy.named_parameters()))

batch = {POSE_KEY: torch.randn(2, 3), VALID_KEY: torch.tensor([True, True])}
policy._captured_features = (torch.randn(2, 5, 8, requires_grad=True), None)
base = torch.tensor(1.0, requires_grad=True)
total, info = policy._add_object_pose_loss(base, {"loss": 1.0}, batch)
check("total = base + weight * aux",
      abs(float(total) - (1.0 + 2.0 * info["object_pose_loss"])) < 1e-5,
      f"total {float(total):.5f}, aux {info['object_pose_loss']:.5f}")
check("capture cleared after use", policy._captured_features is None)
check("metrics reported", {"object_pose_loss", "object_pose_err_cm"} <= set(info))

print("\n9. weight 0 and a missing target are no-ops, not crashes")
policy._captured_features = (torch.randn(2, 5, 8), None)
same, _ = policy._add_object_pose_loss(torch.tensor(1.0), {}, {})   # no pose in batch
check("missing target leaves the loss untouched", float(same) == 1.0)
config.object_pose_weight = 0.0
policy._captured_features = (torch.randn(2, 5, 8), None)
same0, _ = policy._add_object_pose_loss(torch.tensor(1.0), {}, batch)
check("weight 0 leaves the loss untouched", float(same0) == 1.0)
config.object_pose_weight = 2.0

print("\n10. detach makes the head a passive probe")
config.object_pose_detach = True
probe = FakePolicy(config)
trunk_p = torch.randn(2, 5, 8, requires_grad=True)
probe._captured_features = (trunk_p, None)
total_p, _ = probe._add_object_pose_loss(torch.tensor(0.0, requires_grad=True), {}, batch)
total_p.backward()
check("no gradient reaches the trunk when detached", trunk_p.grad is None)
check("the head itself still gets gradient",
      any(p.grad is not None and float(p.grad.abs().sum()) > 0
          for p in probe.object_pose_head.parameters()))

print("\n11. standardisation against the real kitting store")
try:
    from h2r_il.object_pose_inject import PoseStore
    store = PoseStore(STORE, target="position")
    mu, sigma = target_statistics(store.values.numpy(), store.valid.numpy())
    print(f"     mean {np.round(mu, 4).tolist()}  std {np.round(sigma, 4).tolist()}")
    real = ObjectPoseHead(8, 3, 16)
    real.set_statistics(mu, sigma)
    targets = store.values[:256]
    standardised = (targets - real.target_mean) / real.target_std
    check("standardised target is ~zero mean", abs(float(standardised.mean())) < 0.5,
          f"{float(standardised.mean()):.3f}")
    check("standardised target is ~unit scale", 0.2 < float(standardised.std()) < 3.0,
          f"{float(standardised.std()):.3f}")
    raw_scale = float(((targets - targets.mean(0)) ** 2).mean())
    check("raw targets would have been ~100x smaller than a unit loss", raw_scale < 0.05,
          f"raw variance {raw_scale:.5f} -- this is why the target is standardised")
except FileNotFoundError as exc:
    print(f"     skipped (no store): {exc}")

print()
if failures:
    print(f"FAILED: {len(failures)} check(s): {failures}")
    sys.exit(1)
print("all checks passed")
