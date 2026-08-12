#!/usr/bin/env python
"""End-to-end check: does injection actually land on real LeRobot samples?"""
import sys

import numpy as np
import torch

sys.path.insert(0, "/home/ravikola/dev/h2r_il/.claude/worktrees/object-grounding-aux/src")

from lerobot.datasets.lerobot_dataset import LeRobotDataset

from h2r_il.object_pose_inject import POSE_KEY, VALID_KEY, PoseStore, wrap_dataset

STORE = "/mnt/shared_data/h2r_il/outputs/object_pose/kitting/head/store"
ROOT = "/mnt/shared_data/h2r_il/datasets/kitting"

store = PoseStore(STORE, target="position")
ds = LeRobotDataset(repo_id="local/kitting", root=ROOT)
print(f"dataset frames {len(ds)}, store frames {len(store)}")
wrap_dataset(ds, store)

raw = np.load(f"{STORE}/object_pose.npz")

# Spot-check across episodes, including an episode boundary.
for i in (0, 225, 226, 5000, 12345, len(ds) - 1):
    item = ds[i]
    assert POSE_KEY in item and VALID_KEY in item, "keys missing"
    got = item[POSE_KEY].numpy()
    want = raw["position"][int(item["index"])]
    ok = np.allclose(got, want)
    print(f"  idx {i:6d}  index={int(item['index']):6d}  ep={int(item['episode_index']):3d}  "
          f"pose={np.round(got, 4)}  valid={bool(item[VALID_KEY])}  matches_store={ok}")
    assert ok, "pose does not match the store"

# The batch must survive default collation, which is what the DataLoader does.
loader = torch.utils.data.DataLoader(ds, batch_size=4, num_workers=0, shuffle=True)
batch = next(iter(loader))
print(f"\ncollated batch keys include {POSE_KEY}: {POSE_KEY in batch}")
print(f"  {POSE_KEY} {tuple(batch[POSE_KEY].shape)} {batch[POSE_KEY].dtype}")
print(f"  {VALID_KEY} {tuple(batch[VALID_KEY].shape)} {batch[VALID_KEY].dtype}")
print(f"  other keys unaffected: {sorted(k for k in batch if not k.startswith('observation.images'))[:6]}")
print("\nOK")
