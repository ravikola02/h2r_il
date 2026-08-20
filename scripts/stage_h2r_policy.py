#!/usr/bin/env python
"""Re-stage a trained checkpoint under an h2r policy type, without copying weights.

Why this exists: LeRobot resolves a checkpoint's policy class from the ``type``
field in its ``config.json`` and pops that field before applying CLI overrides
(`PreTrainedConfig.from_pretrained`), so ``--policy.type`` cannot switch a
checkpoint from ``pi0`` to ``h2r_pi0``. The type has to be changed on disk.

Copying is not an option -- these checkpoints run to ~22 GB -- so everything
except ``config.json`` is hard-linked (symlinked across filesystems), and only the
rewritten config is a real new file. The original checkpoint is never modified.

The h2r configs are supersets of the ones they extend: every added field has a
default, so an existing config.json parses unchanged and the head starts from its
initialisation while the trunk keeps the trained weights.

    python scripts/stage_h2r_policy.py \\
        --checkpoint outputs/kitting_v1_raw/checkpoints/017000/pretrained_model \\
        --type h2r_groot --out outputs/staged/kitting_v1_raw_h2r
"""

import argparse
import json
import os
import sys
from pathlib import Path

VALID_TYPES = {"h2r_pi0": "pi0", "h2r_groot": "groot"}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True,
                        help="A pretrained_model directory containing config.json.")
    parser.add_argument("--type", required=True, choices=sorted(VALID_TYPES),
                        help="Policy type to stage it as.")
    parser.add_argument("--out", required=True, help="Directory to create.")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite an existing output directory's config.json.")
    args = parser.parse_args(argv)

    source = Path(args.checkpoint).expanduser().resolve()
    config_path = source / "config.json"
    if not config_path.is_file():
        sys.exit(f"no config.json in {source} -- point --checkpoint at a "
                 "pretrained_model directory.")

    config = json.loads(config_path.read_text())
    current = config.get("type")
    expected = VALID_TYPES[args.type]
    if current == args.type:
        print(f"{source} is already staged as {args.type}.")
    elif current != expected:
        # Staging a groot checkpoint as h2r_pi0 would produce a config that parses
        # and a model that fails only once weights are loaded.
        sys.exit(f"checkpoint type is {current!r}, but {args.type!r} extends "
                 f"{expected!r}. Refusing to stage a mismatched pair.")

    destination = Path(args.out).expanduser().absolute()
    destination.mkdir(parents=True, exist_ok=True)
    linked = 0
    for entry in sorted(source.iterdir()):
        if entry.name == "config.json":
            continue
        target = destination / entry.name
        if target.exists() or target.is_symlink():
            continue
        try:
            os.link(entry, target)
        except OSError:
            # Different filesystem, or a directory: a symlink reads the same and
            # still costs nothing.
            target.symlink_to(entry)
        linked += 1

    out_config = destination / "config.json"
    if out_config.exists() and not args.force:
        sys.exit(f"{out_config} already exists; pass --force to rewrite it.")
    config["type"] = args.type
    out_config.write_text(json.dumps(config, indent=2))

    print(f"staged {source}\n    as {destination}  ({linked} entries linked, "
          f"type {current!r} -> {args.type!r})")
    print(f"\nTrain from it with --policy.path={destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
