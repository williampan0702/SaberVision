#!/usr/bin/env python3
"""Extract the Pose-only extension timing used by the MVP double-light gate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
BLADE_V2 = HERE.parent / "blade_v2"
if str(BLADE_V2) not in sys.path:
    sys.path.insert(0, str(BLADE_V2))

from blade_action_timing import detect_attack_extension  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pose-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for pose_path in sorted(args.pose_dir.glob("*_pose.json")):
        pose = json.loads(pose_path.read_text(encoding="utf-8"))
        frames = pose.get("frames", [])
        payload = {
            "schema": "sabervision_pose_extension_timing_v1",
            "attack_extension_start": {
                side: detect_attack_extension(frames, side) for side in ("left", "right")
            },
            "limitations": [
                "Pose extension does not prove attack priority.",
                "This file contains no blade-contact or parry evidence.",
            ],
        }
        case_id = pose_path.stem.removesuffix("_pose")
        (args.output_dir / f"{case_id}_timing.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        count += 1
    print(json.dumps({"cases": count, "output": str(args.output_dir.resolve())}, ensure_ascii=False))


if __name__ == "__main__":
    main()
