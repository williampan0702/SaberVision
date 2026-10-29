#!/usr/bin/env python3
"""Audit whether per-frame weapon-wrist index changes create trajectory jumps."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


def median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pose-dir", type=Path, nargs="+", required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    rows: list[dict[str, object]] = []
    all_switch_jumps: list[float] = []
    all_steady_jumps: list[float] = []
    for pose_dir in args.pose_dir:
        for path in sorted(pose_dir.glob("*_pose.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            for side in ("left", "right"):
                previous_name = None
                previous_xy = None
                switches = 0
                observations = 0
                switch_jumps: list[float] = []
                steady_jumps: list[float] = []
                for frame in payload.get("frames", []):
                    athlete = frame.get(side) or {}
                    name = athlete.get("weapon_wrist")
                    point = (athlete.get("keypoints") or {}).get(name) if name else None
                    box = athlete.get("box") or []
                    if not isinstance(point, dict) or len(box) != 4:
                        continue
                    height = max(1.0, float(box[3]) - float(box[1]))
                    xy = (float(point["x"]), float(point["y"]))
                    observations += 1
                    if previous_xy is not None:
                        jump = ((xy[0] - previous_xy[0]) ** 2 + (xy[1] - previous_xy[1]) ** 2) ** 0.5 / height
                        if previous_name != name:
                            switches += 1
                            switch_jumps.append(jump)
                            all_switch_jumps.append(jump)
                        else:
                            steady_jumps.append(jump)
                            all_steady_jumps.append(jump)
                    previous_name = name
                    previous_xy = xy
                rows.append({
                    "case_id": path.stem.removesuffix("_pose"),
                    "side": side,
                    "observations": observations,
                    "switches": switches,
                    "switch_rate": round(switches / max(1, observations - 1), 5),
                    "median_switch_jump_body_heights": median(switch_jumps),
                    "median_steady_jump_body_heights": median(steady_jumps),
                    "large_switch_jumps": sum(value >= 0.18 for value in switch_jumps),
                })
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "tracks": len(rows),
        "tracks_with_switch": sum(int(row["switches"]) > 0 for row in rows),
        "switches": sum(int(row["switches"]) for row in rows),
        "large_switch_jumps": sum(int(row["large_switch_jumps"]) for row in rows),
        "median_switch_jump_body_heights": median(all_switch_jumps),
        "median_steady_jump_body_heights": median(all_steady_jumps),
        "interpretation": "Low-jump index changes usually indicate pose left/right relabelling; large jumps are likely harmful weapon-arm swaps.",
    }
    args.output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
