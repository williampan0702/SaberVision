from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np


BOX_KEYS = ("x1", "y1", "x2", "y2")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Conservatively interpolate short same-shot gaps in a qualified v4.7.2 device track."
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("track", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--end", type=float, required=True)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--maximum-gap", type=float, default=25.0)
    parser.add_argument("--maximum-center-distance", type=float, default=0.16)
    parser.add_argument("--maximum-area-ratio", type=float, default=1.9)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    with args.track.open("r", encoding="utf-8-sig", newline="") as handle:
        original = list(csv.DictReader(handle))
    original.sort(key=lambda row: float(row["time_sec"]))

    cap = cv2.VideoCapture(str(args.input))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {args.input}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    diagonal = math.hypot(width, height)

    output_rows: list[dict[str, object]] = []
    accepted_gaps = []
    rejected_gaps = []
    for index, row in enumerate(original):
        output_rows.append(dict(row))
        if index + 1 >= len(original):
            continue
        following = original[index + 1]
        start_time = float(row["time_sec"])
        end_time = float(following["time_sec"])
        gap = end_time - start_time
        if gap <= 1.5 / args.fps:
            continue
        first = np.array([float(row[key]) for key in BOX_KEYS], dtype=np.float32)
        second = np.array([float(following[key]) for key in BOX_KEYS], dtype=np.float32)
        first_center = ((first[0] + first[2]) * 0.5, (first[1] + first[3]) * 0.5)
        second_center = ((second[0] + second[2]) * 0.5, (second[1] + second[3]) * 0.5)
        center_distance = math.hypot(
            second_center[0] - first_center[0], second_center[1] - first_center[1]
        ) / diagonal
        first_area = max(1.0, float((first[2] - first[0]) * (first[3] - first[1])))
        second_area = max(1.0, float((second[2] - second[0]) * (second[3] - second[1])))
        area_ratio = max(first_area, second_area) / min(first_area, second_area)
        accepted = (
            gap <= args.maximum_gap
            and center_distance <= args.maximum_center_distance
            and area_ratio <= args.maximum_area_ratio
        )
        audit = {
            "start_sec": start_time,
            "end_sec": end_time,
            "gap_sec": round(gap, 3),
            "center_distance_norm": round(center_distance, 6),
            "area_ratio": round(area_ratio, 6),
        }
        if not accepted:
            rejected_gaps.append(audit)
            continue
        accepted_gaps.append(audit)
        steps = round(gap * args.fps)
        for step in range(1, steps):
            timestamp = round(start_time + step / args.fps, 3)
            if timestamp >= end_time - 1e-6:
                break
            alpha = (timestamp - start_time) / gap
            box = (1.0 - alpha) * first + alpha * second
            output_rows.append(
                {
                    "time_sec": timestamp,
                    "scene_cut": 0,
                    "confidence": 0.0,
                    "x1": round(float(box[0])),
                    "y1": round(float(box[1])),
                    "x2": round(float(box[2])),
                    "y2": round(float(box[3])),
                    "source": "interpolated_gap",
                    "flow_points": 0,
                    "seconds_since_anchor": "",
                }
            )

    output_rows.sort(key=lambda row: float(row["time_sec"]))
    fields = [
        "time_sec",
        "scene_cut",
        "confidence",
        "x1",
        "y1",
        "x2",
        "y2",
        "source",
        "flow_points",
        "seconds_since_anchor",
    ]
    track_path = args.output / "stable_device_track_interpolated.csv"
    with track_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output_rows)
    sampled_frames = round((args.end - args.start) * args.fps) + 1
    summary = {
        "original_rows": len(original),
        "interpolated_rows": len(output_rows) - len(original),
        "final_rows": len(output_rows),
        "sampled_frames": sampled_frames,
        "coverage": len(output_rows) / max(1, sampled_frames),
        "accepted_gaps": accepted_gaps,
        "rejected_gaps": rejected_gaps,
        "settings": {
            "maximum_gap": args.maximum_gap,
            "maximum_center_distance": args.maximum_center_distance,
            "maximum_area_ratio": args.maximum_area_ratio,
        },
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
