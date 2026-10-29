from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def box(row: dict[str, str], prefix: str = "center_box"):
    return tuple(float(row[f"{prefix}_{key}"]) for key in ("x1", "y1", "x2", "y2"))


def compatible(left: dict[str, str], right: dict[str, str], width: int, height: int, maximum_distance: float, maximum_area_ratio: float) -> bool:
    first = box(left)
    second = box(right)
    first_center = ((first[0] + first[2]) * 0.5, (first[1] + first[3]) * 0.5)
    second_center = ((second[0] + second[2]) * 0.5, (second[1] + second[3]) * 0.5)
    distance = math.hypot(first_center[0] - second_center[0], first_center[1] - second_center[1]) / math.hypot(width, height)
    first_area = max(1.0, (first[2] - first[0]) * (first[3] - first[1]))
    second_area = max(1.0, (second[2] - second[0]) * (second[3] - second[1]))
    area_ratio = max(first_area, second_area) / min(first_area, second_area)
    return distance <= maximum_distance and area_ratio <= maximum_area_ratio


def main() -> None:
    parser = argparse.ArgumentParser(description="Conservatively interpolate short same-shot gaps in a three-region track.")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--maximum-gap", type=float, default=1.2)
    parser.add_argument("--maximum-distance", type=float, default=0.075)
    parser.add_argument("--maximum-area-ratio", type=float, default=1.8)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()
    with args.input.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    coordinate_columns = [
        key for key in rows[0]
        if key.endswith(("_x1", "_y1", "_x2", "_y2")) or key in {"x1", "y1", "x2", "y2"}
    ]
    interpolated = 0
    index = 0
    while index < len(rows):
        if rows[index].get("valid") == "1":
            index += 1
            continue
        start = index
        while index < len(rows) and rows[index].get("valid") != "1":
            index += 1
        end = index - 1
        left_index = start - 1
        right_index = index
        if left_index < 0 or right_index >= len(rows):
            continue
        gap_seconds = (right_index - left_index) / args.fps
        left = rows[left_index]
        right = rows[right_index]
        if gap_seconds > args.maximum_gap or not compatible(
            left, right, args.width, args.height, args.maximum_distance, args.maximum_area_ratio
        ):
            continue
        span = right_index - left_index
        for row_index in range(start, end + 1):
            fraction = (row_index - left_index) / span
            row = rows[row_index]
            row["valid"] = "1"
            row["source"] = "three_region_interpolated"
            left_confidence = float(left.get("confidence") or 0.0)
            right_confidence = float(right.get("confidence") or 0.0)
            row["confidence"] = f"{((1.0 - fraction) * left_confidence + fraction * right_confidence):.6f}"
            for column in coordinate_columns:
                if not left.get(column) or not right.get(column):
                    continue
                value = (1.0 - fraction) * float(left[column]) + fraction * float(right[column])
                row[column] = str(round(value))
            for region in ("left_panel", "center_box", "right_panel"):
                row[f"{region}_valid"] = "1"
            interpolated += 1
    args.output.mkdir(parents=True, exist_ok=True)
    output_path = args.output / "triple_panel_track_interpolated.csv"
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    valid = sum(row.get("valid") == "1" for row in rows)
    summary = {
        "sampled_frames": len(rows),
        "valid_frames": valid,
        "interpolated_frames": interpolated,
        "valid_fraction": valid / max(1, len(rows)),
        "maximum_gap": args.maximum_gap,
        "maximum_distance": args.maximum_distance,
        "maximum_area_ratio": args.maximum_area_ratio,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
