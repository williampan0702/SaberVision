from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def center_box(row: dict[str, str]) -> tuple[float, float, float]:
    x1, y1, x2, y2 = (float(row[f"center_box_{key}"]) for key in ("x1", "y1", "x2", "y2"))
    return (0.5 * (x1 + x2), 0.5 * (y1 + y2), max(1.0, (x2 - x1) * (y2 - y1)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify observable miss runs in a three-region track.")
    parser.add_argument("track", type=Path)
    parser.add_argument("visibility", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--end", type=float, default=5791.0)
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()

    track = [row for row in read_csv(args.track) if float(row["time_sec"]) <= args.end]
    visibility = [row for row in read_csv(args.visibility) if float(row["time_sec"]) <= args.end]
    if len(track) != len(visibility):
        raise ValueError(f"row mismatch: track={len(track)} visibility={len(visibility)}")

    output_rows: list[dict[str, object]] = []
    index = 0
    diagonal = math.hypot(args.width, args.height)
    while index < len(track):
        if track[index].get("valid") == "1":
            index += 1
            continue
        start = index
        while index < len(track) and track[index].get("valid") != "1":
            index += 1
        end = index - 1
        miss_indices = [
            i for i in range(start, end + 1)
            if visibility[i].get("loose_observable") == "1"
        ]
        if not miss_indices:
            continue
        left_index = start - 1 if start > 0 and track[start - 1].get("valid") == "1" else None
        right_index = index if index < len(track) and track[index].get("valid") == "1" else None
        distance = None
        area_ratio = None
        if left_index is not None and right_index is not None:
            left_cx, left_cy, left_area = center_box(track[left_index])
            right_cx, right_cy, right_area = center_box(track[right_index])
            distance = math.hypot(left_cx - right_cx, left_cy - right_cy) / diagonal
            area_ratio = max(left_area, right_area) / min(left_area, right_area)
        bracket_seconds = (
            (right_index - left_index) / args.fps
            if left_index is not None and right_index is not None
            else None
        )
        if left_index is None or right_index is None:
            category = "open_ended"
        elif bracket_seconds <= 2.0 and distance <= 0.10 and area_ratio <= 2.2:
            category = "short_fill_candidate"
        elif bracket_seconds <= 3.0:
            category = "short_geometry_change"
        else:
            category = "long_gap_or_cut"
        output_rows.append({
            "start_sec": track[start]["time_sec"],
            "end_sec": track[end]["time_sec"],
            "gap_frames": end - start + 1,
            "gap_seconds": round((end - start + 1) / args.fps, 3),
            "observable_miss_frames": len(miss_indices),
            "first_observable_sec": track[miss_indices[0]]["time_sec"],
            "last_observable_sec": track[miss_indices[-1]]["time_sec"],
            "bracket_seconds": "" if bracket_seconds is None else round(bracket_seconds, 3),
            "center_distance_norm": "" if distance is None else round(distance, 6),
            "center_area_ratio": "" if area_ratio is None else round(area_ratio, 4),
            "category": category,
        })

    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "miss_runs.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    category_runs = Counter(str(row["category"]) for row in output_rows)
    category_frames = Counter()
    for row in output_rows:
        category_frames[str(row["category"])] += int(row["observable_miss_frames"])
    summary = {
        "runs": len(output_rows),
        "observable_miss_frames": sum(int(row["observable_miss_frames"]) for row in output_rows),
        "category_runs": dict(category_runs),
        "category_observable_miss_frames": dict(category_frames),
        "top_runs": sorted(output_rows, key=lambda row: int(row["observable_miss_frames"]), reverse=True)[:20],
    }
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
