from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from track_triple_panel_layout_v47 import (
    add_region_columns,
    candidate,
    region_geometry,
    smooth_regions,
    synthesize_middle,
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def choose_relaxed_pair(rows: list[dict[str, str]], width: int, height: int):
    items = [candidate(row) for row in rows]
    items = [
        item for item in items
        if width * 0.06 <= item["w"] <= width * 0.31
        and height * 0.045 <= item["h"] <= height * 0.27
        and item["confidence"] >= 0.05
    ]
    best = None
    for left_index, first in enumerate(items):
        for second in items[left_index + 1:]:
            left, right = (first, second) if first["cx"] <= second["cx"] else (second, first)
            separation = right["cx"] - left["cx"]
            midpoint = (left["cx"] + right["cx"]) * 0.5
            if not width * 0.28 <= separation <= width * 0.75:
                continue
            if not width * 0.36 <= midpoint <= width * 0.64:
                continue
            if not (left["cx"] <= width * 0.50 and right["cx"] >= width * 0.50):
                continue
            if abs(left["cy"] - right["cy"]) > height * 0.10:
                continue
            width_ratio = max(left["w"], right["w"]) / max(1.0, min(left["w"], right["w"]))
            height_ratio = max(left["h"], right["h"]) / max(1.0, min(left["h"], right["h"]))
            # A close-up scoreboard can be split into two detector boxes.  Its
            # two fragments have noticeably different heights; real left/right
            # synchronized panels remain more alike even when one is clipped.
            if width_ratio > 1.75 or height_ratio > 1.65:
                continue
            confidence_sum = left["confidence"] + right["confidence"]
            if confidence_sum < 0.12:
                continue
            symmetry_penalty = abs(separation / width - 0.47)
            midpoint_penalty = abs(midpoint / width - 0.50)
            shape_penalty = 0.08 * (width_ratio - 1.0) + 0.08 * (height_ratio - 1.0)
            score = confidence_sum - symmetry_penalty - midpoint_penalty - shape_penalty
            if best is None or score > best[0]:
                best = (score, left, right)
    return best


def regions_from_row(row: dict[str, str]):
    if row.get("valid") != "1":
        return {}
    output = {}
    for name in ("left_panel", "center_box", "right_panel"):
        prefix = name
        box = tuple(round(float(row[f"{prefix}_{key}"])) for key in ("x1", "y1", "x2", "y2"))
        left = tuple(round(float(row[f"{prefix}_left_{key}"])) for key in ("x1", "y1", "x2", "y2"))
        right = tuple(round(float(row[f"{prefix}_right_{key}"])) for key in ("x1", "y1", "x2", "y2"))
        output[name] = {"box": box, "left_light": left, "right_light": right}
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Recover strict-track misses using a geometry-gated relaxed side-panel pair.")
    parser.add_argument("track", type=Path)
    parser.add_argument("anchors", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--visibility", type=Path)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--smooth-gap", type=float, default=0.8)
    args = parser.parse_args()

    rows = read_csv(args.track)
    grouped: dict[float, list[dict[str, str]]] = defaultdict(list)
    for row in read_csv(args.anchors):
        grouped[round(float(row["time_sec"]), 3)].append(row)
    visibility = read_csv(args.visibility) if args.visibility else None
    if visibility is not None and len(visibility) != len(rows):
        raise ValueError(f"visibility mismatch: {len(visibility)} != {len(rows)}")

    recovered = 0
    recovered_observable = 0
    previous_regions = {}
    previous_time = float("-inf")
    for index, row in enumerate(rows):
        timestamp = round(float(row["time_sec"]), 3)
        if row.get("valid") == "1":
            previous_regions = regions_from_row(row)
            previous_time = timestamp
            continue
        pair = choose_relaxed_pair(grouped.get(timestamp, []), args.width, args.height)
        if pair is None:
            continue
        middle, score, left, right = synthesize_middle(pair, args.width, args.height)
        regions = region_geometry(left, middle, right)
        if timestamp - previous_time <= args.smooth_gap:
            regions = smooth_regions(previous_regions, regions, args.width, args.height, alpha=0.32)
        row["valid"] = "1"
        row["source"] = "three_region_relaxed_anchor_v49"
        row["confidence"] = f"{score:.6f}"
        row.update(dict(zip(("x1", "y1", "x2", "y2"), regions["center_box"]["box"])))
        add_region_columns(row, regions)
        previous_regions = regions
        previous_time = timestamp
        recovered += 1
        if visibility is not None and visibility[index].get("loose_observable") == "1":
            recovered_observable += 1

    args.output.mkdir(parents=True, exist_ok=True)
    output_path = args.output / "triple_panel_track_v49.csv"
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    valid = sum(row.get("valid") == "1" for row in rows)
    summary = {
        "sampled_frames": len(rows),
        "valid_frames": valid,
        "valid_fraction": valid / max(1, len(rows)),
        "relaxed_recovered_frames": recovered,
        "relaxed_recovered_observable_frames": recovered_observable,
        "rule": "current-frame dual-side anchors; confidence>=0.05 each, sum>=0.12, centre-straddling and bounded symmetry/shape",
    }
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
