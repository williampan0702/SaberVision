from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np


def percentile(values: list[float], value: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float32), value))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Conservative quality gate for a v4.7.2 primary optical-flow device track."
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("track", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--end", type=float)
    parser.add_argument("--minimum-coverage", type=float, default=0.35)
    parser.add_argument("--minimum-anchors", type=int, default=20)
    parser.add_argument("--maximum-edge-fraction", type=float, default=0.02)
    parser.add_argument("--maximum-p95-area", type=float, default=0.12)
    parser.add_argument("--maximum-p95-jump", type=float, default=0.035)
    args = parser.parse_args()

    cap = cv2.VideoCapture(str(args.input))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {args.input}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_duration = float(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / max(1.0, float(cap.get(cv2.CAP_PROP_FPS)))
    cap.release()

    with args.track.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    times = [float(row["time_sec"]) for row in rows]
    if len(times) >= 2:
        steps = [b - a for a, b in zip(times, times[1:]) if 0 < b - a <= 0.5]
        fps = 1.0 / max(1e-6, float(np.median(steps))) if steps else 10.0
    else:
        fps = 10.0
    end = video_duration if args.end is None else min(args.end, video_duration)
    duration = max(0.0, end - args.start)
    sampled_frames = max(1, round(duration * fps) + 1)
    coverage = len(rows) / sampled_frames
    diagonal = math.hypot(width, height)
    frame_area = max(1, width * height)

    areas: list[float] = []
    jumps: list[float] = []
    edge_rows = 0
    previous: tuple[float, float, float] | None = None
    for row in rows:
        timestamp = float(row["time_sec"])
        x1, y1, x2, y2 = (float(row[key]) for key in ("x1", "y1", "x2", "y2"))
        areas.append(max(0.0, (x2 - x1) * (y2 - y1)) / frame_area)
        if x1 <= width * 0.01 or y1 <= height * 0.01 or x2 >= width * 0.99 or y2 >= height * 0.99:
            edge_rows += 1
        cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
        if previous is not None:
            old_time, old_x, old_y = previous
            if 0 < timestamp - old_time <= 0.5:
                jumps.append(math.hypot(cx - old_x, cy - old_y) / diagonal)
        previous = timestamp, cx, cy

    anchor_count = sum(row.get("source") == "anchor" for row in rows)
    edge_fraction = edge_rows / max(1, len(rows))
    p95_area = percentile(areas, 95)
    p95_jump = percentile(jumps, 95)
    checks = {
        "coverage": coverage >= args.minimum_coverage,
        "anchor_support": anchor_count >= args.minimum_anchors,
        "not_edge_locked": edge_fraction <= args.maximum_edge_fraction,
        "plausible_area": p95_area <= args.maximum_p95_area,
        "smooth_motion": p95_jump <= args.maximum_p95_jump,
    }
    result = {
        "qualified": all(checks.values()),
        "checks": checks,
        "metrics": {
            "width": width,
            "height": height,
            "duration_sec": duration,
            "start_sec": args.start,
            "end_sec": end,
            "inferred_fps": fps,
            "sampled_frames": sampled_frames,
            "stable_rows": len(rows),
            "coverage": coverage,
            "anchor_rows": anchor_count,
            "edge_fraction": edge_fraction,
            "median_area_fraction": percentile(areas, 50),
            "p95_area_fraction": p95_area,
            "p95_center_jump_norm": p95_jump,
        },
        "thresholds": {
            "minimum_coverage": args.minimum_coverage,
            "minimum_anchors": args.minimum_anchors,
            "maximum_edge_fraction": args.maximum_edge_fraction,
            "maximum_p95_area": args.maximum_p95_area,
            "maximum_p95_jump": args.maximum_p95_jump,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
