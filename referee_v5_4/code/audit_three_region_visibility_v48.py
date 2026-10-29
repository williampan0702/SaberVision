from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def loose_visible(rows: list[dict[str, str]], width: int, height: int) -> bool:
    items = []
    for row in rows:
        x1, y1, x2, y2 = (float(row[key]) for key in ("x1", "y1", "x2", "y2"))
        w, h = x2 - x1, y2 - y1
        confidence = float(row["confidence"])
        if not (width * 0.05 <= w <= width * 0.36 and height * 0.04 <= h <= height * 0.31 and confidence >= 0.02):
            continue
        items.append({"w": w, "h": h, "cx": (x1 + x2) * 0.5, "cy": (y1 + y2) * 0.5})
    for index, left in enumerate(items):
        for right in items[index + 1:]:
            if left["cx"] > right["cx"]:
                left, right = right, left
            separation = right["cx"] - left["cx"]
            if not width * 0.20 <= separation <= width * 0.80:
                continue
            if abs(left["cy"] - right["cy"]) > height * 0.15:
                continue
            width_ratio = max(left["w"], right["w"]) / max(1.0, min(left["w"], right["w"]))
            height_ratio = max(left["h"], right["h"]) / max(1.0, min(left["h"], right["h"]))
            if width_ratio <= 2.5 and height_ratio <= 2.5:
                return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate visible-frame coverage using a loose independent side-panel pair gate.")
    parser.add_argument("anchors", type=Path)
    parser.add_argument("track", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()
    anchors = defaultdict(list)
    for row in read_csv(args.anchors):
        if not row.get("time_sec"):
            continue
        anchors[round(float(row["time_sec"]), 3)].append(row)
    track = read_csv(args.track)
    audit = []
    for row in track:
        timestamp = round(float(row["time_sec"]), 3)
        observable = loose_visible(anchors.get(timestamp, []), args.width, args.height)
        strict = row.get("valid") == "1"
        audit.append({
            "time_sec": f"{timestamp:.3f}",
            "loose_observable": int(observable),
            "strict_three_region": int(strict),
            "covered_when_observable": int(observable and strict),
        })
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "visibility_audit.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(audit[0]))
        writer.writeheader()
        writer.writerows(audit)
    observable_count = sum(row["loose_observable"] for row in audit)
    covered_count = sum(row["covered_when_observable"] for row in audit)
    false_negative_count = observable_count - covered_count
    buckets = []
    for minute_start in range(0, 120, 10):
        start = minute_start * 60
        end = (minute_start + 10) * 60
        segment = [row for row in audit if start <= float(row["time_sec"]) < end]
        observable = sum(row["loose_observable"] for row in segment)
        covered = sum(row["covered_when_observable"] for row in segment)
        buckets.append({
            "minute_start": minute_start,
            "sampled": len(segment),
            "loose_observable": observable,
            "strict_covered": covered,
            "visible_frame_coverage": covered / max(1, observable),
        })
    summary = {
        "sampled_frames": len(audit),
        "loose_observable_frames": observable_count,
        "strict_covered_observable_frames": covered_count,
        "loose_observable_false_negatives": false_negative_count,
        "visible_frame_coverage": covered_count / max(1, observable_count),
        "raw_all_frame_strict_coverage": sum(row["strict_three_region"] for row in audit) / max(1, len(audit)),
        "definition": "loose dual-side-panel geometry is the observable denominator; strict symmetric triple-panel layout is the numerator",
        "buckets": buckets,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
