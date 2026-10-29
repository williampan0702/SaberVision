from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def read(path: Path, source: str) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [{**row, "locator_source": source} for row in csv.DictReader(handle)]


def iou(a: dict[str, str], b: dict[str, str]) -> float:
    ax1, ay1, ax2, ay2 = (float(a[key]) for key in ("x1", "y1", "x2", "y2"))
    bx1, by1, bx2, by2 = (float(b[key]) for key in ("x1", "y1", "x2", "y2"))
    ix1, iy1, ix2, iy2 = max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - intersection
    return intersection / union if union > 0 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Fuse complementary V5.2 score-device anchor models.")
    parser.add_argument("primary", type=Path)
    parser.add_argument("fallback", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--duplicate-iou", type=float, default=0.75)
    args = parser.parse_args()
    grouped: dict[float, list[dict[str, str]]] = defaultdict(list)
    for row in read(args.primary, "primary_v49") + read(args.fallback, "legacy_v46b"):
        grouped[round(float(row["time_sec"]), 3)].append(row)
    merged = []
    duplicates = 0
    source_counts = defaultdict(int)
    for timestamp in sorted(grouped):
        kept = []
        for row in sorted(grouped[timestamp], key=lambda item: float(item["confidence"]), reverse=True):
            if any(iou(row, other) >= args.duplicate_iou for other in kept):
                duplicates += 1
                continue
            kept.append(row)
        for detection, row in enumerate(kept):
            row["detection"] = detection
            source_counts[row["locator_source"]] += 1
            merged.append(row)
    args.output.mkdir(parents=True, exist_ok=True)
    fields = list(merged[0]) if merged else [
        "time_sec", "detection", "confidence", "x1", "y1", "x2", "y2",
        "saturated_fraction", "signal_fraction", "bright_fraction", "edge_fraction",
        "gray_std", "locator_source",
    ]
    with (args.output / "device_anchors.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(merged)
    summary = {
        "primary_candidates": sum(len(rows) for rows in [read(args.primary, "primary_v49")]),
        "fallback_candidates": sum(len(rows) for rows in [read(args.fallback, "legacy_v46b")]),
        "merged_candidates": len(merged),
        "duplicates_removed": duplicates,
        "source_counts": dict(source_counts),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
