#!/usr/bin/env python3
"""Build lamp-anchored action features for the fixed double-light set."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


def number(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def maximum(values: list[float]) -> float | None:
    return max(values) if values else None


def median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def event_start(timeline_path: Path) -> float:
    with timeline_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    doubles = [row for row in rows if row.get("side") == "double"]
    if not doubles:
        raise ValueError(f"No double-light event in {timeline_path}")
    return float(doubles[0]["start_time"])


def sustained_onset(samples: list[dict[str, float]], key: str, threshold: float) -> float | None:
    """First sample in the final sustained positive run, tolerating two short gaps."""
    if not samples:
        return None
    positive = [sample.get(key, -999.0) >= threshold for sample in samples]
    end = len(positive) - 1
    # Start from the last positive observation before the lamp.
    while end >= 0 and not positive[end]:
        end -= 1
    if end < 0:
        return None
    start = end
    gaps = 0
    while start > 0:
        if positive[start - 1]:
            gaps = 0
            start -= 1
        elif gaps < 2:
            gaps += 1
            start -= 1
        else:
            break
    positives = sum(positive[start : end + 1])
    if positives < 3:
        return None
    return samples[start]["time"]


def side_samples(frames: list[dict], side: str, lamp_time: float, window_sec: float) -> list[dict[str, float]]:
    direction = 1.0 if side == "left" else -1.0
    samples: list[dict[str, float]] = []
    for frame in frames:
        time_sec = number(frame.get("time_sec"))
        if time_sec is None or time_sec > lamp_time or time_sec < lamp_time - window_sec:
            continue
        athlete = frame.get(side)
        if not isinstance(athlete, dict):
            continue
        features = athlete.get("features") or {}
        box = athlete.get("box") or []
        keypoints = athlete.get("keypoints") or {}
        wrist_name = athlete.get("weapon_wrist")
        wrist = keypoints.get(wrist_name) if wrist_name else None
        shoulders = features.get("shoulders")
        height = number(box[3] - box[1]) if len(box) == 4 else None
        wrist_lead = None
        if isinstance(wrist, dict) and isinstance(shoulders, list) and len(shoulders) == 2 and height and height > 1:
            wrist_x = number(wrist.get("x"))
            shoulder_x = number(shoulders[0])
            if wrist_x is not None and shoulder_x is not None:
                wrist_lead = direction * (wrist_x - shoulder_x) / height
        sample = {
            "time": time_sec,
            "speed": number(features.get("inward_speed_body_heights_per_sec")) or 0.0,
            "elbow": number(features.get("weapon_elbow_angle")) or 0.0,
            "wrist_lead": wrist_lead if wrist_lead is not None else -9.0,
            "stance": number(features.get("stance_ratio")) or 0.0,
            "quality": number(features.get("pose_quality")) or 0.0,
        }
        sample["attack_signal"] = min(3.0, max(-2.0, sample["speed"])) + 0.008 * max(0.0, sample["elbow"] - 110.0) + 0.8 * max(0.0, sample["wrist_lead"])
        samples.append(sample)
    return samples


def metrics(samples: list[dict[str, float]], lamp_time: float) -> dict[str, float | int | None]:
    row: dict[str, float | int | None] = {"samples": len(samples)}
    for seconds, suffix in ((0.35, "035"), (0.75, "075"), (1.5, "150")):
        recent = [sample for sample in samples if sample["time"] >= lamp_time - seconds]
        speeds = [sample["speed"] for sample in recent]
        elbows = [sample["elbow"] for sample in recent if sample["elbow"] > 0]
        wrist = [sample["wrist_lead"] for sample in recent if sample["wrist_lead"] > -8]
        signals = [sample["attack_signal"] for sample in recent]
        row[f"speed_mean_{suffix}"] = mean(speeds)
        row[f"speed_max_{suffix}"] = maximum(speeds)
        row[f"speed_positive_fraction_{suffix}"] = mean([float(value >= 0.18) for value in speeds])
        row[f"elbow_median_{suffix}"] = median(elbows)
        row[f"elbow_max_{suffix}"] = maximum(elbows)
        row[f"wrist_lead_median_{suffix}"] = median(wrist)
        row[f"wrist_lead_max_{suffix}"] = maximum(wrist)
        row[f"attack_signal_mean_{suffix}"] = mean(signals)
        row[f"attack_signal_max_{suffix}"] = maximum(signals)
    for key, threshold in (("speed", 0.18), ("attack_signal", 0.55)):
        onset = sustained_onset(samples, key, threshold)
        row[f"{key}_onset_time"] = onset
        row[f"{key}_lead_time"] = lamp_time - onset if onset is not None else None
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--pose-dir", type=Path, required=True)
    parser.add_argument("--mens-timeline-root", type=Path, required=True)
    parser.add_argument("--womens-timeline-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--window-sec", type=float, default=2.5)
    args = parser.parse_args()

    with args.spec.open("r", encoding="utf-8-sig", newline="") as handle:
        cases = list(csv.DictReader(handle))
    output_rows: list[dict[str, object]] = []
    for case in cases:
        case_id = case["case_id"]
        pose_path = args.pose_dir / f"{case_id}_pose.json"
        pose = json.loads(pose_path.read_text(encoding="utf-8"))
        override = number(case.get("lamp_time_override"))
        if override is not None:
            lamp_time = override
        else:
            timeline_root = args.mens_timeline_root if case["venue"] == "mens" else args.womens_timeline_root
            timeline_path = timeline_root / Path(case["video"]).stem / "timeline.csv"
            lamp_time = event_start(timeline_path)
        output: dict[str, object] = {
            "case_id": case_id,
            "venue": case["venue"],
            "official_outcome": case["official_outcome"],
            "label_confidence": case["confidence"],
            "lamp_time": round(lamp_time, 4),
            "fps": pose.get("fps"),
        }
        for side in ("left", "right"):
            side_metrics = metrics(side_samples(pose["frames"], side, lamp_time, args.window_sec), lamp_time)
            output.update({f"{side}_{key}": value for key, value in side_metrics.items()})
        output_rows.append(output)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(output_rows[0])
    with args.output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)
    print(json.dumps({"cases": len(output_rows), "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
