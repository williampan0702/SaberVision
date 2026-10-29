#!/usr/bin/env python3
"""Build lamp-anchored temporal Pose features for double-light phrase analysis."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


WINDOWS = ((0.35, "035"), (0.75, "075"), (1.40, "140"))
BIN_EDGES = (-1.40, -1.20, -1.00, -0.80, -0.60, -0.40, -0.20, 0.001)


def number(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def slope(samples: list[dict[str, float]], key: str) -> float | None:
    usable = [(sample["time"], sample[key]) for sample in samples if key in sample]
    if len(usable) < 3:
        return None
    times = [item[0] for item in usable]
    values = [item[1] for item in usable]
    t_mean = statistics.fmean(times)
    v_mean = statistics.fmean(values)
    denom = sum((value - t_mean) ** 2 for value in times)
    if denom <= 1e-9:
        return None
    return sum((time - t_mean) * (value - v_mean) for time, value in usable) / denom


def sustained_onset(samples: list[dict[str, float]], key: str, threshold: float) -> float | None:
    for index in range(max(0, len(samples) - 3)):
        window = samples[index : index + 4]
        if len(window) == 4 and sum(sample.get(key, -999.0) >= threshold for sample in window) >= 3:
            return window[0]["time"]
    return None


def side_series(frames: list[dict], side: str, lamp_time: float) -> list[dict[str, float]]:
    direction = 1.0 if side == "left" else -1.0
    raw: list[dict[str, float]] = []
    heights: list[float] = []
    for frame in frames:
        time_sec = number(frame.get("time_sec"))
        if time_sec is None or not lamp_time - 1.40 <= time_sec <= lamp_time:
            continue
        athlete = frame.get(side)
        if not isinstance(athlete, dict):
            continue
        box = athlete.get("box") or []
        height = number(box[3] - box[1]) if len(box) == 4 else None
        features = athlete.get("features") or {}
        pelvis = features.get("pelvis") or []
        pelvis_x = number(pelvis[0]) if len(pelvis) == 2 else None
        if height is None or height <= 1 or pelvis_x is None:
            continue
        heights.append(height)
        keypoints = athlete.get("keypoints") or {}
        wrist_name = athlete.get("weapon_wrist")
        wrist = keypoints.get(wrist_name) if wrist_name else None
        shoulders = features.get("shoulders") or []
        wrist_x = number(wrist.get("x")) if isinstance(wrist, dict) else None
        shoulder_x = number(shoulders[0]) if len(shoulders) == 2 else None
        raw.append({
            "time": time_sec,
            "speed": number(features.get("inward_speed_body_heights_per_sec")) or 0.0,
            "pelvis_x": pelvis_x,
            "height": height,
            "wrist_lead": direction * (wrist_x - shoulder_x) / height if wrist_x is not None and shoulder_x is not None else math.nan,
            "elbow": number(features.get("weapon_elbow_angle")) or math.nan,
            "stance": number(features.get("stance_ratio")) or math.nan,
            "quality": number(features.get("pose_quality")) or 0.0,
            "lunge": float(features.get("lunge_candidate") is True),
        })
    if not raw:
        return []
    reference_height = median(heights) or 1.0
    for sample in raw:
        sample["inward_position"] = direction * sample["pelvis_x"] / reference_height
    return raw


def window_metrics(samples: list[dict[str, float]], lamp_time: float, seconds: float) -> dict[str, float | None]:
    recent = [sample for sample in samples if sample["time"] >= lamp_time - seconds]
    if not recent:
        return {}
    result: dict[str, float | None] = {}
    speeds = [sample["speed"] for sample in recent]
    result["speed_mean"] = mean(speeds)
    result["speed_median"] = median(speeds)
    result["speed_max"] = max(speeds)
    result["speed_min"] = min(speeds)
    result["speed_positive_fraction"] = mean([float(value >= 0.20) for value in speeds])
    result["speed_reverse_fraction"] = mean([float(value <= -0.20) for value in speeds])
    result["speed_slope"] = slope(recent, "speed")
    result["position_slope"] = slope(recent, "inward_position")
    if len(recent) >= 2:
        result["position_change"] = recent[-1]["inward_position"] - recent[0]["inward_position"]
    for key in ("wrist_lead", "elbow", "stance", "quality", "lunge"):
        values = [sample[key] for sample in recent if math.isfinite(sample.get(key, math.nan))]
        result[f"{key}_mean"] = mean(values)
        result[f"{key}_max"] = max(values) if values else None
        if len(values) >= 2:
            result[f"{key}_change"] = values[-1] - values[0]
    return result


def side_features(samples: list[dict[str, float]], lamp_time: float) -> dict[str, float | int | None]:
    output: dict[str, float | int | None] = {"samples": len(samples)}
    for seconds, suffix in WINDOWS:
        for key, value in window_metrics(samples, lamp_time, seconds).items():
            output[f"{key}_{suffix}"] = value
    for threshold, name in ((0.20, "move"), (0.60, "strong_move")):
        onset = sustained_onset(samples, "speed", threshold)
        output[f"{name}_onset_lead"] = lamp_time - onset if onset is not None else None
    for index, (lo, hi) in enumerate(zip(BIN_EDGES, BIN_EDGES[1:])):
        selected = [sample for sample in samples if lamp_time + lo <= sample["time"] < lamp_time + hi]
        output[f"speed_bin_{index}"] = mean([sample["speed"] for sample in selected])
        output[f"wrist_bin_{index}"] = mean([
            sample["wrist_lead"] for sample in selected if math.isfinite(sample.get("wrist_lead", math.nan))
        ])
        output[f"elbow_bin_{index}"] = mean([
            sample["elbow"] for sample in selected if math.isfinite(sample.get("elbow", math.nan))
        ])
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--pose-dir", type=Path, required=True)
    parser.add_argument("--lamp-feature-files", type=Path, nargs="*", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    lamp_times: dict[str, float] = {}
    for feature_path in args.lamp_feature_files:
        with feature_path.open("r", encoding="utf-8-sig", newline="") as handle:
            for item in csv.DictReader(handle):
                value = number(item.get("lamp_time"))
                if value is not None:
                    lamp_times[item["case_id"]] = value
    with args.spec.open("r", encoding="utf-8-sig", newline="") as handle:
        cases = list(csv.DictReader(handle))
    rows: list[dict[str, object]] = []
    for case in cases:
        case_id = case["case_id"]
        pose = json.loads((args.pose_dir / f"{case_id}_pose.json").read_text(encoding="utf-8"))
        lamp_time = number(case.get("lamp_time_override"))
        if lamp_time is None:
            lamp_time = lamp_times.get(case_id)
        if lamp_time is None:
            raise ValueError(f"Missing lamp time for {case_id}")
        row: dict[str, object] = {
            "case_id": case_id,
            "venue": case["venue"],
            "official_outcome": case["official_outcome"],
            "label_confidence": case.get("confidence", ""),
            "lamp_time": lamp_time,
        }
        for side in ("left", "right"):
            metrics = side_features(side_series(pose.get("frames", []), side, lamp_time), lamp_time)
            row.update({f"{side}_{key}": value for key, value in metrics.items()})
        rows.append(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    leading = ["case_id", "venue", "official_outcome", "label_confidence", "lamp_time"]
    fieldnames = leading + sorted({key for row in rows for key in row if key not in leading})
    with args.output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"cases": len(rows), "features": len(fieldnames) - 5, "output": str(args.output.resolve())}, ensure_ascii=False))


if __name__ == "__main__":
    main()
