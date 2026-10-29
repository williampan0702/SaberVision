from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from statistics import median

from evaluate_action_rules import (
    evaluate_clip,
    find_exchange_phase,
    load_json,
    side_features,
    smooth_median,
)
from tune_lunge_rule_v2 import resolve_pose, score


BASE = {
    "movement_min_votes": 3,
    "lunge_min_run": 1,
    "method": "phase_net",
    "net_movement_threshold": 0.10,
    "lunge_source": "leg_geometry",
    "stance_threshold": 0.36,
    "speed_threshold": 0.02,
    "elbow_threshold": 110.0,
    "bent_knee_max": 135.0,
    "rear_knee_min": 145.0,
    "dynamic_fallback": True,
    "dynamic_stance": 0.28,
    "dynamic_speed": 0.85,
    "dynamic_elbow": 130.0,
    "dynamic_bent_knee_max": 155.0,
    "dynamic_rear_knee_min": 140.0,
}


def phase_features(data: dict, side: str) -> dict[str, float]:
    start_frame, contact_frame = find_exchange_phase(data)
    rows = []
    distances = []
    for frame in data.get("frames", []):
        frame_no = int(frame.get("frame", -1))
        if start_frame is not None and frame_no < start_frame:
            continue
        if contact_frame is not None and frame_no > contact_frame + 3:
            continue
        feat = side_features(frame, side)
        if not feat:
            continue
        rows.append(feat)
        distance = frame.get("opponent_distance_body_heights")
        if distance is not None:
            distances.append(float(distance))

    movement = evaluate_clip(data=data, side=side, **BASE)
    speeds = [
        float(row.get("inward_speed_body_heights_per_sec") or 0.0)
        for row in rows
    ]
    stances = [
        float(row["stance_ratio"])
        for row in rows if row.get("stance_ratio") is not None
    ]
    elbows = [
        float(row["weapon_elbow_angle"])
        for row in rows if row.get("weapon_elbow_angle") is not None
    ]
    forward_leans = []
    for row in rows:
        lean = row.get("torso_lean_deg")
        if lean is not None:
            forward_leans.append(float(lean) if side == "left" else -float(lean))

    closure = 0.0
    if len(distances) >= 3:
        smoothed = smooth_median(distances, radius=1)
        edge = max(1, min(3, len(smoothed) // 3))
        closure = median(smoothed[:edge]) - min(smoothed)

    return {
        "inward_net": float(movement["inward_net_body_heights"] or 0.0),
        "max_speed": max(speeds, default=0.0),
        "distance_closure": closure,
        "max_forward_lean": max(forward_leans, default=0.0),
        "max_stance": max(stances, default=0.0),
        "max_elbow": max(elbows, default=0.0),
        "base_lunge": bool(movement["predicted_lunge"]),
    }


def stage_flag(features: dict[str, float], thresholds: dict[str, float]) -> bool:
    return bool(
        features["inward_net"] >= thresholds["inward_net"]
        and features["max_speed"] >= thresholds["max_speed"]
        and features["distance_closure"] >= thresholds["distance_closure"]
        and features["max_forward_lean"] >= thresholds["forward_lean"]
        and features["max_stance"] >= thresholds["stance"]
        and features["max_elbow"] >= thresholds["elbow"]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top", type=int, default=30)
    args = parser.parse_args()

    config = load_json(args.config)
    datasets = {}
    feature_rows = []
    for item in config["datasets"]:
        labels = load_json(Path(item["labels"]))["clips"]
        pose_dirs = [Path(value) for value in item["pose_dirs"]]
        examples = []
        for clip_id in sorted(labels):
            data = load_json(resolve_pose(clip_id, pose_dirs))
            for side in ("left", "right"):
                features = phase_features(data, side)
                gt = bool(labels[clip_id][side]["lunge_present"])
                examples.append((gt, features))
                feature_rows.append({
                    "dataset": item["name"], "clip_id": clip_id,
                    "side": side, "gt_lunge": gt, **features,
                })
        datasets[item["name"]] = {
            "examples": examples,
            "minimum_f1": float(item["minimum_f1"]),
        }

    valid = []
    rejected = 0
    grid = itertools.product(
        (0.35, 0.45, 0.55),
        (0.65, 0.75, 0.85),
        (0.40, 0.55, 0.70),
        (12.0, 18.0, 24.0),
        (0.40, 0.50, 0.60),
        (135.0, 145.0, 155.0),
    )
    for inward_net, max_speed, distance_closure, lean, stance, elbow in grid:
        thresholds = {
            "inward_net": inward_net,
            "max_speed": max_speed,
            "distance_closure": distance_closure,
            "forward_lean": lean,
            "stance": stance,
            "elbow": elbow,
        }
        metrics = {}
        passes = True
        for name, dataset in datasets.items():
            rows = [
                (gt, bool(features["base_lunge"] or stage_flag(features, thresholds)))
                for gt, features in dataset["examples"]
            ]
            metrics[name] = score(rows)
            if metrics[name]["f1"] + 1e-12 < dataset["minimum_f1"]:
                passes = False
                break
        if not passes:
            rejected += 1
            continue
        target = metrics[config["optimization_target"]]
        regression_f1 = sum(
            value["f1"] for name, value in metrics.items()
            if name != config["optimization_target"]
        )
        rank = (
            target["f1"], target["precision"], regression_f1,
            -target["fp"], target["recall"],
        )
        valid.append((rank, thresholds, metrics))

    valid.sort(key=lambda item: item[0], reverse=True)
    report = {
        "base_rule": BASE,
        "constraints": {
            name: dataset["minimum_f1"] for name, dataset in datasets.items()
        },
        "grid_candidates": rejected + len(valid),
        "rejected_by_constraints": rejected,
        "valid_candidates": len(valid),
        "best": (
            {"thresholds": valid[0][1], "metrics": valid[0][2]}
            if valid else None
        ),
        "top_valid": [
            {"thresholds": thresholds, "metrics": metrics}
            for _, thresholds, metrics in valid[: args.top]
        ],
        "feature_rows": feature_rows,
        "warning": "AI visual labels are engineering references, not expert ground truth.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in report.items() if k != "feature_rows"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
