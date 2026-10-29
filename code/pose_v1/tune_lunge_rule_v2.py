from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

from evaluate_action_rules import evaluate_clip, load_json


def score(rows):
    tp = sum(gt and pred for gt, pred in rows)
    fp = sum((not gt) and pred for gt, pred in rows)
    fn = sum(gt and (not pred) for gt, pred in rows)
    tn = sum((not gt) and (not pred) for gt, pred in rows)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    tpr = tp / (tp + fn) if tp + fn else 0.0
    tnr = tn / (tn + fp) if tn + fp else 0.0
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall, "f1": f1,
        "balanced_accuracy": 0.5 * (tpr + tnr),
    }


def resolve_pose(clip_id: str, pose_dirs: list[Path]) -> Path:
    candidates = []
    for pose_dir in pose_dirs:
        descriptive = pose_dir / f"{clip_id}_pose.json"
        if descriptive.exists():
            candidates.append(descriptive)
        candidates.extend(sorted(pose_dir.glob(f"exchange_{clip_id}_*.json")))
    if not candidates:
        raise FileNotFoundError(f"No Pose JSON for {clip_id}")
    return candidates[0]


def predict(data, side, params):
    return evaluate_clip(
        data=data,
        side=side,
        movement_min_votes=3,
        lunge_min_run=params["min_run"],
        method="phase_net",
        net_movement_threshold=0.10,
        lunge_source="leg_geometry",
        stance_threshold=params["stance"],
        speed_threshold=params["speed"],
        elbow_threshold=params["elbow"],
        bent_knee_max=params["bent_knee"],
        rear_knee_min=params["rear_knee"],
    )["predicted_lunge"]


def main():
    parser = argparse.ArgumentParser(
        description="Tune lunge thresholds on one calibration set."
    )
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--pose-dir", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()

    labels = load_json(args.labels)["clips"]
    data_by_clip = {
        clip_id: load_json(resolve_pose(clip_id, args.pose_dir))
        for clip_id in labels
    }
    truth = [
        (clip_id, side, bool(labels[clip_id][side]["lunge_present"]))
        for clip_id in sorted(labels)
        for side in ("left", "right")
    ]

    candidates = []
    grid = itertools.product(
        (0.28, 0.32, 0.36, 0.40, 0.44),
        (0.00, 0.02, 0.05, 0.08),
        (90.0, 100.0, 110.0, 120.0),
        (135.0, 145.0, 155.0, 165.0),
        (125.0, 135.0, 145.0, 155.0),
        (1, 2),
    )
    for stance, speed, elbow, bent_knee, rear_knee, min_run in grid:
        params = {
            "stance": stance, "speed": speed, "elbow": elbow,
            "bent_knee": bent_knee, "rear_knee": rear_knee,
            "min_run": min_run,
        }
        rows = [
            (gt, predict(data_by_clip[clip_id], side, params))
            for clip_id, side, gt in truth
        ]
        metrics = score(rows)
        # Prefer balanced accuracy, then F1, then fewer false positives,
        # then stricter geometry if metrics tie.
        rank = (
            metrics["balanced_accuracy"], metrics["f1"], -metrics["fp"],
            stance, speed, elbow, -bent_knee, rear_knee, min_run,
        )
        candidates.append((rank, params, metrics))
    candidates.sort(key=lambda item: item[0], reverse=True)

    report = {
        "calibration_labels": str(args.labels),
        "athlete_sequences": len(truth),
        "best_parameters": candidates[0][1],
        "best_metrics": candidates[0][2],
        "top_candidates": [
            {"parameters": params, "metrics": metrics}
            for _, params, metrics in candidates[:args.top]
        ],
        "warning": (
            "Parameters are calibrated on AI visual labels. Independent "
            "regression sets must decide whether the change is accepted."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
