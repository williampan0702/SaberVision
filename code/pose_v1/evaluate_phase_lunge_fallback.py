from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from evaluate_action_rules import evaluate_clip, load_json
from search_phase_lunge_fallback import BASE, phase_features, stage_flag
from tune_lunge_rule_v2 import resolve_pose, score


DEFAULT_THRESHOLDS = {
    "inward_net": 0.35,
    "max_speed": 0.65,
    "distance_closure": 0.40,
    "forward_lean": 24.0,
    "stance": 0.40,
    "elbow": 135.0,
}

HIGHSPEED_THRESHOLDS = {
    "inward_net": 0.25,
    "max_speed": 1.00,
    "max_stance": 0.60,
    "max_elbow": 150.0,
}

BASE_EVENT_MIN_SPEED = 0.40


def highspeed_event_flag(features: dict[str, float]) -> bool:
    return bool(
        features["inward_net"] >= HIGHSPEED_THRESHOLDS["inward_net"]
        and features["max_speed"] >= HIGHSPEED_THRESHOLDS["max_speed"]
        and features["max_stance"] >= HIGHSPEED_THRESHOLDS["max_stance"]
        and features["max_elbow"] >= HIGHSPEED_THRESHOLDS["max_elbow"]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--pose-dir", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    label_payload = load_json(args.labels)
    rows = []
    movement_pairs = []
    base_pairs = []
    phase_pairs = []
    highspeed_pairs = []
    motion_gated_pairs = []
    for clip_id, clip_label in label_payload["clips"].items():
        pose_path = resolve_pose(clip_id, args.pose_dir)
        data = load_json(pose_path)
        for side in ("left", "right"):
            gt = clip_label[side]
            base = evaluate_clip(data=data, side=side, **BASE)
            features = phase_features(data, side)
            phase = stage_flag(features, DEFAULT_THRESHOLDS)
            final = bool(base["predicted_lunge"] or phase)
            highspeed = highspeed_event_flag(features)
            motion_gated_base = bool(
                base["predicted_lunge"]
                and features["max_speed"] >= BASE_EVENT_MIN_SPEED
            )
            highspeed_final = bool(motion_gated_base or highspeed)
            gt_lunge = bool(gt["lunge_present"])
            movement_pairs.append((gt["dominant_movement"], base["predicted_movement"]))
            base_pairs.append((gt_lunge, bool(base["predicted_lunge"])))
            phase_pairs.append((gt_lunge, final))
            highspeed_pairs.append((gt_lunge, highspeed_final))
            motion_gated_pairs.append((gt_lunge, motion_gated_base))
            rows.append({
                "clip_id": clip_id,
                "side": side,
                "gt_movement": gt["dominant_movement"],
                "pred_movement": base["predicted_movement"],
                "movement_correct": int(gt["dominant_movement"] == base["predicted_movement"]),
                "gt_lunge": int(gt_lunge),
                "base_lunge": int(bool(base["predicted_lunge"])),
                "motion_gated_base_lunge": int(motion_gated_base),
                "phase_fallback": int(phase),
                "final_lunge": int(final),
                "highspeed_event_fallback": int(highspeed),
                "highspeed_final_lunge": int(highspeed_final),
                "lunge_correct": int(gt_lunge == highspeed_final),
                **{key: round(float(value), 5) for key, value in features.items() if key != "base_lunge"},
                "pose_json": str(pose_path),
            })

    movement_correct = sum(gt == pred for gt, pred in movement_pairs)
    payload = {
        "review_status": label_payload.get("review_status"),
        "athlete_sequences": len(rows),
        "movement": {
            "correct": movement_correct,
            "total": len(movement_pairs),
            "accuracy": movement_correct / len(movement_pairs) if movement_pairs else 0.0,
        },
        "base_lunge": score(base_pairs),
        "phase_fallback_lunge": score(phase_pairs),
        "highspeed_event_lunge": score(highspeed_pairs),
        "motion_gated_base_lunge": score(motion_gated_pairs),
        "phase_thresholds": DEFAULT_THRESHOLDS,
        "highspeed_thresholds": HIGHSPEED_THRESHOLDS,
        "base_event_min_speed": BASE_EVENT_MIN_SPEED,
        "warning": "AI visual labels are engineering references, not expert ground truth.",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "phase_lunge_evaluation.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "phase_lunge_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
