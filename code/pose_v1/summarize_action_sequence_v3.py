from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluate_action_rules import evaluate_clip, find_exchange_phase, load_json
from evaluate_phase_lunge_fallback import BASE_EVENT_MIN_SPEED, highspeed_event_flag
from search_phase_lunge_fallback import phase_features
from summarize_action_sequence_v2 import BALANCED, movement_timeline


def reliability_from_stats(stats: dict, total: int) -> dict[str, object]:
    if total <= 0:
        return {
            "level": "low", "action_status": "abstain",
            "two_fencer_pair_coverage": 0.0,
            "left_raw_pose_coverage": 0.0,
            "right_raw_pose_coverage": 0.0,
        }
    pair = int(stats.get("frames_with_two_fencer_detections", 0)) / total
    left = int(stats.get("left_pose_frames", 0)) / total
    right = int(stats.get("right_pose_frames", 0)) / total
    minimum_pose = min(left, right)
    if pair >= 0.65 and minimum_pose >= 0.70:
        level = "high"
    elif pair >= 0.30 and minimum_pose >= 0.40:
        level = "medium"
    else:
        level = "low"
    return {
        "level": level,
        "action_status": "ok" if level != "low" else "abstain",
        "two_fencer_pair_coverage": round(pair, 4),
        "left_raw_pose_coverage": round(left, 4),
        "right_raw_pose_coverage": round(right, 4),
    }


def summarize_one(path: Path, lunge_mode: str) -> dict[str, object]:
    data = load_json(path)
    frames = data.get("frames", [])
    total = len(frames)
    stats = data.get("stats") or {}
    reliability = reliability_from_stats(stats, total)
    start_frame, contact_frame = find_exchange_phase(data)
    athletes = {}
    for side in ("left", "right"):
        dynamic = BALANCED if lunge_mode in {"balanced", "recall"} else {}
        base = evaluate_clip(
            data=data, side=side, movement_min_votes=3, lunge_min_run=1,
            method="phase_net", net_movement_threshold=0.10,
            lunge_source="leg_geometry", stance_threshold=0.36,
            speed_threshold=0.02, elbow_threshold=110.0,
            bent_knee_max=135.0, rear_knee_min=145.0,
            **dynamic,
        )
        event = phase_features(data, side)
        motion_gated_base = bool(
            base["predicted_lunge"]
            and event["max_speed"] >= BASE_EVENT_MIN_SPEED
        )
        highspeed_event = highspeed_event_flag(event)
        # A high-speed event is useful evidence that an explosive attack took
        # place, but the reviewed regression set shows that it cannot by
        # itself distinguish a true lunge from a fast advancing step.  Keep it
        # as a separate tri-state evidence channel instead of silently turning
        # it into a binary lunge in the balanced/default mode.
        if motion_gated_base:
            lunge_evidence_status = "confirmed_geometry"
        elif highspeed_event:
            lunge_evidence_status = "possible_highspeed_event"
        else:
            lunge_evidence_status = "none"
        if lunge_mode == "recall":
            lunge = bool(motion_gated_base or highspeed_event)
            source = (
                "balanced_geometry" if motion_gated_base
                else "highspeed_event" if highspeed_event
                else None
            )
        else:
            lunge = bool(base["predicted_lunge"])
            source = "balanced_geometry" if lunge else None
        segments, transitions = movement_timeline(
            data, side, start_frame, contact_frame
        )
        athletes[side] = {
            "action_status": reliability["action_status"],
            "dominant_pre_contact_movement": (
                base["predicted_movement"]
                if reliability["action_status"] == "ok" else None
            ),
            "inward_net_body_heights": (
                round(base["inward_net_body_heights"], 4)
                if base["inward_net_body_heights"] is not None else None
            ),
            "lunge_present": (
                lunge if reliability["action_status"] == "ok" else None
            ),
            "lunge_candidate_before_abstention": lunge,
            "lunge_source": source,
            "lunge_evidence_status": (
                lunge_evidence_status
                if reliability["action_status"] == "ok" else "abstain"
            ),
            "explosive_attack_event": (
                highspeed_event
                if reliability["action_status"] == "ok" else None
            ),
            "requires_downstream_lunge_resolution": bool(
                reliability["action_status"] == "ok"
                and highspeed_event
                and not motion_gated_base
            ),
            "geometry_candidate_frames": base["lunge_frames"],
            "event_features": {
                key: round(float(value), 4)
                for key, value in event.items() if key != "base_lunge"
            },
            "movement_segments": segments,
            "movement_transitions": transitions,
        }
    return {
        "source_pose_json": str(path),
        "total_frames": total,
        "detector": {
            "custom_pair_frames": int(stats.get("frames_with_custom_fencer_pair", 0)),
            "fallback_pair_frames": int(stats.get("frames_with_fallback_person_pair", 0)),
            "fallback_allowed_for_clip": bool(stats.get("fallback_allowed_for_clip", False)),
        },
        "reliability": reliability,
        "phase_start_frame": start_frame,
        "contact_frame": contact_frame,
        "athletes": athletes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pose-dir", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--lunge-mode", choices=("conservative", "balanced", "recall"),
        default="balanced",
    )
    args = parser.parse_args()

    paths = []
    for pose_dir in args.pose_dir:
        paths.extend(sorted(pose_dir.glob("*.json")))
    sequences = {path.stem: summarize_one(path, args.lunge_mode) for path in paths}
    counts = {"high": 0, "medium": 0, "low": 0}
    for item in sequences.values():
        counts[item["reliability"]["level"]] += 1
    payload = {
        "schema_version": "3.1",
        "lunge_mode": args.lunge_mode,
        "reliability_counts": counts,
        "decision_policy": {
            "low_reliability": "abstain",
            "balanced": "validated default candidate mode",
            "possible_highspeed_event": (
                "keep as explosive-attack evidence; do not force a lunge until "
                "another subsystem or later classifier resolves it"
            ),
            "recall": (
                "experimental event-level high-speed fallback; improves aggregate recall "
                "but introduced two regression false positives"
            ),
        },
        "sequences": sequences,
        "limitations": [
            "AI-derived action summaries are not referee ground truth.",
            "Abstention is based on detector/Pose coverage, not a calibrated probability.",
            "Recall mode must not be used as the sole input to a final fencing decision.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "sequences": len(sequences), "lunge_mode": args.lunge_mode,
        "reliability_counts": counts, "output": str(args.output),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
