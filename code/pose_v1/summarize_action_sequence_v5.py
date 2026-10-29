from __future__ import annotations

import argparse
import json
from pathlib import Path

from summarize_action_sequence_v4 import summarize_one as summarize_v4


def fallback_only_static_background(sequence: dict) -> bool:
    detector = sequence["detector"]
    if detector["custom_pair_frames"] > 0:
        return False
    if detector["fallback_pair_frames"] <= 0:
        return False
    athletes = sequence["athletes"]
    return all(
        abs(float(athletes[side]["event_features"].get("inward_net") or 0.0))
        < 0.10
        and float(athletes[side]["event_features"].get("max_speed") or 0.0)
        < 0.10
        for side in ("left", "right")
    )


def summarize_one(path: Path) -> dict:
    sequence = summarize_v4(path)
    if fallback_only_static_background(sequence):
        sequence["sequence_status"] = "no_valid_athlete_pair"
        sequence["sequence_status_reason"] = (
            "Only static generic-person fallback tracks were present; the "
            "custom fencer detector never confirmed a pair. This pattern is "
            "treated as a likely poster/photo false positive."
        )
        for athlete in sequence["athletes"].values():
            athlete["action_status"] = "abstain"
            athlete["dominant_pre_contact_movement"] = None
            athlete["lunge_present"] = None
            athlete["lunge_evidence_status"] = "invalid_athlete_pair"
            athlete["requires_downstream_lunge_resolution"] = False
        return sequence

    sequence["sequence_status"] = "ok"
    sequence["sequence_status_reason"] = None
    for athlete in sequence["athletes"].values():
        if athlete["action_status"] != "ok":
            continue
        if athlete["lunge_evidence_status"] != "none":
            continue
        best = athlete["temporal_window_evidence"].get("best_window")
        component_count = int(best.get("component_count", 0)) if best else 0
        if component_count >= 3:
            athlete["lunge_present"] = None
            athlete["lunge_evidence_status"] = "incomplete_posture_evidence"
            athlete["lunge_source"] = "partial_temporal_geometry"
            athlete["requires_downstream_lunge_resolution"] = True
    return sequence


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Conservative v5 Pose action summary with safe negatives."
    )
    parser.add_argument("--pose-dir", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = []
    for pose_dir in args.pose_dir:
        paths.extend(sorted(pose_dir.glob("*.json")))
    sequences = {path.stem: summarize_one(path) for path in paths}
    state_counts: dict[str, int] = {}
    sequence_status_counts: dict[str, int] = {}
    for sequence in sequences.values():
        status = sequence["sequence_status"]
        sequence_status_counts[status] = sequence_status_counts.get(status, 0) + 1
        for athlete in sequence["athletes"].values():
            state = athlete["lunge_evidence_status"]
            state_counts[state] = state_counts.get(state, 0) + 1
    payload = {
        "schema_version": "5.0",
        "sequence_status_counts": sequence_status_counts,
        "lunge_evidence_counts": state_counts,
        "decision_policy": {
            "confirmed_geometry": True,
            "none_with_at_most_two_components": False,
            "supported_window_geometry": None,
            "possible_highspeed_event": None,
            "incomplete_posture_evidence": None,
            "abstain": None,
            "invalid_athlete_pair": None,
        },
        "sequences": sequences,
        "limitations": [
            "This revision improves output semantics; it does not increase true-lunge detection recall.",
            "AI visual engineering labels are not expert referee ground truth.",
            "Null means that synchronized weapon, scorebox, or later action evidence must resolve the event."
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "sequences": len(sequences),
        "sequence_status_counts": sequence_status_counts,
        "lunge_evidence_counts": state_counts,
        "output": str(args.output),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
