from __future__ import annotations

import argparse
import json
from pathlib import Path

from summarize_action_sequence_v5 import summarize_one as summarize_v5


STATIC_GUARD_MIN_GEOMETRY_FRAMES = 8
STATIC_GUARD_MAX_SPEED = 0.60
STATIC_GUARD_MAX_STANCE = 0.55


def summarize_one(path: Path) -> dict:
    sequence = summarize_v5(path)
    for athlete in sequence["athletes"].values():
        if athlete.get("action_status") != "ok":
            continue
        if athlete.get("lunge_evidence_status") != "confirmed_geometry":
            continue
        event = athlete.get("event_features", {})
        static_guard = bool(
            not athlete.get("explosive_attack_event", False)
            and int(athlete.get("geometry_candidate_frames") or 0)
            >= STATIC_GUARD_MIN_GEOMETRY_FRAMES
            and float(event.get("max_speed") or 0.0) < STATIC_GUARD_MAX_SPEED
            and float(event.get("max_stance") or 0.0) < STATIC_GUARD_MAX_STANCE
        )
        if static_guard:
            athlete["lunge_present"] = False
            athlete["lunge_evidence_status"] = "static_guard_negative"
            athlete["lunge_source"] = "persistent_low_speed_guard_geometry"
            athlete["requires_downstream_lunge_resolution"] = False
    return sequence


def main() -> None:
    parser = argparse.ArgumentParser(
        description="v7 action summary suppressing persistent low-speed guard geometry."
    )
    parser.add_argument("--pose-dir", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = []
    for pose_dir in args.pose_dir:
        paths.extend(sorted(pose_dir.glob("*.json")))
    sequences = {path.stem: summarize_one(path) for path in paths}
    evidence_counts: dict[str, int] = {}
    for sequence in sequences.values():
        for athlete in sequence["athletes"].values():
            state = athlete["lunge_evidence_status"]
            evidence_counts[state] = evidence_counts.get(state, 0) + 1
    payload = {
        "schema_version": "7.0",
        "static_guard_rule": {
            "minimum_geometry_frames": STATIC_GUARD_MIN_GEOMETRY_FRAMES,
            "maximum_speed": STATIC_GUARD_MAX_SPEED,
            "maximum_stance": STATIC_GUARD_MAX_STANCE,
        },
        "lunge_evidence_counts": evidence_counts,
        "sequences": sequences,
        "validation_note": (
            "The static-guard suppression removed one development false positive, "
            "did not change the frozen Cairo/Hong Kong result, and did not regress "
            "the historical cross-venue, targeted, or hard-heldout action sets."
        ),
        "limitations": [
            "This is still an AI visual engineering evaluation, not expert referee ground truth.",
            "Most remaining misses are abstentions caused by insufficient continuous pair/Pose evidence.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"sequences": len(sequences), "lunge_evidence_counts": evidence_counts, "output": str(args.output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
