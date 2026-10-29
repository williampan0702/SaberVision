from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluate_action_rules import load_json
from summarize_action_sequence_v3 import summarize_one as summarize_v3
from temporal_lunge_evidence import classify_lunge_evidence


def summarize_one(
    path: Path,
    confirmed_stance_threshold: float = 0.36,
) -> dict[str, object]:
    # Keep the validated movement timeline, detector reliability and event
    # features from v3.  Only replace the lunge decision surface.
    summary = summarize_v3(path, "balanced")
    data = load_json(path)
    for side in ("left", "right"):
        athlete = summary["athletes"][side]
        evidence = classify_lunge_evidence(
            data, side, confirmed_stance_threshold=confirmed_stance_threshold
        )
        if athlete["action_status"] == "abstain":
            athlete["lunge_present"] = None
            athlete["lunge_candidate_before_abstention"] = (
                evidence["lunge_present"]
            )
            athlete["lunge_source"] = None
            athlete["lunge_evidence_status"] = "abstain"
            athlete["requires_downstream_lunge_resolution"] = False
        else:
            athlete["lunge_present"] = evidence["lunge_present"]
            athlete["lunge_candidate_before_abstention"] = (
                evidence["lunge_present"]
            )
            athlete["lunge_source"] = {
                "confirmed_geometry": "single_frame_or_contiguous_geometry",
                "supported_window_geometry": "short_temporal_window",
                "possible_highspeed_event": "highspeed_event",
                "none": None,
            }[evidence["state"]]
            athlete["lunge_evidence_status"] = evidence["state"]
            athlete["requires_downstream_lunge_resolution"] = evidence[
                "requires_downstream_resolution"
            ]
        best_window = evidence["window"].get("best_window")
        athlete["temporal_window_evidence"] = {
            "supported": evidence["supported_window_geometry"],
            "best_window": best_window,
            "thresholds": evidence["window"]["thresholds"],
        }
        athlete["geometry_candidate_frames"] = evidence["base"][
            "lunge_frames"
        ]
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize fencing actions with tri-state temporal evidence."
    )
    parser.add_argument("--pose-dir", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    paths = []
    for pose_dir in args.pose_dir:
        paths.extend(sorted(pose_dir.glob("*.json")))
    sequences = {path.stem: summarize_one(path) for path in paths}
    reliability_counts = {"high": 0, "medium": 0, "low": 0}
    evidence_counts = {
        "confirmed_geometry": 0,
        "supported_window_geometry": 0,
        "possible_highspeed_event": 0,
        "none": 0,
        "abstain": 0,
    }
    for sequence in sequences.values():
        reliability_counts[sequence["reliability"]["level"]] += 1
        for athlete in sequence["athletes"].values():
            evidence_counts[athlete["lunge_evidence_status"]] += 1

    payload = {
        "schema_version": "4.0",
        "reliability_counts": reliability_counts,
        "lunge_evidence_counts": evidence_counts,
        "decision_policy": {
            "confirmed_geometry": True,
            "supported_window_geometry": None,
            "possible_highspeed_event": None,
            "none": False,
            "abstain": None,
            "null_meaning": (
                "Do not force a binary lunge result; combine with weapon, "
                "scorebox or later action evidence."
            ),
        },
        "sequences": sequences,
        "limitations": [
            "AI-derived action summaries are not referee ground truth.",
            "Window evidence deliberately escalates ambiguous fast actions instead of forcing a label.",
            "A final fencing decision must combine this output with other synchronized subsystems.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "sequences": len(sequences),
        "reliability_counts": reliability_counts,
        "lunge_evidence_counts": evidence_counts,
        "output": str(args.output),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
