#!/usr/bin/env python3
"""Create outcome-blind, conservative fencing-process suggestions from Pose features."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def optional_float(value: object) -> float | None:
    try:
        text = str(value).strip()
        return float(text) if text else None
    except (TypeError, ValueError):
        return None


def suggest(feature: dict[str, str]) -> dict[str, object]:
    left_onset = optional_float(feature.get("left_attack_signal_onset_time"))
    right_onset = optional_float(feature.get("right_attack_signal_onset_time"))
    if left_onset is None and right_onset is None:
        left_onset = optional_float(feature.get("left_speed_onset_time"))
        right_onset = optional_float(feature.get("right_speed_onset_time"))

    if left_onset is not None and right_onset is not None:
        onset_gap = right_onset - left_onset
        if abs(onset_gap) <= 0.12:
            initiator = "simultaneous"
        else:
            initiator = "left" if onset_gap > 0 else "right"
    elif left_onset is not None:
        onset_gap = None
        initiator = "left"
    elif right_onset is not None:
        onset_gap = None
        initiator = "right"
    else:
        onset_gap = None
        initiator = "unclear"

    left_035 = optional_float(feature.get("left_speed_mean_035")) or 0.0
    right_035 = optional_float(feature.get("right_speed_mean_035")) or 0.0
    left_075 = optional_float(feature.get("left_speed_mean_075")) or 0.0
    right_075 = optional_float(feature.get("right_speed_mean_075")) or 0.0
    delta_035 = left_035 - right_035
    delta_075 = left_075 - right_075
    same_direction = delta_035 * delta_075 > 0
    weaker_margin = min(abs(delta_035), abs(delta_075)) if same_direction else 0.0
    motion_side = "left" if same_direction and delta_035 > 0 else "right" if same_direction else "unclear"

    suggestion = "blade_exchange_unclear"
    confidence = "low"
    basis: list[str] = []
    if initiator == "simultaneous":
        suggestion = "simultaneous_attack_candidate"
        confidence = "medium" if weaker_margin < 0.80 else "low"
        basis.append("attack-signal onsets are within 0.12s")
    elif initiator in {"left", "right"} and motion_side == initiator:
        suggestion = f"{initiator}_attack_candidate"
        if weaker_margin >= 1.70 and (onset_gap is None or abs(onset_gap) >= 0.18):
            confidence = "medium"
        basis.append(f"{initiator} begins first and has same-side motion advantage")
    elif initiator in {"left", "right"} and motion_side not in {initiator, "unclear"}:
        basis.append("initiation and final motion point to opposite sides")
    else:
        basis.append("onset or two-window motion is insufficient")
    basis.append(f"speed deltas: 0.35s={delta_035:+.3f}, 0.75s={delta_075:+.3f}")
    return {
        "ai_process_suggestion": suggestion,
        "ai_suggestion_confidence": confidence,
        "ai_attack_initiator": initiator,
        "attack_onset_gap_sec": round(onset_gap, 4) if onset_gap is not None else None,
        "motion_advantage_side": motion_side,
        "speed_delta_035": round(delta_035, 4),
        "speed_delta_075": round(delta_075, 4),
        "ai_suggestion_basis": "; ".join(basis),
        "capability": "review_suggestion_only_not_training_truth_or_ruling",
    }


def load_features(paths: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows.extend(csv.DictReader(handle))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, nargs="+", required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    output_rows = []
    for feature in load_features(args.features):
        output_rows.append({
            "case_id": feature["case_id"],
            "venue": feature.get("venue", ""),
            **suggest(feature),
            # Outcome is appended only after the outcome-blind suggestion exists.
            "official_outcome": feature.get("official_outcome", ""),
        })
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)

    directional = [row for row in output_rows if row["ai_process_suggestion"] in {"left_attack_candidate", "right_attack_candidate"}]
    directional_match = [
        row for row in directional
        if row["official_outcome"] == str(row["ai_process_suggestion"]).split("_", 1)[0]
    ]
    simultaneous = [row for row in output_rows if row["ai_process_suggestion"] == "simultaneous_attack_candidate"]
    summary = {
        "cases": len(output_rows),
        "directional_attack_suggestions": len(directional),
        "directional_outcome_proxy_matches": len(directional_match),
        "directional_outcome_proxy_match_rate": round(len(directional_match) / max(1, len(directional)), 4),
        "simultaneous_candidates": len(simultaneous),
        "simultaneous_candidates_with_no_point_outcome": sum(row["official_outcome"] == "no_point" for row in simultaneous),
        "unclear": sum(row["ai_process_suggestion"] == "blade_exchange_unclear" for row in output_rows),
        "safe_for_training_truth": False,
        "safe_for_automatic_ruling": False,
        "warning": "Outcome agreement is only a proxy audit; suggestions remain review aids and are not phrase labels.",
    }
    args.output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
