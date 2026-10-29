#!/usr/bin/env python3
"""Replay the frozen double-light speed gate on one or more feature sets."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

from mvp_decision import (
    LAMP_SPEED_MARGIN_BODY_HEIGHTS_PER_SEC,
    adjudicate_event,
)


def finite_float(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def evaluate_row(row: dict[str, str]) -> dict[str, object]:
    values = {
        name: finite_float(row.get(name))
        for name in (
            "left_speed_mean_035", "right_speed_mean_035",
            "left_speed_mean_075", "right_speed_mean_075",
        )
    }
    input_issue = None
    if any(value is None for value in values.values()):
        delta_035 = None
        delta_075 = None
        input_issue = "missing_or_nonfinite_speed_feature"
        lamp_speed = {"state": "unavailable", "candidate_side": None}
    else:
        delta_035 = float(values["left_speed_mean_035"]) - float(
            values["right_speed_mean_035"]
        )
        delta_075 = float(values["left_speed_mean_075"]) - float(
            values["right_speed_mean_075"]
        )
        if delta_035 > LAMP_SPEED_MARGIN_BODY_HEIGHTS_PER_SEC:
            side = "left"
        elif delta_035 < -LAMP_SPEED_MARGIN_BODY_HEIGHTS_PER_SEC:
            side = "right"
        else:
            side = None
        lamp_speed = {
            "state": "candidate" if side else "balanced_or_ambiguous",
            "candidate_side": side,
            "left_minus_right": delta_035,
            "consistency_left_minus_right": delta_075,
            "direction_consistent_075": delta_035 * delta_075 > 0,
        }
    verdict = adjudicate_event({
        "touch_status": "resolved_by_any_clear_source",
        "touch_state": "double",
        "action": {"lamp_anchored_speed": lamp_speed},
    })
    prediction = verdict["awarded_side"] or (
        "no_point" if verdict["point_awarded"] is False else "review"
    )
    official = str(row.get("official_outcome") or "").strip()
    return {
        "case_id": row.get("case_id") or "",
        "venue": row.get("venue") or "",
        "label_confidence": row.get("label_confidence") or "",
        "official_outcome": official,
        "prediction": prediction,
        "correct_if_decided": prediction == official if prediction != "review" else "",
        "verdict_status": verdict["status"],
        "speed_delta_035": round(delta_035, 5) if delta_035 is not None else "",
        "speed_delta_075": round(delta_075, 5) if delta_075 is not None else "",
        "input_issue": input_issue or "",
    }


def evaluate_rows(rows: list[dict[str, str]]) -> tuple[list[dict[str, object]], dict]:
    results = [evaluate_row(row) for row in rows]
    ids = [str(row["case_id"]) for row in results if row["case_id"]]
    duplicates = sorted(case_id for case_id, count in Counter(ids).items() if count > 1)
    decided = [row for row in results if row["prediction"] != "review"]
    correct = [row for row in decided if row["prediction"] == row["official_outcome"]]
    no_point = [row for row in results if row["official_outcome"] == "no_point"]
    malformed = [row for row in results if row["input_issue"]]
    summary = {
        "policy": (
            "Both 0.35s and 0.75s lamp-anchored speed differences must be finite, "
            "point to the same side, and reach the frozen 1.70 body-heights/s margin."
        ),
        "cases": len(results),
        "unique_case_ids": len(set(ids)),
        "duplicate_case_ids": duplicates,
        "automatic_decisions": len(decided),
        "correct_automatic_decisions": len(correct),
        "coverage": round(len(decided) / len(results), 5) if results else 0.0,
        "selective_accuracy": round(len(correct) / len(decided), 5) if decided else None,
        "review_required": len(results) - len(decided),
        "known_no_point_cases": len(no_point),
        "no_point_false_awards": sum(
            row["prediction"] in {"left", "right"} for row in no_point
        ),
        "malformed_feature_rows": len(malformed),
        "wrong_automatic_cases": [
            row["case_id"] for row in decided
            if row["prediction"] != row["official_outcome"]
        ],
        "warning": (
            "Broadcast outcomes are engineering proxies, not expert phrase annotations."
        ),
    }
    return results, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, nargs="+", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()

    rows: list[dict[str, str]] = []
    for path in args.features:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows.extend(csv.DictReader(handle))
    results, summary = evaluate_rows(rows)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    fieldnames = list(results[0]) if results else [
        "case_id", "venue", "label_confidence", "official_outcome", "prediction",
        "correct_if_decided", "verdict_status", "speed_delta_035",
        "speed_delta_075", "input_issue",
    ]
    with args.output_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
