#!/usr/bin/env python3
"""Evaluate the conservative MVP double-light policy on the fixed case set."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from mvp_decision import (
    LAMP_SPEED_MARGIN_BODY_HEIGHTS_PER_SEC,
    adjudicate_event,
)


def load_rows(paths: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows.extend(csv.DictReader(handle))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, nargs="+", required=True)
    parser.add_argument("--timing-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()
    output_rows: list[dict[str, object]] = []
    for row in load_rows(args.features):
        left = float(row["left_speed_mean_035"])
        right = float(row["right_speed_mean_035"])
        delta = left - right
        delta_075 = float(row["left_speed_mean_075"]) - float(row["right_speed_mean_075"])
        side = "left" if delta > LAMP_SPEED_MARGIN_BODY_HEIGHTS_PER_SEC else "right" if delta < -LAMP_SPEED_MARGIN_BODY_HEIGHTS_PER_SEC else None
        timing_path = args.timing_dir / f"{row['case_id']}_timing.json"
        timing = json.loads(timing_path.read_text(encoding="utf-8"))
        event = {
            "touch_status": "resolved_by_any_clear_source",
            "touch_state": "double",
            "action": {
                "lamp_anchored_speed": {
                    "state": "candidate" if side else "balanced_or_ambiguous",
                    "candidate_side": side,
                    "left_minus_right": round(delta, 5),
                    "consistency_left_minus_right": round(delta_075, 5),
                    "direction_consistent_075": delta * delta_075 > 0,
                }
            },
            "blade_timing": timing,
        }
        verdict = adjudicate_event(event)
        prediction = verdict["awarded_side"] or ("no_point" if verdict["point_awarded"] is False else "review")
        official = row["official_outcome"]
        output_rows.append({
            "case_id": row["case_id"],
            "venue": row["venue"],
            "label_confidence": row["label_confidence"],
            "official_outcome": official,
            "prediction": prediction,
            "correct_if_decided": prediction == official if prediction != "review" else "",
            "verdict_status": verdict["status"],
            "speed_delta": round(delta, 5),
            "speed_delta_075": round(delta_075, 5),
        })

    decided = [row for row in output_rows if row["prediction"] != "review"]
    correct = [row for row in decided if row["prediction"] == row["official_outcome"]]
    no_point = [row for row in output_rows if row["official_outcome"] == "no_point"]
    summary = {
        "cases": len(output_rows),
        "automatic_decisions": len(decided),
        "correct_automatic_decisions": len(correct),
        "coverage": round(len(decided) / max(1, len(output_rows)), 5),
        "selective_accuracy": round(len(correct) / max(1, len(decided)), 5),
        "review_required": len(output_rows) - len(decided),
        "known_no_point_cases": len(no_point),
        "no_point_false_awards": sum(row["prediction"] in {"left", "right"} for row in no_point),
        "wrong_automatic_cases": [row["case_id"] for row in decided if row["prediction"] != row["official_outcome"]],
        "warning": "Small engineering set; labels are broadcast-outcome proxies, not expert phrase annotations.",
    }
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    args.output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
