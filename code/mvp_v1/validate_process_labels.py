#!/usr/bin/env python3
"""Validate exported SaberVision phrase labels before training or evaluation."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


VALID_LABELS = {
    "simultaneous_attack", "left_attack", "right_attack", "left_attack_no",
    "right_attack_no", "left_parry_riposte", "right_parry_riposte",
    "blade_exchange_unclear", "camera_unclear",
}
VALID_CONFIDENCE = {"high", "medium", "low"}
WORKSPACE_SCHEMA = "sabervision.process.labels.v4"


def is_true(value: object) -> bool:
    return str(value or "").strip().casefold() in {"true", "1", "yes"}


def row_errors(row: dict[str, str]) -> list[str]:
    label = row.get("process_label", "").strip()
    initiator = row.get("attack_initiator", "").strip()
    riposte = row.get("parry_riposte_side", "").strip()
    confidence = row.get("process_confidence", "").strip()
    errors: list[str] = []
    if label not in VALID_LABELS:
        errors.append("missing_or_invalid_process_label")
    if confidence not in VALID_CONFIDENCE:
        errors.append("missing_or_invalid_confidence")
    expected_initiator = {
        "simultaneous_attack": "simultaneous",
        "left_attack": "left", "left_attack_no": "left",
        "right_attack": "right", "right_attack_no": "right",
        "left_parry_riposte": "right", "right_parry_riposte": "left",
    }.get(label)
    if expected_initiator and initiator != expected_initiator:
        errors.append(f"initiator_must_be_{expected_initiator}")
    expected_riposte = {
        "left_parry_riposte": "left",
        "right_parry_riposte": "right",
    }.get(label)
    if expected_riposte and riposte != expected_riposte:
        errors.append(f"riposte_side_must_be_{expected_riposte}")
    if label in {"simultaneous_attack", "left_attack", "right_attack"} and riposte not in {"", "none"}:
        errors.append("riposte_side_must_be_none")
    return errors


def training_exclusion_reasons(row: dict[str, str]) -> list[str]:
    """Explain why an internally valid row must not enter model training."""
    reasons: list[str] = []
    if row.get("process_confidence", "").strip() not in {"high", "medium"}:
        reasons.append("low_confidence")
    if row.get("process_label", "").strip() in {
        "blade_exchange_unclear", "camera_unclear"
    }:
        reasons.append("unclear_process")
    if row.get("process_label_source", "").strip() != "human_review":
        reasons.append("untrusted_label_source")
    if row.get("workspace_schema", "").strip() != WORKSPACE_SCHEMA:
        reasons.append("untrusted_workspace_schema")
    if not is_true(row.get("outcome_hidden_during_label")):
        reasons.append("outcome_visibility_unverified")
    if not is_true(row.get("outcome_header_redacted")):
        reasons.append("outcome_header_not_verified_redacted")
    return reasons


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--accepted", type=Path, required=True)
    parser.add_argument("--rejected", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    with args.input.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    accepted: list[dict[str, str]] = []
    rejected: list[dict[str, str]] = []
    for row in rows:
        errors = row_errors(row)
        exclusions = training_exclusion_reasons(row)
        row = dict(row)
        row["validation_errors"] = ";".join(errors)
        row["training_exclusion_reasons"] = ";".join(exclusions)
        row["eligible_for_training"] = str(not errors and not exclusions).lower()
        (rejected if errors else accepted).append(row)

    fields = list(rows[0]) + [
        "validation_errors", "training_exclusion_reasons", "eligible_for_training"
    ] if rows else []
    for path, selected in ((args.accepted, accepted), (args.rejected, rejected)):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(selected)
    summary = {
        "rows": len(rows),
        "accepted": len(accepted),
        "rejected": len(rejected),
        "eligible_for_training": sum(row["eligible_for_training"] == "true" for row in accepted),
        "label_counts": dict(Counter(row.get("process_label", "") for row in accepted)),
        "rejection_reasons": dict(Counter(
            reason for row in rejected for reason in row["validation_errors"].split(";") if reason
        )),
        "training_exclusion_reasons": dict(Counter(
            reason
            for row in accepted
            for reason in row["training_exclusion_reasons"].split(";")
            if reason
        )),
        "training_gate": (
            "Only internally consistent high/medium-confidence human labels created "
            "while the outcome was hidden on a redacted sheet are eligible."
        ),
    }
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
