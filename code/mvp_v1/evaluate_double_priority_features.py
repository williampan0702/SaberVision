#!/usr/bin/env python3
"""Evaluate selective double-light priority rules without fitting the holdout set."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


FEATURES = [
    "speed_mean_035",
    "speed_mean_075",
    "speed_mean_150",
    "speed_positive_fraction_075",
    "elbow_median_035",
    "elbow_median_075",
    "wrist_lead_median_035",
    "wrist_lead_median_075",
    "attack_signal_mean_035",
    "attack_signal_mean_075",
    "speed_lead_time",
    "attack_signal_lead_time",
]


def as_float(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def split_rows(rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    counters: defaultdict[str, int] = defaultdict(int)
    development: list[dict[str, str]] = []
    holdout: list[dict[str, str]] = []
    for row in rows:
        outcome = row["official_outcome"]
        if outcome not in {"left", "right"} or row["label_confidence"] != "high":
            continue
        counters[outcome] += 1
        (holdout if counters[outcome] % 3 == 0 else development).append(row)
    return development, holdout


def feature_diff(row: dict[str, str], feature: str) -> float | None:
    left = as_float(row.get(f"left_{feature}", ""))
    right = as_float(row.get(f"right_{feature}", ""))
    if left is None or right is None:
        return None
    return left - right


def predict(row: dict[str, str], feature: str, direction: int, margin: float) -> str:
    diff = feature_diff(row, feature)
    if diff is None:
        return "review"
    score = direction * diff
    if score > margin:
        return "left"
    if score < -margin:
        return "right"
    return "review"


def metrics(rows: list[dict[str, str]], feature: str, direction: int, margin: float) -> dict[str, float | int]:
    predictions = [(row, predict(row, feature, direction, margin)) for row in rows]
    decided = [(row, pred) for row, pred in predictions if pred != "review"]
    correct = sum(pred == row["official_outcome"] for row, pred in decided)
    return {
        "total": len(rows),
        "decided": len(decided),
        "correct": correct,
        "coverage": len(decided) / len(rows) if rows else 0.0,
        "selective_accuracy": correct / len(decided) if decided else 0.0,
    }


def choose_rule(development: list[dict[str, str]], feature: str) -> dict[str, object]:
    diffs = sorted(abs(value) for row in development if (value := feature_diff(row, feature)) is not None)
    margins = sorted({0.0, *diffs})
    candidates: list[dict[str, object]] = []
    for direction in (1, -1):
        for margin in margins:
            result = metrics(development, feature, direction, margin)
            candidates.append({"feature": feature, "direction": direction, "margin": margin, **result})
    # Prefer >=80% selective accuracy, then broad coverage. If no rule reaches
    # that bar, report the most accurate diagnostic rule without promoting it.
    return max(
        candidates,
        key=lambda item: (
            item["selective_accuracy"] >= 0.80 and item["decided"] >= 4,
            item["coverage"] if item["selective_accuracy"] >= 0.80 else 0.0,
            item["selective_accuracy"],
            item["decided"],
            -float(item["margin"]),
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()

    with args.features.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    development, holdout = split_rows(rows)
    results: list[dict[str, object]] = []
    for feature in FEATURES:
        rule = choose_rule(development, feature)
        holdout_metrics = metrics(holdout, feature, int(rule["direction"]), float(rule["margin"]))
        results.append({
            "feature": feature,
            "direction": rule["direction"],
            "margin": round(float(rule["margin"]), 6),
            "development_decided": rule["decided"],
            "development_correct": rule["correct"],
            "development_coverage": round(float(rule["coverage"]), 4),
            "development_selective_accuracy": round(float(rule["selective_accuracy"]), 4),
            "holdout_decided": holdout_metrics["decided"],
            "holdout_correct": holdout_metrics["correct"],
            "holdout_coverage": round(float(holdout_metrics["coverage"]), 4),
            "holdout_selective_accuracy": round(float(holdout_metrics["selective_accuracy"]), 4),
        })
    results.sort(key=lambda item: (item["holdout_selective_accuracy"], item["holdout_decided"], item["development_coverage"]), reverse=True)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    payload = {
        "method": "feature direction and abstention margin selected on development only",
        "development_cases": [row["case_id"] for row in development],
        "holdout_cases": [row["case_id"] for row in holdout],
        "excluded_from_fit": [row["case_id"] for row in rows if row not in development and row not in holdout],
        "results": results,
        "warning": "Small engineering dataset; results are diagnostic and not expert-referee validation.",
    }
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
