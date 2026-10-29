#!/usr/bin/env python3
"""Benchmark a symmetric, selective Pose outcome proxy on double-light cases.

This is deliberately an evaluation tool, not a fencing-rule oracle.  It uses
broadcast score progression labels and reports only selective left/right
coverage.  No-point rows count as errors if the model awards either side.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


DEFAULT_METRICS = [
    "speed_mean_035",
    "speed_max_035",
    "speed_positive_fraction_035",
    "elbow_median_035",
    "elbow_max_035",
    "wrist_lead_median_035",
    "wrist_lead_max_035",
    "attack_signal_mean_035",
    "attack_signal_max_035",
    "speed_mean_075",
    "speed_max_075",
    "speed_positive_fraction_075",
    "elbow_median_075",
    "elbow_max_075",
    "wrist_lead_median_075",
    "wrist_lead_max_075",
    "attack_signal_mean_075",
    "attack_signal_max_075",
    "speed_mean_150",
    "speed_max_150",
    "speed_positive_fraction_150",
    "elbow_median_150",
    "elbow_max_150",
    "wrist_lead_median_150",
    "wrist_lead_max_150",
    "attack_signal_mean_150",
    "attack_signal_max_150",
    "speed_lead_time",
    "attack_signal_lead_time",
]


def finite(value: str | None) -> float:
    try:
        number = float(value) if value not in {None, ""} else math.nan
    except (TypeError, ValueError):
        return math.nan
    return number if math.isfinite(number) else math.nan


def canonical_group(venue: str) -> str:
    value = venue.lower()
    if value.startswith("seoul"):
        return "seoul"
    return value


def load_rows(paths: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for path in paths:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                case_id = row["case_id"]
                if case_id in seen:
                    continue
                seen.add(case_id)
                rows.append(row)
    return rows


def infer_metrics(rows: list[dict[str, str]]) -> list[str]:
    if not rows:
        return []
    keys = set(rows[0])
    inferred = sorted(
        key.removeprefix("left_")
        for key in keys
        if key.startswith("left_") and f"right_{key.removeprefix('left_')}" in keys
    )
    return inferred or DEFAULT_METRICS


def encode(rows: list[dict[str, str]], metrics: list[str]) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    matrix: list[list[float]] = []
    labels: list[float] = []
    case_ids: list[str] = []
    groups: list[str] = []
    mapping = {"left": 1.0, "right": -1.0, "no_point": 0.0}
    for row in rows:
        if row.get("official_outcome") not in mapping:
            continue
        values = []
        for metric in metrics:
            left = finite(row.get(f"left_{metric}"))
            right = finite(row.get(f"right_{metric}"))
            values.append(left - right if math.isfinite(left) and math.isfinite(right) else math.nan)
        matrix.append(values)
        labels.append(mapping[row["official_outcome"]])
        case_ids.append(row["case_id"])
        groups.append(canonical_group(row.get("venue", "unknown")))
    return np.asarray(matrix, dtype=np.float64), np.asarray(labels), case_ids, groups


def prepare(train_x: np.ndarray, test_x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    median = np.nanmedian(train_x, axis=0)
    median = np.where(np.isfinite(median), median, 0.0)
    train = np.where(np.isfinite(train_x), train_x, median)
    test = np.where(np.isfinite(test_x), test_x, median)
    scale = np.std(train, axis=0)
    scale = np.where(scale >= 1e-6, scale, 1.0)
    center = np.mean(train, axis=0)
    return (train - center) / scale, (test - center) / scale, center, scale


def fit_ridge(train_x: np.ndarray, train_y: np.ndarray, alpha: float) -> np.ndarray:
    # Mirror augmentation enforces left/right symmetry and removes venue-side bias.
    augmented_x = np.vstack([train_x, -train_x])
    augmented_y = np.concatenate([train_y, -train_y])
    gram = augmented_x.T @ augmented_x
    return np.linalg.solve(gram + alpha * np.eye(gram.shape[0]), augmented_x.T @ augmented_y)


def predict_fold(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    alpha: float,
) -> np.ndarray:
    normalized_train, normalized_test, _, _ = prepare(train_x, test_x)
    weights = fit_ridge(normalized_train, train_y, alpha)
    return normalized_test @ weights


def selective_stats(scores: np.ndarray, labels: np.ndarray, threshold: float) -> dict[str, int | float]:
    decided = np.abs(scores) >= threshold
    predictions = np.where(scores >= 0, 1.0, -1.0)
    correct = decided & (labels != 0) & (predictions == labels)
    wrong = decided & ~correct
    return {
        "cases": int(labels.size),
        "automatic": int(decided.sum()),
        "correct": int(correct.sum()),
        "wrong": int(wrong.sum()),
        "no_point_false_awards": int((decided & (labels == 0)).sum()),
        "coverage": round(float(decided.mean()), 5) if labels.size else 0.0,
        "selective_accuracy": round(float(correct.sum() / max(1, decided.sum())), 5),
    }


def leave_group_out(x: np.ndarray, y: np.ndarray, groups: list[str], alpha: float) -> np.ndarray:
    scores = np.zeros_like(y)
    group_array = np.asarray(groups)
    for group in sorted(set(groups)):
        test = group_array == group
        train = ~test
        scores[test] = predict_fold(x[train], y[train], x[test], alpha)
    return scores


def choose_configuration(x: np.ndarray, y: np.ndarray, groups: list[str]) -> tuple[float, float, np.ndarray, dict]:
    best: tuple[int, float, float, float, np.ndarray, dict] | None = None
    for alpha in (0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0):
        scores = leave_group_out(x, y, groups, alpha)
        candidates = sorted({round(float(abs(score)), 6) for score in scores if abs(score) > 0}, reverse=True)
        for threshold in candidates:
            stats = selective_stats(scores, y, threshold)
            if stats["wrong"] != 0:
                continue
            key = (int(stats["automatic"]), -threshold, -alpha)
            if best is None or key > best[:3]:
                best = (key[0], key[1], key[2], alpha, scores.copy(), stats)
    if best is None:
        raise RuntimeError("No zero-error selective configuration")
    _, neg_threshold, _, alpha, scores, stats = best
    return alpha, -neg_threshold, scores, stats


def prediction_name(score: float, threshold: float) -> str:
    if abs(score) < threshold:
        return "review"
    return "left" if score > 0 else "right"


def write_cases(
    path: Path,
    case_ids: list[str],
    groups: list[str],
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> None:
    label_names = {1.0: "left", -1.0: "right", 0.0: "no_point"}
    rows = []
    for case_id, group, label, score in zip(case_ids, groups, labels, scores):
        prediction = prediction_name(float(score), threshold)
        rows.append({
            "case_id": case_id,
            "group": group,
            "official_outcome": label_names[float(label)],
            "model_score": round(float(score), 6),
            "threshold": round(threshold, 6),
            "prediction": prediction,
            "correct_if_decided": "" if prediction == "review" else prediction == label_names[float(label)],
        })
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--development-features", type=Path, nargs="+", required=True)
    parser.add_argument("--holdout-features", type=Path, nargs="+")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dev_rows = load_rows(args.development_features)
    metrics = infer_metrics(dev_rows)
    dev_x, dev_y, dev_ids, dev_groups = encode(dev_rows, metrics)
    alpha, threshold, oof_scores, oof_stats = choose_configuration(dev_x, dev_y, dev_groups)
    write_cases(args.output_dir / "development_group_oof_cases.csv", dev_ids, dev_groups, dev_y, oof_scores, threshold)

    payload: dict[str, object] = {
        "schema": "sabervision_pose_outcome_proxy_benchmark_v1",
        "model": "symmetric_ridge_regression_with_group_holdout_and_rejection",
        "feature_count": len(metrics),
        "features": metrics,
        "development_cases": len(dev_ids),
        "development_groups": sorted(set(dev_groups)),
        "selected_alpha": alpha,
        "selected_absolute_score_threshold": round(threshold, 6),
        "development_group_oof": oof_stats,
        "limitations": [
            "Labels are broadcast outcomes, not expert phrase annotations.",
            "The threshold was selected on development group-out predictions.",
            "This benchmark must not enter final adjudication unless a separate holdout is safe.",
        ],
    }

    if args.holdout_features:
        holdout_rows = load_rows(args.holdout_features)
        hold_x, hold_y, hold_ids, hold_groups = encode(holdout_rows, metrics)
        normalized_dev, normalized_hold, _, _ = prepare(dev_x, hold_x)
        weights = fit_ridge(normalized_dev, dev_y, alpha)
        hold_scores = normalized_hold @ weights
        hold_stats = selective_stats(hold_scores, hold_y, threshold)
        write_cases(args.output_dir / "holdout_cases.csv", hold_ids, hold_groups, hold_y, hold_scores, threshold)
        payload["holdout"] = hold_stats
        payload["safe_for_integration"] = hold_stats["wrong"] == 0 and hold_stats["automatic"] > 0
    else:
        payload["safe_for_integration"] = False

    (args.output_dir / "benchmark.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
