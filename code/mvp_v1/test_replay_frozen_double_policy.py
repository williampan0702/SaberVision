from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parent / "replay_frozen_double_policy.py"
SPEC = importlib.util.spec_from_file_location("replay_frozen_double_policy", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def row(case_id: str, outcome: str, delta_035: float, delta_075: float) -> dict[str, str]:
    return {
        "case_id": case_id,
        "venue": "fixture",
        "label_confidence": "high",
        "official_outcome": outcome,
        "left_speed_mean_035": str(delta_035),
        "right_speed_mean_035": "0",
        "left_speed_mean_075": str(delta_075),
        "right_speed_mean_075": "0",
    }


class FrozenReplayTests(unittest.TestCase):
    def test_strong_consistent_margin_awards_side(self) -> None:
        result = MODULE.evaluate_row(row("strong_left", "left", 2.0, 1.8))
        self.assertEqual(result["prediction"], "left")
        self.assertTrue(result["correct_if_decided"])

    def test_boundary_no_point_case_remains_review(self) -> None:
        result = MODULE.evaluate_row(row("boundary", "no_point", 2.31, 1.56))
        self.assertEqual(result["prediction"], "review")

    def test_nonfinite_feature_remains_review(self) -> None:
        result = MODULE.evaluate_row(row("bad", "left", float("inf"), 2.0))
        self.assertEqual(result["prediction"], "review")
        self.assertEqual(result["input_issue"], "missing_or_nonfinite_speed_feature")

    def test_summary_reports_duplicates_and_false_awards(self) -> None:
        rows = [
            row("same", "left", 2.0, 2.0),
            row("same", "no_point", -2.0, -2.0),
        ]
        _results, summary = MODULE.evaluate_rows(rows)
        self.assertEqual(summary["duplicate_case_ids"], ["same"])
        self.assertEqual(summary["no_point_false_awards"], 1)
        self.assertEqual(summary["wrong_automatic_cases"], ["same"])


if __name__ == "__main__":
    unittest.main()
