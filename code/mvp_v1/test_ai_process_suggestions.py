from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parent / "build_ai_process_suggestions.py"
SPEC = importlib.util.spec_from_file_location("build_ai_process_suggestions", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def feature(left_onset: str, right_onset: str, delta_035: float, delta_075: float) -> dict[str, str]:
    return {
        "left_attack_signal_onset_time": left_onset,
        "right_attack_signal_onset_time": right_onset,
        "left_speed_mean_035": str(delta_035), "right_speed_mean_035": "0",
        "left_speed_mean_075": str(delta_075), "right_speed_mean_075": "0",
        "official_outcome": "right",  # Must not influence suggestion.
    }


class SuggestionTests(unittest.TestCase):
    def test_same_input_ignores_official_outcome(self) -> None:
        row = feature("1.0", "1.5", 2.0, 1.8)
        first = MODULE.suggest(row)
        row["official_outcome"] = "left"
        self.assertEqual(first, MODULE.suggest(row))
        self.assertEqual(first["ai_process_suggestion"], "left_attack_candidate")

    def test_close_onsets_are_simultaneous_candidate(self) -> None:
        result = MODULE.suggest(feature("1.0", "1.08", 0.1, 0.2))
        self.assertEqual(result["ai_process_suggestion"], "simultaneous_attack_candidate")

    def test_conflicting_initiation_and_motion_stays_unclear(self) -> None:
        result = MODULE.suggest(feature("1.0", "1.5", -2.0, -1.8))
        self.assertEqual(result["ai_process_suggestion"], "blade_exchange_unclear")
        self.assertEqual(result["capability"], "review_suggestion_only_not_training_truth_or_ruling")


if __name__ == "__main__":
    unittest.main()
