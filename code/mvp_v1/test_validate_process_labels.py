from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parent / "validate_process_labels.py"
SPEC = importlib.util.spec_from_file_location("validate_process_labels", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class ValidationTests(unittest.TestCase):
    def test_valid_left_parry_riposte(self) -> None:
        row = {"process_label": "left_parry_riposte", "attack_initiator": "right", "parry_riposte_side": "left", "process_confidence": "high"}
        self.assertEqual(MODULE.row_errors(row), [])

    def test_parry_riposte_rejects_same_side_initiator(self) -> None:
        row = {"process_label": "left_parry_riposte", "attack_initiator": "left", "parry_riposte_side": "left", "process_confidence": "medium"}
        self.assertIn("initiator_must_be_right", MODULE.row_errors(row))

    def test_attack_rejects_riposte_side(self) -> None:
        row = {"process_label": "right_attack", "attack_initiator": "right", "parry_riposte_side": "left", "process_confidence": "high"}
        self.assertIn("riposte_side_must_be_none", MODULE.row_errors(row))

    def test_missing_confidence_is_rejected(self) -> None:
        row = {"process_label": "simultaneous_attack", "attack_initiator": "simultaneous", "parry_riposte_side": "none", "process_confidence": ""}
        self.assertIn("missing_or_invalid_confidence", MODULE.row_errors(row))

    def test_clean_human_blind_label_is_training_eligible(self) -> None:
        row = {
            "process_label": "left_attack", "attack_initiator": "left",
            "parry_riposte_side": "none", "process_confidence": "medium",
            "process_label_source": "human_review",
            "workspace_schema": MODULE.WORKSPACE_SCHEMA,
            "outcome_hidden_during_label": "true",
            "outcome_header_redacted": "true",
        }
        self.assertEqual(MODULE.row_errors(row), [])
        self.assertEqual(MODULE.training_exclusion_reasons(row), [])

    def test_revealed_outcome_is_excluded_from_training(self) -> None:
        row = {
            "process_label": "right_attack", "attack_initiator": "right",
            "parry_riposte_side": "none", "process_confidence": "high",
            "process_label_source": "human_review",
            "workspace_schema": MODULE.WORKSPACE_SCHEMA,
            "outcome_hidden_during_label": "false",
            "outcome_header_redacted": "true",
        }
        self.assertIn(
            "outcome_visibility_unverified",
            MODULE.training_exclusion_reasons(row),
        )

    def test_ai_or_legacy_source_is_excluded_from_training(self) -> None:
        row = {
            "process_label": "simultaneous_attack",
            "attack_initiator": "simultaneous", "parry_riposte_side": "none",
            "process_confidence": "high", "outcome_hidden_during_label": "true",
            "workspace_schema": MODULE.WORKSPACE_SCHEMA,
            "outcome_header_redacted": "true",
        }
        self.assertIn(
            "untrusted_label_source",
            MODULE.training_exclusion_reasons(row),
        )

    def test_legacy_workspace_schema_is_excluded_from_training(self) -> None:
        row = {
            "process_label": "left_attack", "attack_initiator": "left",
            "parry_riposte_side": "none", "process_confidence": "high",
            "process_label_source": "human_review",
            "outcome_hidden_during_label": "true",
            "outcome_header_redacted": "true",
        }
        self.assertIn(
            "untrusted_workspace_schema",
            MODULE.training_exclusion_reasons(row),
        )


if __name__ == "__main__":
    unittest.main()
