from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).resolve().parent / "build_process_label_workspace.py"
SPEC = importlib.util.spec_from_file_location("build_process_label_workspace", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class WorkspaceTests(unittest.TestCase):
    def test_offline_workspace_has_export_and_safety_notice(self) -> None:
        html = MODULE.render_workspace([{
            "case_id": "case_01", "venue": "test", "lamp_time": "1.2",
            "official_outcome": "left", "result_confidence": "high",
            "sequence_sheet": "C:/temp/case_01.jpg", "image": "case_01.jpg",
            "clip": "assets/clips/case_01.mp4",
            "motion_hint": "左方在两个窗口有运动优势候选",
            "process_label": "", "attack_initiator": "", "parry_riposte_side": "",
            "process_confidence": "", "notes": "",
        }])
        self.assertIn("导出 CSV", html)
        self.assertIn("localStorage", html)
        self.assertIn("自动排除在训练集之外", html)
        self.assertIn("case_01.jpg", html)
        self.assertIn("case_01.mp4", html)
        self.assertIn("playbackRate=.5", html)
        self.assertIn("Pose 运动提示（非判罚）", html)
        self.assertIn("outcome_hidden_during_label", html)
        self.assertIn("process_label_source", html)
        self.assertIn("outcome_header_redacted", html)
        self.assertIn("join('\\r\\n')", html)
        self.assertIn("sabervision_phrase_labels_v4_blind", html)
        self.assertIn("sabervision.process.labels.v4", html)
        self.assertIn("备份版本不匹配", html)
        self.assertNotIn("sabervision_phrase_labels_v1", html)
        self.assertNotIn("https://", html)

    def test_motion_hint_does_not_fill_process_label(self) -> None:
        rows = [{"case_id": "case_01", "process_label": ""}]
        # No feature file means an explicit unavailable hint, not an invented label.
        MODULE.add_motion_hints(rows, [])
        self.assertEqual(rows[0]["motion_hint"], "Pose 运动提示不可用")
        self.assertEqual(rows[0]["process_label"], "")

    def test_contact_sheet_result_header_is_redacted(self) -> None:
        image = np.full((1000, 1200, 3), 255, dtype=np.uint8)
        redacted = MODULE.redact_outcome_header_pixels(image)
        self.assertLess(float(redacted[:45].mean()), 2.0)
        self.assertGreater(float(redacted[100:].mean()), 250.0)
        self.assertGreater(float(image[:45].mean()), 250.0)

    def test_legacy_prefill_is_cleared(self) -> None:
        row = {
            "process_label": "left_attack", "process_confidence": "high",
            "attack_initiator": "left", "notes": "legacy",
            "process_label_source": "human_review",
            "outcome_hidden_during_label": "true",
            "outcome_header_redacted": "true",
        }
        MODULE.sanitize_prefill_row(row)
        self.assertEqual(row["process_label"], "")
        self.assertEqual(row["notes"], "")
        self.assertEqual(row["workspace_schema"], MODULE.WORKSPACE_SCHEMA)

    def test_verified_v4_prefill_is_preserved(self) -> None:
        row = {
            "workspace_schema": MODULE.WORKSPACE_SCHEMA,
            "process_label": "right_attack", "process_confidence": "medium",
            "attack_initiator": "right", "notes": "blind human label",
            "process_label_source": "human_review",
            "outcome_hidden_during_label": "true",
            "outcome_header_redacted": "true",
        }
        MODULE.sanitize_prefill_row(row)
        self.assertEqual(row["process_label"], "right_attack")
        self.assertEqual(row["notes"], "blind human label")


if __name__ == "__main__":
    unittest.main()
