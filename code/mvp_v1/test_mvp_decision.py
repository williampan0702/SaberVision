from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parent / "mvp_decision.py"
SPEC = importlib.util.spec_from_file_location("mvp_decision", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class DecisionTests(unittest.TestCase):
    def test_single_light_is_final(self) -> None:
        result = MODULE.adjudicate_event({
            "touch_state": "left",
            "touch_status": "resolved_by_any_clear_source",
        })
        self.assertEqual(result["status"], "final_single_light")
        self.assertEqual(result["awarded_side"], "left")

    def test_double_light_does_not_use_action_asymmetry_alone(self) -> None:
        action = {
            "left": {"movement": "advance", "lunge_present": True},
            "right": {"movement": "retreat", "lunge_present": False},
        }
        result = MODULE.adjudicate_event({
            "touch_state": "double",
            "touch_status": "resolved_by_any_clear_source",
            "action": action,
        })
        self.assertEqual(result["status"], "review_required_double_light")
        self.assertIsNone(result["awarded_side"])

    def test_double_light_requires_corroborated_timing(self) -> None:
        result = MODULE.adjudicate_event({
            "touch_state": "double",
            "touch_status": "resolved_by_any_clear_source",
            "action": {
                "left": {},
                "right": {},
                "lamp_anchored_speed": {
                    "state": "candidate",
                    "candidate_side": "left",
                    "reason": "test speed evidence",
                    "left_minus_right": 2.0,
                    "consistency_left_minus_right": 1.8,
                    "direction_consistent_075": True,
                },
            },
            "blade_timing": {
                "attack_extension_start": {
                    "left": {"state": "candidate", "start_sec": 1.0},
                    "right": {"state": "candidate", "start_sec": 1.3},
                }
            },
        })
        self.assertEqual(result["status"], "prototype_priority_award_strong_speed")
        self.assertEqual(result["awarded_side"], "left")

    def test_strong_stable_speed_survives_noisy_extension_onset(self) -> None:
        result = MODULE.adjudicate_event({
            "touch_state": "double",
            "touch_status": "resolved_by_any_clear_source",
            "action": {
                "lamp_anchored_speed": {
                    "state": "candidate",
                    "candidate_side": "right",
                    "left_minus_right": -2.4,
                    "consistency_left_minus_right": -2.0,
                    "direction_consistent_075": True,
                },
            },
            "blade_timing": {
                "attack_extension_start": {
                    "left": {"state": "candidate", "start_sec": 1.0},
                    "right": {"state": "candidate", "start_sec": 1.4},
                }
            },
        })
        self.assertEqual(result["status"], "prototype_priority_award_strong_speed")
        self.assertEqual(result["awarded_side"], "right")

    def test_medium_speed_support_remains_review(self) -> None:
        result = MODULE.adjudicate_event({
            "touch_state": "double",
            "touch_status": "resolved_by_any_clear_source",
            "action": {
                "lamp_anchored_speed": {
                    "state": "candidate",
                    "candidate_side": "right",
                    "left_minus_right": -1.1,
                },
            },
            "blade_timing": {
                "attack_extension_start": {
                    "left": {"state": "candidate", "start_sec": 1.3},
                    "right": {"state": "candidate", "start_sec": 1.0},
                }
            },
        })
        self.assertEqual(result["status"], "review_required_double_light")
        self.assertIsNone(result["awarded_side"])

    def test_cross_venue_no_point_boundary_remains_review(self) -> None:
        result = MODULE.adjudicate_event({
            "touch_state": "double",
            "touch_status": "resolved_by_any_clear_source",
            "action": {
                "lamp_anchored_speed": {
                    "state": "candidate",
                    "candidate_side": "left",
                    "left_minus_right": 2.31458,
                    "consistency_left_minus_right": 1.55772,
                    "direction_consistent_075": True,
                },
            },
        })
        self.assertEqual(result["status"], "review_required_double_light")
        self.assertIsNone(result["awarded_side"])

    def test_malformed_pose_samples_are_ignored(self) -> None:
        frames = []
        for index in range(5):
            frames.append({
                "time_sec": 9.70 + index * 0.05,
                "left": {"features": {"inward_speed_body_heights_per_sec": 2.5}},
                "right": {"features": {"inward_speed_body_heights_per_sec": 0.2}},
            })
        frames.extend([
            {"time_sec": "bad", "left": {"features": {"inward_speed_body_heights_per_sec": 99}}},
            {
                "time_sec": 9.95,
                "left": {"features": {"inward_speed_body_heights_per_sec": "bad"}},
                "right": {"features": {"inward_speed_body_heights_per_sec": "nan"}},
            },
        ])
        result = MODULE.lamp_anchored_speed_evidence({"frames": frames}, 10.0)
        self.assertEqual(result["state"], "candidate")
        self.assertEqual(result["candidate_side"], "left")
        self.assertEqual(result["sample_counts"], {"left": 5, "right": 5})

    def test_infinite_speed_cannot_trigger_automatic_award(self) -> None:
        result = MODULE.adjudicate_event({
            "touch_state": "double",
            "touch_status": "resolved_by_any_clear_source",
            "action": {
                "lamp_anchored_speed": {
                    "state": "candidate", "candidate_side": "left",
                    "left_minus_right": "inf",
                    "consistency_left_minus_right": "inf",
                    "direction_consistent_075": True,
                }
            },
        })
        self.assertEqual(result["status"], "review_required_double_light")
        self.assertIsNone(result["awarded_side"])

    def test_nonfinite_event_time_makes_speed_unavailable(self) -> None:
        result = MODULE.lamp_anchored_speed_evidence({"frames": []}, "nan")
        self.assertEqual(result["state"], "unavailable")

    def test_new_automatic_margin_is_inclusive(self) -> None:
        result = MODULE.adjudicate_event({
            "touch_state": "double",
            "touch_status": "resolved_by_any_clear_source",
            "action": {
                "lamp_anchored_speed": {
                    "state": "candidate",
                    "candidate_side": "right",
                    "left_minus_right": -1.70,
                    "consistency_left_minus_right": -1.70,
                    "direction_consistent_075": True,
                },
            },
        })
        self.assertEqual(result["status"], "prototype_priority_award_strong_speed")
        self.assertEqual(result["awarded_side"], "right")

    def test_confirmed_simultaneous_double_is_no_point(self) -> None:
        result = MODULE.adjudicate_event({
            "touch_state": "double",
            "touch_status": "resolved_by_any_clear_source",
            "simultaneous_evidence": {"state": "confirmed"},
        })
        self.assertEqual(result["status"], "final_simultaneous_no_point")
        self.assertFalse(result["point_awarded"])
        self.assertIsNone(result["awarded_side"])

    def test_conflicting_double_light_evidence_requires_review(self) -> None:
        action = {
            "left": {"movement": "advance", "lunge_present": False},
            "right": {"movement": "retreat", "lunge_present": True},
        }
        result = MODULE.adjudicate_event({
            "touch_state": "double",
            "touch_status": "resolved_by_any_clear_source",
            "action": action,
        })
        self.assertEqual(result["status"], "review_required_conflicting_priority")
        self.assertIsNone(result["awarded_side"])

    def test_unresolved_device_is_never_forced(self) -> None:
        result = MODULE.adjudicate_event({
            "touch_state": "double",
            "touch_status": "unresolved_conflict",
        })
        self.assertEqual(result["status"], "review_required")
        self.assertIsNone(result["awarded_side"])

    def test_independent_automatic_source_conflict_is_review(self) -> None:
        result = MODULE.adjudicate_event({
            "touch_state": None,
            "touch_status": "automatic_source_conflict",
            "device_evidence": {
                "automatic_source_results": {
                    "device_gated_lamp_events": "right",
                    "physical_temporal_lamp_events": "double",
                }
            },
        })
        self.assertEqual(result["status"], "review_required")
        self.assertIsNone(result["awarded_side"])


if __name__ == "__main__":
    unittest.main()
