from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parent / "build_mvp_review_packet.py"
SPEC = importlib.util.spec_from_file_location("build_mvp_review_packet", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def frame(time_sec: float, left_speed: float, right_speed: float) -> dict:
    return {
        "time_sec": time_sec,
        "left": {"features": {"inward_speed_body_heights_per_sec": left_speed}},
        "right": {"features": {"inward_speed_body_heights_per_sec": right_speed}},
    }


class MotionSummaryTests(unittest.TestCase):
    def test_only_left_sustained_initiation(self) -> None:
        frames = [frame(index / 10, 0.6, -0.2) for index in range(11)]
        result = MODULE.motion_summary(frames, 1.0)
        self.assertIn("only left has sustained initiation", result["cue"])
        self.assertIn("left has stronger final approach", result["cue"])

    def test_both_start_together(self) -> None:
        frames = [frame(index / 10, 0.5, 0.55) for index in range(11)]
        result = MODULE.motion_summary(frames, 1.0)
        self.assertIn("both initiate together", result["cue"])
        self.assertIn("final approach is balanced", result["cue"])

    def test_motion_cue_is_explicitly_not_a_ruling(self) -> None:
        result = MODULE.motion_summary([], 1.0)
        self.assertEqual(result["capability"], "motion_review_cue_not_fencing_ruling")
        self.assertIn("unclear", result["cue"])

    def test_nonfinite_pose_values_are_ignored(self) -> None:
        frames = [
            frame(0.8, float("inf"), 0.1),
            {"time_sec": "nan", "left": {}, "right": {}},
            {"time_sec": "bad", "left": {}, "right": {}},
        ]
        result = MODULE.motion_summary(frames, 1.0)
        self.assertIn("initiation unclear", result["cue"])
        self.assertEqual(result["final_speed_left_minus_right"], None)

    def test_nonfinite_lamp_time_returns_unavailable_cue(self) -> None:
        result = MODULE.motion_summary([], float("nan"))
        self.assertEqual(
            result["cue"], "initiation unclear; final approach unavailable"
        )

    def test_review_index_contains_safety_notice_and_event(self) -> None:
        content = MODULE.render_review_index([{
            "event_id": 2,
            "start_sec": 3.2,
            "touch_state": "double",
            "verdict_status": "review_required",
            "device_display": "SOURCE CONFLICT",
            "device_summary": "physical=DOUBLE | device-gated=RIGHT",
            "motion_cue": "both initiate together; final approach is balanced",
            "review_clip": "event_2_review.mp4",
            "review_sheet": "event_2_review.jpg",
        }])
        self.assertIn("not official sabre priority decisions", content)
        self.assertIn("Event 2", content)
        self.assertIn("event_2_review.jpg", content)
        self.assertIn("event_2_review.mp4", content)
        self.assertIn("<video", content)
        self.assertIn("SOURCE CONFLICT", content)

    def test_device_conflict_lists_each_source(self) -> None:
        result = MODULE.device_summary({
            "touch_status": "automatic_source_conflict",
            "device_evidence": {"automatic_source_results": {
                "physical_temporal_lamp_events": "double",
                "device_gated_lamp_events": "right",
            }},
        })
        self.assertEqual(result["display"], "SOURCE CONFLICT")
        self.assertEqual(result["detail"], "physical=DOUBLE | device-gated=RIGHT")

    def test_resolved_double_explains_cross_source_support(self) -> None:
        result = MODULE.device_summary({
            "touch_state": "double",
            "touch_status": "resolved_by_any_clear_source",
            "device_evidence": {
                "resolution_rule": "explicit_double_with_cross_source_colour_support",
                "automatic_source_results": {
                    "physical_temporal_lamp_events": "left",
                    "device_gated_lamp_events": "double",
                },
            },
        })
        self.assertEqual(result["display"], "DOUBLE")
        self.assertIn("supported by both physical colours", result["detail"])


if __name__ == "__main__":
    unittest.main()
