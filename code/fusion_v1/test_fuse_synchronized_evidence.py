from __future__ import annotations

import unittest

from fuse_synchronized_evidence import (
    _first_float,
    fuse,
    resolve_device_cluster,
    source_state,
    summarize_blade,
    summarize_blade_timing,
)


def event(source, state, authority="automatic", start=1.0, end=1.2):
    return {
        "source": source,
        "source_file": "fixture.csv",
        "source_row": 1,
        "authority": authority,
        "state": state,
        "start_sec": start,
        "end_sec": end,
        "channels": [],
        "review_reason": None,
    }


class FusionPolicyTests(unittest.TestCase):
    def test_nonfinite_device_times_are_not_parsed(self):
        self.assertIsNone(_first_float({"start_sec": "nan"}, ("start_sec",)))
        self.assertIsNone(_first_float({"start_sec": "inf"}, ("start_sec",)))

    def test_one_automatic_source_is_enough(self):
        result = resolve_device_cluster([event("physical_device", "left")])
        self.assertEqual(result["touch_status"], "resolved_by_any_clear_source")
        self.assertEqual(result["touch_state"], "left")

    def test_same_source_left_and_right_means_double(self):
        result = resolve_device_cluster([
            event("three_regions", "left"), event("three_regions", "right")
        ])
        self.assertEqual(result["touch_state"], "double")

    def test_late_opposite_colour_does_not_change_touch_onset(self):
        self.assertEqual(source_state([
            event("physical_temporal_lamp_events", "right", start=6.13, end=10.0),
            event("physical_temporal_lamp_events", "left", start=9.60, end=10.0),
        ]), "right")

    def test_near_but_non_overlapping_colours_are_not_double(self):
        self.assertEqual(source_state([
            event("physical_temporal_lamp_events", "left", start=1.00, end=1.05),
            event("physical_temporal_lamp_events", "right", start=1.20, end=1.30),
        ]), "left")

    def test_physical_double_can_resolve_single_subset_miss(self):
        result = resolve_device_cluster([
            event("physical_temporal_lamp_events", "double", start=5.60, end=9.66),
            event("device_gated_lamp_events", "right", start=5.77, end=8.49),
        ])
        self.assertEqual(result["touch_status"], "resolved_by_any_clear_source")
        self.assertEqual(result["touch_state"], "double")
        self.assertEqual(result["resolution_rule"], "physical_near_simultaneous_double_over_single_subset")
        self.assertEqual(result["start_sec"], 5.6)

    def test_late_physical_double_does_not_override_early_single(self):
        result = resolve_device_cluster([
            event("physical_temporal_lamp_events", "double", start=9.60, end=10.0),
            event("device_gated_lamp_events", "right", start=6.13, end=10.0),
        ])
        self.assertEqual(result["touch_status"], "automatic_source_conflict")

    def test_explicit_double_accepts_nearby_two_colour_support(self):
        result = resolve_device_cluster([
            event("device_gated_lamp_events", "double", start=3.73, end=6.69),
            event("physical_temporal_lamp_events", "left", start=3.80, end=5.93),
            event("physical_temporal_lamp_events", "right", start=4.27, end=6.80),
        ])
        self.assertEqual(result["touch_status"], "resolved_by_any_clear_source")
        self.assertEqual(result["touch_state"], "double")
        self.assertEqual(result["start_sec"], 3.73)
        self.assertEqual(
            result["resolution_rule"],
            "explicit_double_with_cross_source_colour_support",
        )

    def test_explicit_double_accepts_one_already_active_colour(self):
        result = resolve_device_cluster([
            event("device_gated_lamp_events", "double", start=3.73, end=6.69),
            event("physical_temporal_lamp_events", "left", start=1.47, end=7.47),
            event("physical_temporal_lamp_events", "right", start=4.27, end=6.80),
        ])
        self.assertEqual(result["touch_state"], "double")
        self.assertEqual(
            result["resolution_rule"],
            "explicit_double_with_cross_source_colour_support",
        )

    def test_explicit_double_rejects_late_second_colour(self):
        result = resolve_device_cluster([
            event("device_gated_lamp_events", "double", start=3.73, end=7.00),
            event("physical_temporal_lamp_events", "left", start=3.80, end=7.00),
            event("physical_temporal_lamp_events", "right", start=5.00, end=7.00),
        ])
        self.assertEqual(result["touch_status"], "automatic_source_conflict")

    def test_explicit_double_rejects_single_colour_subset(self):
        result = resolve_device_cluster([
            event("device_gated_lamp_events", "double", start=3.73, end=7.00),
            event("physical_temporal_lamp_events", "left", start=3.80, end=7.00),
        ])
        self.assertEqual(result["touch_status"], "automatic_source_conflict")

    def test_different_automatic_sources_conflict(self):
        result = resolve_device_cluster([
            event("physical_device", "left"),
            event("broadcast_overlay", "right"),
        ])
        self.assertEqual(result["touch_status"], "automatic_source_conflict")
        self.assertIsNone(result["touch_state"])

    def test_review_only_never_becomes_automatic(self):
        result = resolve_device_cluster([
            event("weak_locator", "right", authority="review")
        ])
        self.assertEqual(result["touch_status"], "review_required")
        self.assertIsNone(result["touch_state"])

    def test_missing_action_and_blade_do_not_block_single_light(self):
        payload = fuse(
            None, None, None, [event("physical_device", "right")]
        )
        fused = payload["events"][0]
        self.assertEqual(fused["awarded_side"], "right")
        self.assertEqual(fused["priority"]["status"], "resolved_by_single_light")
        self.assertEqual(
            fused["device_evidence"]["resolution_rule"],
            "automatic_sources_agree",
        )

    def test_blade_timing_is_context_and_does_not_auto_award_double(self):
        timing = {
            "attack_extension_start": {
                "left": {"state": "candidate", "start_sec": 0.7, "end_sec": 0.9},
                "right": {"state": "not_observed", "start_sec": None, "end_sec": None},
            },
            "tip_approach_candidates": {"left": [], "right": []},
            "blade_proximity_candidates": [],
            "possible_contact_windows": [],
            "possible_parry_candidates": [],
        }
        payload = fuse(
            None,
            None,
            None,
            [event("physical_device", "double")],
            blade_timing_payload=timing,
        )
        fused = payload["events"][0]
        self.assertIsNone(fused["awarded_side"])
        self.assertEqual(fused["priority"]["status"], "unresolved_double")
        self.assertEqual(
            fused["blade_timing"]["attack_extension_start"]["left"]["state"],
            "candidate",
        )

    def test_malformed_blade_geometry_is_ignored(self):
        payload = {
            "frames": [
                {"time_sec": "nan", "left": {"guard": [1, 2], "tip": [3, 4]}},
                {"time_sec": 1.0, "left": {"guard": ["bad", 2], "tip": [3, 4]}},
                {"time_sec": 1.1, "left": {"guard": [1, 2], "tip": ["inf", 4]}},
            ]
        }
        result = summarize_blade(payload, 1.1)
        self.assertEqual(result["left"]["observations"], 0)

    def test_malformed_blade_timing_is_ignored(self):
        payload = {
            "attack_extension_start": {
                "left": {"state": "candidate", "start_sec": "inf"}
            },
            "possible_contact_windows": [
                {"time_sec": "nan"}, {"time_sec": "bad"}
            ],
        }
        result = summarize_blade_timing(payload, 1.0)
        self.assertEqual(
            result["attack_extension_start"]["left"]["state"],
            "not_in_event_window",
        )
        self.assertEqual(result["possible_contact_windows"], [])


if __name__ == "__main__":
    unittest.main()
