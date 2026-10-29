from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).resolve().parent / "render_mvp_demo.py"
SPEC = importlib.util.spec_from_file_location("render_mvp_demo", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class RenderSafetyTests(unittest.TestCase):
    def test_invalid_blade_coordinates_do_not_draw_or_crash(self) -> None:
        frame = np.zeros((80, 120, 3), dtype=np.uint8)
        before = frame.copy()
        MODULE.line(
            frame,
            {"guard": ["nan", 2], "tip": [30, float("inf")]},
            (255, 255, 255),
        )
        self.assertTrue(np.array_equal(frame, before))

    def test_event_lookup_ignores_invalid_timestamps(self) -> None:
        events = [
            {"event_id": "bad", "start_sec": "nan", "end_sec": 5},
            {"event_id": 2, "start_sec": 1.0, "end_sec": 1.2},
        ]
        self.assertEqual(MODULE.event_at(events, 1.1)["event_id"], 2)

    def test_event_lookup_uses_latest_previous_event_even_if_unsorted(self) -> None:
        events = [
            {"event_id": 2, "start_sec": 4.0, "end_sec": 4.2},
            {"event_id": 1, "start_sec": 1.0, "end_sec": 1.2},
        ]
        self.assertEqual(MODULE.event_at(events, 5.0)["event_id"], 2)


if __name__ == "__main__":
    unittest.main()
