from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path



MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "referee_v5_4" / "code" / "scan_physical_lamp_temporal_v54.py"
)
SPEC = importlib.util.spec_from_file_location("scan_physical_lamp_temporal_v54", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def event(state: str, start: float, end: float) -> dict:
    return {"state": state, "start_sec": start, "end_sec": end, "peak_score": 0.1}


class LampOnsetTests(unittest.TestCase):
    def test_near_onsets_become_double(self) -> None:
        result = MODULE.fuse_color_events([
            event("left", 5.60, 9.66), event("right", 5.67, 8.49),
        ])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["state"], "double")

    def test_late_opposite_colour_remains_separate(self) -> None:
        result = MODULE.fuse_color_events([
            event("right", 6.13, 10.0), event("left", 9.60, 10.0),
        ])
        self.assertEqual([item["state"] for item in result], ["right", "left"])


if __name__ == "__main__":
    unittest.main()
