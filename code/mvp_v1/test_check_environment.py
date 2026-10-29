from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parent / "check_environment.py"
SPEC = importlib.util.spec_from_file_location("check_environment", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class FakePath:
    def __init__(self, exists: bool, size: int) -> None:
        self._exists = exists
        self._size = size

    def is_file(self) -> bool:
        return self._exists

    def stat(self):
        return type("Stat", (), {"st_size": self._size})()

    def resolve(self):
        return self

    def __str__(self) -> str:
        return "fixture.pt"


class EnvironmentCheckTests(unittest.TestCase):
    def test_nonempty_model_is_usable(self) -> None:
        result = MODULE.file_status(FakePath(True, 2048))
        self.assertTrue(result["usable"])

    def test_missing_model_is_not_usable(self) -> None:
        result = MODULE.file_status(FakePath(False, 0))
        self.assertFalse(result["usable"])

    def test_implausibly_small_model_is_not_usable(self) -> None:
        result = MODULE.file_status(FakePath(True, 100))
        self.assertFalse(result["usable"])


if __name__ == "__main__":
    unittest.main()
