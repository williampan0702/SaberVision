from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
TEST_FILES = tuple(sorted((ROOT / "code").rglob("test_*.py")))


def main() -> int:
    for test_file in TEST_FILES:
        print(f"\n=== {test_file.relative_to(ROOT)} ===", flush=True)
        completed = subprocess.run([sys.executable, str(test_file)], cwd=ROOT)
        if completed.returncode != 0:
            return completed.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
