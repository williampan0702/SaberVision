from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2


def main() -> None:
    parser = argparse.ArgumentParser(description="Read video geometry for the V5.0 runner.")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    cap = cv2.VideoCapture(str(args.input))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {args.input}")
    data = {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": float(cap.get(cv2.CAP_PROP_FPS) or 0.0),
        "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    data["duration_sec"] = data["frames"] / data["fps"] if data["fps"] else 0.0
    cap.release()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(data, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
