#!/usr/bin/env python3
"""Report whether the current Python can run the local SaberVision MVP."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import shutil
import sys
from pathlib import Path


def package_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def file_status(path: Path) -> dict[str, object]:
    exists = path.is_file()
    size = path.stat().st_size if exists else 0
    return {
        "path": str(path.resolve()),
        "exists": exists,
        "size_bytes": size,
        "usable": exists and size > 1024,
    }


def collect(package_root: Path) -> dict[str, object]:
    dependencies = {
        name: package_version(distribution)
        for name, distribution in {
            "numpy": "numpy",
            "opencv": "opencv-python",
            "opencv_headless": "opencv-python-headless",
            "torch": "torch",
            "ultralytics": "ultralytics",
        }.items()
    }
    torch_info: dict[str, object] = {
        "installed": dependencies["torch"] is not None,
        "version": dependencies["torch"],
        "cuda_build": None,
        "cuda_available": False,
        "device_count": 0,
        "devices": [],
        "error": None,
    }
    if torch_info["installed"]:
        try:
            import torch

            torch_info["cuda_build"] = torch.version.cuda
            torch_info["cuda_available"] = bool(torch.cuda.is_available())
            torch_info["device_count"] = int(torch.cuda.device_count())
            torch_info["devices"] = [
                torch.cuda.get_device_name(index)
                for index in range(int(torch_info["device_count"]))
            ]
        except Exception as exc:  # Environment diagnostics must report, not crash.
            torch_info["error"] = f"{type(exc).__name__}: {exc}"

    models = {
        "fencer": file_status(package_root / "models" / "fencer_v4.pt"),
        "pose": file_status(package_root / "models" / "fencing_pose_v1.pt"),
        "fallback_pose": file_status(package_root / "models" / "yolo11n-pose.pt"),
    }
    opencv_present = bool(dependencies["opencv"] or dependencies["opencv_headless"])
    ready = bool(
        dependencies["numpy"]
        and opencv_present
        and dependencies["torch"]
        and dependencies["ultralytics"]
        and shutil.which("ffmpeg")
        and all(item["usable"] for item in models.values())
        and not torch_info["error"]
    )
    return {
        "ready": ready,
        "python": {
            "executable": sys.executable,
            "version": sys.version.split()[0],
        },
        "dependencies": dependencies,
        "torch": torch_info,
        "tools": {
            "ffmpeg": shutil.which("ffmpeg"),
            "ffprobe": shutil.which("ffprobe"),
        },
        "models": models,
        "performance_note": (
            "CUDA is available; model inference can use the NVIDIA GPU."
            if torch_info["cuda_available"]
            else "The MVP can start if ready=true, but inference will use CPU and be much slower."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = collect(args.package_root.resolve())
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    return 0 if result["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
