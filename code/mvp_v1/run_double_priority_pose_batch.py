#!/usr/bin/env python3
"""Run the fixed double-light evaluation set through the current Pose pipeline."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--fencer-model", type=Path, required=True)
    parser.add_argument("--pose-model", type=Path, required=True)
    parser.add_argument("--fallback-model", type=Path, required=True)
    parser.add_argument("--reuse-root", type=Path)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def existing_outputs(root: Path, case_id: str) -> tuple[Path, Path, Path]:
    case_name = case_id.replace("mens_", "mens_").replace("womens_", "womens_")
    return (
        root / "pose_video" / f"{case_name}_pose.mp4",
        root / "pose_json" / f"{case_name}_pose.json",
        root / "pose_csv" / f"{case_name}_pose.csv",
    )


def try_reuse(reuse_root: Path | None, case_id: str, targets: tuple[Path, Path, Path]) -> bool:
    if reuse_root is None:
        return False
    source_dir = reuse_root / f"{case_id}_double" / "pose"
    sources = (source_dir / "pose_overlay.mp4", source_dir / "pose.json", source_dir / "pose.csv")
    if not all(path.exists() for path in sources):
        return False
    for source, target in zip(sources, targets):
        shutil.copy2(source, target)
    return True


def main() -> int:
    args = parse_args()
    output_root = args.output_root.resolve()
    for dirname in ("pose_video", "pose_json", "pose_csv", "logs"):
        (output_root / dirname).mkdir(parents=True, exist_ok=True)

    with args.spec.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    pipeline = REPO_ROOT / "work" / "pose_v1" / "fencing_pose_pipeline.py"
    results: list[dict[str, object]] = []
    total = len(rows)
    for index, row in enumerate(rows, start=1):
        case_id = row["case_id"]
        video = Path(row["video"])
        out_video, out_json, out_csv = existing_outputs(output_root, case_id)
        targets = (out_video, out_json, out_csv)
        print(f"[{index:02d}/{total:02d}] {case_id}", flush=True)

        if not args.force and all(path.exists() and path.stat().st_size > 0 for path in targets):
            print("  reuse: batch output already exists", flush=True)
            results.append({"case_id": case_id, "status": "existing"})
            continue
        if not args.force and try_reuse(args.reuse_root, case_id, targets):
            print("  reuse: copied matching MVP output", flush=True)
            results.append({"case_id": case_id, "status": "copied"})
            continue
        if not video.exists():
            print(f"  error: missing video {video}", flush=True)
            results.append({"case_id": case_id, "status": "missing_video", "video": str(video)})
            continue

        command = [
            str(args.python),
            str(pipeline),
            "--video", str(video),
            "--output-video", str(out_video),
            "--output-json", str(out_json),
            "--output-csv", str(out_csv),
            "--fencer-model", str(args.fencer_model),
            "--pose-model", str(args.pose_model),
            "--fencer-imgsz", "960",
            "--fencer-pair-profile", "multivenue",
            "--pose-imgsz", "512",
            "--fallback-person-model", str(args.fallback_model),
            "--lunge-mode", "balanced",
        ]
        completed = subprocess.run(command, text=True, capture_output=True, encoding="utf-8", errors="replace")
        log_path = output_root / "logs" / f"{case_id}.log"
        log_path.write_text(completed.stdout + "\n--- STDERR ---\n" + completed.stderr, encoding="utf-8")
        status = "ok" if completed.returncode == 0 and all(path.exists() for path in targets) else "failed"
        print(f"  {status}: return code {completed.returncode}", flush=True)
        results.append({"case_id": case_id, "status": status, "returncode": completed.returncode})

    manifest = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "spec": str(args.spec.resolve()),
        "count": total,
        "results": results,
    }
    manifest_path = output_root / "pose_batch_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    failures = [item for item in results if item["status"] in {"failed", "missing_video"}]
    print(f"complete: {total - len(failures)}/{total}; failures={len(failures)}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
