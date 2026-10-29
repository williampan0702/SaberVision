from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]


def first_existing(paths: Iterable[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path.resolve()
    return None


def default_path(*relative_candidates: str) -> Path | None:
    roots = [REPO_ROOT, HERE.parents[2] if len(HERE.parents) > 2 else REPO_ROOT]
    candidates: list[Path] = []
    for root in roots:
        for relative in relative_candidates:
            candidates.append(root / relative)
    return first_existing(candidates)


def content_sha256(path: Path, full_hash_limit: int = 64 * 1024 * 1024) -> tuple[str, str]:
    """Hash compact files fully and sample both ends of very large videos."""
    size = path.stat().st_size
    digest = hashlib.sha256()
    if size <= full_hash_limit:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest(), "full"
    chunk_size = 4 * 1024 * 1024
    with path.open("rb") as handle:
        digest.update(handle.read(chunk_size))
        handle.seek(max(0, size - chunk_size))
        digest.update(handle.read(chunk_size))
    digest.update(str(size).encode("ascii"))
    return digest.hexdigest(), "first_last_4mib_plus_size"


def file_signature(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    stat = resolved.stat()
    digest, hash_mode = content_sha256(resolved)
    return {
        "path": str(resolved),
        "size_bytes": stat.st_size,
        "modified_ns": stat.st_mtime_ns,
        "content_sha256": digest,
        "hash_mode": hash_mode,
    }


def evidence_signatures(paths: Iterable[Path]) -> list[dict[str, Any]]:
    """Fingerprint ordered evidence inputs without silently de-duplicating them."""
    return [file_signature(path) for path in paths]


def implementation_signature(roots: dict[str, Path]) -> dict[str, Any]:
    """Create a location-independent digest of executable pipeline sources."""
    digest = hashlib.sha256()
    files: list[tuple[str, Path]] = []
    for label, root in sorted(roots.items()):
        resolved = root.resolve()
        if resolved.is_file():
            files.append((f"{label}/{resolved.name}", resolved))
            continue
        for path in sorted(resolved.rglob("*")):
            if path.is_file() and path.suffix.casefold() in {".py", ".ps1"}:
                files.append((f"{label}/{path.relative_to(resolved).as_posix()}", path))
    for logical_name, path in files:
        digest.update(logical_name.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return {
        "content_sha256": digest.hexdigest(),
        "file_count": len(files),
        "source_groups": sorted(roots),
    }


def write_json_atomic(path: Path, payload: object) -> None:
    """Replace JSON through a same-directory temporary file."""
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def finish_manifest(path: Path, manifest: dict[str, Any], started: float) -> None:
    manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    write_json_atomic(path, manifest)


def video_metadata_issue(payload: object) -> str | None:
    if not isinstance(payload, dict):
        return "invalid_video_metadata"
    streams = payload.get("streams")
    if not isinstance(streams, list):
        return "missing_video_stream"
    video_streams = [
        stream for stream in streams
        if isinstance(stream, dict) and stream.get("codec_type") == "video"
    ]
    if not video_streams:
        return "missing_video_stream"
    stream = video_streams[0]
    try:
        width = int(stream.get("width") or 0)
        height = int(stream.get("height") or 0)
        duration = float((payload.get("format") or {}).get("duration") or 0)
    except (TypeError, ValueError):
        return "invalid_video_metadata"
    if width <= 0 or height <= 0:
        return "invalid_video_dimensions"
    if not (duration > 0 and duration < float("inf")):
        return "invalid_video_duration"
    return None


def probe_video_issue(path: Path) -> str | None:
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        return None
    try:
        completed = subprocess.run(
            [
                ffprobe, "-v", "error", "-show_entries",
                "stream=codec_type,width,height:format=duration",
                "-of", "json", str(path),
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired):
        return "video_probe_failed"
    if completed.returncode != 0:
        return "video_probe_failed"
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return "invalid_video_metadata"
    return video_metadata_issue(payload)


def _normalized_path(value: object) -> str:
    return str(value or "").replace("\\", "/").rstrip("/").casefold()


def reuse_mismatches(
    previous: dict[str, Any],
    input_signature: dict[str, Any],
    model_signatures: dict[str, dict[str, Any]],
    current_implementation_signature: dict[str, Any] | None = None,
) -> list[str]:
    """Return reasons why cached inference outputs are unsafe to reuse."""
    mismatches: list[str] = []
    if previous.get("status") != "complete":
        mismatches.append("previous_run_not_complete")

    previous_input = previous.get("input_signature") or {
        "path": previous.get("input_video")
    }
    if _normalized_path(previous_input.get("path")) != _normalized_path(
        input_signature.get("path")
    ):
        mismatches.append("input_video_path_changed")
    else:
        for field in ("size_bytes", "modified_ns", "content_sha256"):
            if previous_input.get(field) is not None and (
                previous_input.get(field) != input_signature.get(field)
            ):
                mismatches.append(f"input_video_{field}_changed")

    previous_model_paths = previous.get("models") or {}
    previous_model_signatures = previous.get("model_signatures") or {}
    for name, current in model_signatures.items():
        old = previous_model_signatures.get(name) or {
            "path": previous_model_paths.get(name)
        }
        if _normalized_path(old.get("path")) != _normalized_path(current.get("path")):
            mismatches.append(f"model_{name}_path_changed")
            continue
        for field in ("size_bytes", "modified_ns", "content_sha256"):
            if old.get(field) is not None and old.get(field) != current.get(field):
                mismatches.append(f"model_{name}_{field}_changed")
    if current_implementation_signature is not None:
        previous_implementation = previous.get("implementation_signature")
        if not isinstance(previous_implementation, dict):
            mismatches.append("previous_manifest_missing_implementation_signature")
        elif previous_implementation.get("content_sha256") != (
            current_implementation_signature.get("content_sha256")
        ):
            mismatches.append("pipeline_implementation_changed")
    return mismatches


def artifact_issue(path: Path) -> str | None:
    """Return why a pipeline artifact is unusable, or None when it is sane."""
    try:
        if not path.exists():
            return "missing"
        if not path.is_file():
            return "not_a_file"
        size = path.stat().st_size
        if size <= 0:
            return "empty"
        suffix = path.suffix.casefold()
        if suffix == ".json":
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
            required_json_types: dict[str, dict[str, type]] = {
                "pose.json": {"frames": list, "stats": dict},
                "action_summary.json": {"sequences": dict},
                "blade.json": {"frames": list, "stats": dict},
                "blade_timing.json": {"attack_extension_start": dict},
                "verdict.json": {"events": list, "summary": dict},
                "review_packet_summary.json": {"events": list},
            }
            requirements = required_json_types.get(getattr(path, "name", ""))
            if requirements:
                if not isinstance(payload, dict) or any(
                    not isinstance(payload.get(key), expected_type)
                    for key, expected_type in requirements.items()
                ):
                    return "invalid_json_schema"
        elif suffix == ".csv":
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                header = next(csv.reader(handle), [])
            if not header or not any(cell.strip() for cell in header):
                return "missing_csv_header"
            required_csv_columns: dict[str, set[str]] = {
                "pose.csv": {"frame", "time_sec"},
                "blade_timing.csv": {"evidence_type", "start_sec"},
                "verdict.csv": {"event_id", "verdict_status", "awarded_side"},
                "review_manifest.csv": {"event_id", "review_sheet", "review_clip"},
                "final_events.csv": {"state", "start_sec", "end_sec"},
                "review_events.csv": {"state", "start_sec", "end_sec"},
                "device_gated_lamp_events.csv": {"state", "start_sec", "end_sec"},
                "physical_temporal_lamp_events.csv": {"state", "start_sec", "end_sec"},
            }
            required_columns = required_csv_columns.get(getattr(path, "name", ""))
            if required_columns and not required_columns.issubset(
                {cell.strip() for cell in header}
            ):
                return "invalid_csv_schema"
        elif suffix in {".html", ".htm"}:
            head = path.read_text(encoding="utf-8", errors="replace")[:4096].casefold()
            if "<html" not in head and "<!doctype html" not in head:
                return "invalid_html"
        elif suffix in {".mp4", ".mov", ".mkv", ".avi"}:
            if size < 1024:
                return "video_too_small"
            # Fake path objects keep unit tests filesystem-free. Real pipeline
            # files receive a bounded FFprobe validation when available.
            if isinstance(path, Path):
                return probe_video_issue(path)
    except (OSError, UnicodeError, json.JSONDecodeError, csv.Error) as exc:
        return f"unreadable_{type(exc).__name__}"
    return None


def artifact_issues(paths: Iterable[Path]) -> dict[str, str]:
    return {
        str(path): issue
        for path in paths
        if (issue := artifact_issue(path)) is not None
    }


def device_evidence_issue(path: Path) -> str | None:
    """Validate score-box CSV semantics even when the user chose a custom name."""
    issue = artifact_issue(path)
    if issue:
        return issue
    if path.suffix.casefold() != ".csv":
        return "device_evidence_not_csv"
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            header = {cell.strip() for cell in next(csv.reader(handle), [])}
    except (OSError, UnicodeError, csv.Error) as exc:
        return f"unreadable_{type(exc).__name__}"
    if not {"state", "start_sec", "end_sec"}.issubset(header):
        return "invalid_device_csv_schema"
    return None


def optional_artifact_available(path: Path, enabled: bool) -> bool:
    """Only expose optional evidence enabled for this run and valid on disk."""
    return enabled and artifact_issue(path) is None


def run_stage(
    name: str,
    command: list[str],
    log_dir: Path,
    stages: list[dict[str, Any]],
    expected: Iterable[Path] = (),
    reuse: bool = False,
    checkpoint: Callable[[], None] | None = None,
) -> None:
    def save_checkpoint() -> None:
        if checkpoint is not None:
            checkpoint()

    expected_paths = list(expected)
    cached_issues = artifact_issues(expected_paths)
    if reuse and expected_paths and not cached_issues:
        print(f"[MVP] {name}: 复用已有结果", flush=True)
        stages.append(
            {
                "name": name,
                "status": "reused",
                "command": command,
                "outputs": [str(path.resolve()) for path in expected_paths],
            }
        )
        save_checkpoint()
        return

    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{name}.log"
    print(f"[MVP] {name}: 开始", flush=True)
    started = time.perf_counter()
    record: dict[str, Any] = {
        "name": name,
        "status": "running",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "log": str(log_path.resolve()),
        "outputs": [],
    }
    stages.append(record)
    save_checkpoint()
    with log_path.open("w", encoding="utf-8") as log_file:
        completed = subprocess.run(
            command,
            cwd=str(REPO_ROOT),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    elapsed = time.perf_counter() - started
    record.update({
        "status": "ok" if completed.returncode == 0 else "failed",
        "returncode": completed.returncode,
        "elapsed_seconds": round(elapsed, 3),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "outputs": [str(path.resolve()) for path in expected_paths if path.exists()],
    })
    if completed.returncode != 0:
        save_checkpoint()
        tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-30:]
        raise RuntimeError(f"阶段 {name} 失败，详见 {log_path}\n" + "\n".join(tail))
    generated_issues = artifact_issues(expected_paths)
    if generated_issues:
        record["status"] = "failed"
        record["artifact_errors"] = generated_issues
        save_checkpoint()
        details = "、".join(f"{path} ({reason})" for path, reason in generated_issues.items())
        raise RuntimeError(f"阶段 {name} 产物无效：{details}")
    save_checkpoint()
    print(f"[MVP] {name}: 完成（{elapsed:.1f} 秒）", flush=True)


def discover_one(root: Path, filename: str) -> Path | None:
    matches = (
        [path for path in sorted(root.rglob(filename)) if artifact_issue(path) is None]
        if root.exists()
        else []
    )
    if not matches:
        return None
    preferred = [path for path in matches if path.parent.name.lower() == "final"]
    return (preferred or matches)[0].resolve()


def transcode_h264(
    source: Path,
    destination: Path,
    stages: list[dict[str, Any]],
    checkpoint: Callable[[], None] | None = None,
) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        shutil.copy2(source, destination)
        issue = artifact_issue(destination)
        if issue:
            raise RuntimeError(f"转码兜底复制产物无效：{destination} ({issue})")
        stages.append(
            {
                "name": "h264_transcode",
                "status": "copied_mp4v_ffmpeg_missing",
                "outputs": [str(destination.resolve())],
            }
        )
        if checkpoint is not None:
            checkpoint()
        return
    command = [
        ffmpeg,
        "-y",
        "-v",
        "error",
        "-i",
        str(source),
        "-c:v",
        "libx264",
        "-crf",
        "20",
        "-preset",
        "medium",
        "-movflags",
        "+faststart",
        str(destination),
    ]
    run_stage(
        "h264_transcode", command, destination.parent / "logs", stages,
        [destination], checkpoint=checkpoint,
    )


def parser() -> argparse.ArgumentParser:
    default_fencer = default_path(
        "outputs/fencer_v4_multivenue/delivery/fencer_yolov8n_v4_multivenue_safe_best.pt",
        "models/fencer_v4.pt",
    )
    default_pose = default_path(
        "outputs/Pose_v1_真实验证集100/training_runs/delivery/yolo11n_fencing_pose_v1_crops_best.pt",
        "models/fencing_pose_v1.pt",
    )
    default_fallback = default_path("models/yolo11n-pose.pt")
    default_referee = first_existing(
        [
            Path(r"D:\Codex输出\SaberVision_referee_recognition_v5_4_physical_fusion\run_universal_referee_v54.ps1"),
            REPO_ROOT / "referee_v5_4/run_universal_referee_v54.ps1",
            (HERE.parents[2] / "referee_v5_4/run_universal_referee_v54.ps1")
            if len(HERE.parents) > 2
            else REPO_ROOT / "referee_v5_4/run_universal_referee_v54.ps1",
        ]
    )

    p = argparse.ArgumentParser(description="SaberVision 离线判罚 MVP 一键流水线")
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--python", type=Path, default=Path(sys.executable))
    p.add_argument("--fencer-model", type=Path, default=default_fencer)
    p.add_argument("--pose-model", type=Path, default=default_pose)
    p.add_argument("--fallback-person-model", type=Path, default=default_fallback)
    p.add_argument("--referee-runner", type=Path, default=default_referee)
    p.add_argument("--device-final", type=Path, action="append", default=[])
    p.add_argument("--device-review", type=Path, action="append", default=[])
    p.add_argument("--reuse", action="store_true", help="存在完整阶段输出时复用")
    p.add_argument("--skip-blade", action="store_true", help="仅用于调试；最终 MVP 建议保留剑身证据")
    return p


def main() -> int:
    run_started = time.perf_counter()
    args = parser().parse_args()
    video = args.video.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = output_dir / "logs"
    manifest_path = output_dir / "run_manifest.json"
    previous_manifest: dict[str, Any] | None = None
    previous_manifest_error: str | None = None
    if args.reuse and manifest_path.exists():
        try:
            previous_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            previous_manifest_error = str(exc)
    stages: list[dict[str, Any]] = []
    manifest: dict[str, Any] = {
        "schema": "sabervision.mvp.run.v1",
        "input_video": str(video),
        "output_dir": str(output_dir),
        "status": "running",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "models": {},
        "reuse": {"requested": bool(args.reuse), "enabled": False, "mismatches": []},
        "stages": stages,
    }
    # Invalidate any older complete manifest before a stage can overwrite its
    # outputs.  An interrupted process therefore leaves `running`, never a
    # misleading reusable success marker.
    write_json_atomic(manifest_path, manifest)

    def checkpoint() -> None:
        write_json_atomic(manifest_path, manifest)

    pose_script = default_path("work/pose_v1/fencing_pose_pipeline.py", "code/pose_v1/fencing_pose_pipeline.py")
    action_script = default_path(
        "work/pose_v1/summarize_action_sequence_v8.py",
        "code/pose_v1/summarize_action_sequence_v8.py",
    )
    blade_script = default_path("work/blade_v1/weapon_direction.py", "code/blade_v1/weapon_direction.py")
    blade_timing_script = default_path(
        "work/blade_v2/blade_action_timing.py", "code/blade_v2/blade_action_timing.py"
    )
    decision_script = HERE / "mvp_decision.py"
    render_script = HERE / "render_mvp_demo.py"
    review_packet_script = HERE / "build_mvp_review_packet.py"

    required = {
        "输入视频": video,
        "运动员模型": args.fencer_model,
        "Pose 模型": args.pose_model,
        "通用 Pose 兜底模型": args.fallback_person_model,
        "Pose 脚本": pose_script,
        "动作脚本": action_script,
        "判罚脚本": decision_script,
        "渲染脚本": render_script,
        "双灯复核图脚本": review_packet_script,
    }
    if not args.skip_blade:
        required.update({"剑身脚本": blade_script, "剑身时序脚本": blade_timing_script})
    dependency_failures: list[str] = []
    for name, path in required.items():
        if path is None:
            dependency_failures.append(f"{name} (missing)")
            continue
        issue = artifact_issue(Path(path))
        if issue:
            dependency_failures.append(f"{name} ({issue})")
    for kind, paths in (
        ("裁判器自动证据", args.device_final),
        ("裁判器人工复核证据", args.device_review),
    ):
        for index, path in enumerate(paths, start=1):
            issue = device_evidence_issue(path)
            if issue:
                dependency_failures.append(f"{kind} {index} ({issue})")
    if dependency_failures:
        message = "MVP 依赖无效：" + "、".join(dependency_failures)
        manifest["status"] = "failed"
        manifest["error"] = message
        finish_manifest(manifest_path, manifest, run_started)
        print(f"[MVP] 失败：{message}", file=sys.stderr, flush=True)
        return 1

    manifest["models"] = {
        "fencer": str(args.fencer_model.resolve()),
        "pose": str(args.pose_model.resolve()),
        "fallback_person_pose": str(args.fallback_person_model.resolve()),
    }
    try:
        input_signature = file_signature(video)
        model_signatures = {
            "fencer": file_signature(args.fencer_model),
            "pose": file_signature(args.pose_model),
            "fallback_person_pose": file_signature(args.fallback_person_model),
        }
        implementation_roots = {
            "mvp_runner": HERE / "sabervision_mvp.py",
            "mvp_decision": decision_script,
            "mvp_render": render_script,
            "mvp_review": review_packet_script,
            "fusion": HERE.parent / "fusion_v1" / "fuse_synchronized_evidence.py",
            "pose": Path(pose_script).parent,
        }
        if not args.skip_blade:
            implementation_roots.update({
                "blade_v1": Path(blade_script).parent,
                "blade_v2": Path(blade_timing_script).parent,
            })
        if args.referee_runner is not None:
            implementation_roots["scorebox"] = args.referee_runner.resolve().parent
        current_implementation_signature = implementation_signature(
            implementation_roots
        )
    except OSError as exc:
        manifest["status"] = "failed"
        manifest["error"] = f"无法读取输入、模型或流水线实现指纹：{exc}"
        finish_manifest(manifest_path, manifest, run_started)
        print(f"[MVP] 失败：{manifest['error']}", file=sys.stderr, flush=True)
        return 1
    manifest["input_signature"] = input_signature
    manifest["model_signatures"] = model_signatures
    manifest["implementation_signature"] = current_implementation_signature

    pose_dir = output_dir / "pose"
    pose_video = pose_dir / "pose_overlay.mp4"
    pose_json = pose_dir / "pose.json"
    pose_csv = pose_dir / "pose.csv"
    action_dir = output_dir / "action"
    action_summary = action_dir / "action_summary.json"
    blade_dir = output_dir / "blade"
    blade_video = blade_dir / "blade_overlay.mp4"
    blade_json = blade_dir / "blade.json"
    blade_timing_json = blade_dir / "blade_timing.json"
    blade_timing_csv = blade_dir / "blade_timing.csv"
    decision_dir = output_dir / "decision"
    verdict_json = decision_dir / "verdict.json"
    verdict_csv = decision_dir / "verdict.csv"
    review_packet_dir = decision_dir / "review_packets"
    review_packet_summary = decision_dir / "review_packet_summary.json"
    raw_demo = output_dir / "mvp_demo_raw.mp4"
    final_demo = output_dir / "SaberVision_MVP_demo.mp4"

    try:
        reuse_enabled = False
        if args.reuse:
            if previous_manifest_error:
                raise ValueError(
                    "无法安全复用：旧 run_manifest.json 无法读取。"
                    "请改用新的输出目录，或不带 --reuse 重新运行。"
                )
            if previous_manifest is None:
                print("[MVP] 未找到旧运行清单，本次执行完整流水线", flush=True)
            else:
                mismatches = reuse_mismatches(
                    previous_manifest,
                    input_signature,
                    model_signatures,
                    current_implementation_signature,
                )
                manifest["reuse"]["mismatches"] = mismatches
                if mismatches:
                    raise ValueError(
                        "无法安全复用，输入视频、模型或流水线实现与旧结果不一致："
                        + "、".join(mismatches)
                        + "。请改用新的输出目录，或不带 --reuse 重新运行。"
                    )
                reuse_enabled = True
                manifest["reuse"]["enabled"] = True
                print("[MVP] 缓存指纹验证通过", flush=True)
        run_stage(
            "pose",
            [
                str(args.python),
                str(pose_script),
                "--video",
                str(video),
                "--output-video",
                str(pose_video),
                "--output-json",
                str(pose_json),
                "--output-csv",
                str(pose_csv),
                "--fencer-model",
                str(args.fencer_model),
                "--pose-model",
                str(args.pose_model),
                "--fencer-imgsz",
                "960",
                "--fencer-pair-profile",
                "multivenue",
                "--pose-imgsz",
                "512",
                "--fallback-person-model",
                str(args.fallback_person_model),
                "--lunge-mode",
                "balanced",
            ],
            logs_dir,
            stages,
            [pose_video, pose_json, pose_csv],
            reuse_enabled,
            checkpoint,
        )

        action_dir.mkdir(parents=True, exist_ok=True)
        run_stage(
            "action",
            [
                str(args.python),
                str(action_script),
                "--pose-dir",
                str(pose_dir),
                "--output",
                str(action_summary),
            ],
            logs_dir,
            stages,
            [action_summary],
            reuse_enabled,
            checkpoint,
        )

        if not args.skip_blade:
            run_stage(
                "blade",
                [
                    str(args.python),
                    str(blade_script),
                    "--video",
                    str(video),
                    "--output-video",
                    str(blade_video),
                    "--output-json",
                    str(blade_json),
                    "--fencer-model",
                    str(args.fencer_model),
                ],
                logs_dir,
                stages,
                [blade_video, blade_json],
                reuse_enabled,
                checkpoint,
            )
            run_stage(
                "blade_timing",
                [
                    str(args.python),
                    str(blade_timing_script),
                    "--pose-json",
                    str(pose_json),
                    "--blade-json",
                    str(blade_json),
                    "--output-json",
                    str(blade_timing_json),
                    "--output-csv",
                    str(blade_timing_csv),
                ],
                logs_dir,
                stages,
                [blade_timing_json, blade_timing_csv],
                reuse_enabled,
                checkpoint,
            )

        device_final = [path.resolve() for path in args.device_final]
        device_review = [path.resolve() for path in args.device_review]
        if device_final or device_review:
            manifest["device_source_policy"] = "explicit_external_sources"
        if not device_final and not device_review:
            if args.referee_runner is None or not args.referee_runner.exists():
                raise FileNotFoundError("未提供裁判器结果，也未找到 v5.4 裁判器一键脚本")
            device_dir = output_dir / "device"
            final_found = discover_one(device_dir, "final_events.csv")
            review_found = discover_one(device_dir, "review_events.csv")
            gated_found = discover_one(device_dir, "device_gated_lamp_events.csv")
            physical_found = discover_one(device_dir, "physical_temporal_lamp_events.csv")
            if reuse_enabled and (final_found or review_found or gated_found or physical_found):
                stages.append(
                    {
                        "name": "scorebox",
                        "status": "reused",
                        "outputs": [
                            str(path)
                            for path in (gated_found, physical_found, final_found, review_found)
                            if path
                        ],
                    }
                )
                checkpoint()
                print("[MVP] scorebox: 复用已有结果", flush=True)
            else:
                powershell = shutil.which("powershell.exe") or shutil.which("powershell")
                if powershell is None:
                    raise FileNotFoundError("找不到 PowerShell，无法运行裁判器识别模块")
                run_stage(
                    "scorebox",
                    [
                        powershell,
                        "-NoProfile",
                        "-ExecutionPolicy",
                        "Bypass",
                        "-File",
                        str(args.referee_runner),
                        "-InputVideo",
                        str(video),
                        "-OutputDir",
                        str(device_dir),
                        "-Python",
                        str(args.python),
                        "-NoPreview",
                    ],
                    logs_dir,
                    stages,
                    [],
                    False,
                    checkpoint,
                )
                final_found = discover_one(device_dir, "final_events.csv")
                review_found = discover_one(device_dir, "review_events.csv")
                gated_found = discover_one(device_dir, "device_gated_lamp_events.csv")
                physical_found = discover_one(device_dir, "physical_temporal_lamp_events.csv")

            # Keep independent automatic sources independent.  The scorebox
            # package's fused CSV can collapse a disagreement into "double";
            # the MVP policy requires such disagreements to remain reviewable.
            independent_automatic = [
                path for path in (gated_found, physical_found) if path is not None
            ]
            if independent_automatic:
                device_final.extend(independent_automatic)
                manifest["device_source_policy"] = "independent_raw_sources"
            elif final_found:
                device_final.append(final_found)
                manifest["device_source_policy"] = "fused_fallback"
            if review_found:
                device_review.append(review_found)

        manifest["device_input_signatures"] = {
            "automatic": evidence_signatures(device_final),
            "review": evidence_signatures(device_review),
        }

        decision_dir.mkdir(parents=True, exist_ok=True)
        decision_command = [
            str(args.python),
            str(decision_script),
            "--action-summary",
            str(action_summary),
            "--action-sequence-key",
            pose_json.stem,
            "--pose-json",
            str(pose_json),
            "--output-json",
            str(verdict_json),
            "--output-csv",
            str(verdict_csv),
        ]
        if optional_artifact_available(blade_timing_json, not args.skip_blade):
            decision_command.extend(["--blade-timing-json", str(blade_timing_json)])
        if optional_artifact_available(blade_json, not args.skip_blade):
            decision_command.extend(["--blade-json", str(blade_json)])
        if device_final:
            decision_command.append("--device-final")
            decision_command.extend(str(path) for path in device_final)
        if device_review:
            decision_command.append("--device-review")
            decision_command.extend(str(path) for path in device_review)
        run_stage(
            "decision",
            decision_command,
            logs_dir,
            stages,
            [verdict_json, verdict_csv],
            False,
            checkpoint,
        )

        run_stage(
            "review_packets",
            [
                str(args.python),
                str(review_packet_script),
                "--pose-video",
                str(pose_video),
                "--pose-json",
                str(pose_json),
                "--verdict-json",
                str(verdict_json),
                "--output-dir",
                str(review_packet_dir),
                "--summary",
                str(review_packet_summary),
            ],
            logs_dir,
            stages,
            [
                review_packet_summary,
                review_packet_dir / "review_manifest.csv",
                review_packet_dir / "review_index.html",
            ],
            False,
            checkpoint,
        )

        render_command = [
            str(args.python),
            str(render_script),
            "--pose-video",
            str(pose_video),
            "--verdict-json",
            str(verdict_json),
            "--output",
            str(raw_demo),
        ]
        if optional_artifact_available(blade_json, not args.skip_blade):
            render_command.extend(["--blade-json", str(blade_json)])
        run_stage(
            "render", render_command, logs_dir, stages, [raw_demo], False,
            checkpoint,
        )
        transcode_h264(raw_demo, final_demo, stages, checkpoint)

        manifest["outputs"] = {
            "demo_video": str(final_demo),
            "verdict_json": str(verdict_json),
            "verdict_csv": str(verdict_csv),
            "review_packet_summary": str(review_packet_summary),
            "review_packet_dir": str(review_packet_dir),
            "review_index": str(review_packet_dir / "review_index.html"),
            "pose_json": str(pose_json),
            "action_summary": str(action_summary),
            "blade_timing": (
                str(blade_timing_json)
                if optional_artifact_available(blade_timing_json, not args.skip_blade)
                else None
            ),
            "device_final": [str(path) for path in device_final],
            "device_review": [str(path) for path in device_review],
        }
        signed_outputs = {
            "demo_video": final_demo,
            "verdict_json": verdict_json,
            "verdict_csv": verdict_csv,
            "review_packet_summary": review_packet_summary,
            "pose_video": pose_video,
            "pose_json": pose_json,
            "pose_csv": pose_csv,
            "action_summary": action_summary,
            "raw_demo_video": raw_demo,
        }
        if optional_artifact_available(blade_timing_json, not args.skip_blade):
            signed_outputs["blade_timing"] = blade_timing_json
            signed_outputs["blade_timing_csv"] = blade_timing_csv
            signed_outputs["blade_json"] = blade_json
            signed_outputs["blade_video"] = blade_video
        for path in sorted(review_packet_dir.rglob("*")):
            if path.is_file():
                logical_name = path.relative_to(review_packet_dir).as_posix()
                signed_outputs[f"review_packets/{logical_name}"] = path
        manifest["output_signatures"] = {
            name: file_signature(path) for name, path in signed_outputs.items()
        }
        manifest["status"] = "complete"
        print(f"[MVP] 全部完成：{final_demo}", flush=True)
        print(f"[MVP] 判罚结果：{verdict_json}", flush=True)
        return 0
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = str(exc)
        print(f"[MVP] 失败：{exc}", file=sys.stderr, flush=True)
        return 1
    finally:
        finish_manifest(manifest_path, manifest, run_started)


if __name__ == "__main__":
    raise SystemExit(main())
