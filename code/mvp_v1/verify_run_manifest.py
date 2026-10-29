#!/usr/bin/env python3
"""Verify signed SaberVision MVP outputs, including after a directory move."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any


def content_sha256(path: Path, full_hash_limit: int = 64 * 1024 * 1024) -> str:
    size = path.stat().st_size
    digest = hashlib.sha256()
    if size <= full_hash_limit:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    chunk_size = 4 * 1024 * 1024
    with path.open("rb") as handle:
        digest.update(handle.read(chunk_size))
        handle.seek(max(0, size - chunk_size))
        digest.update(handle.read(chunk_size))
    digest.update(str(size).encode("ascii"))
    return digest.hexdigest()


def resolve_signed_path(
    signature: dict[str, Any], manifest: dict[str, Any], manifest_path: Path
) -> Path:
    original_text = str(signature.get("path") or "")
    original = Path(original_text)
    old_root_text = str(manifest.get("output_dir") or "")
    if old_root_text:
        # Manifests may be verified on a different operating system from the
        # one that produced them.  Parse stored paths using their own syntax
        # instead of the verifier host's pathlib flavour.
        stored_path_type = (
            PureWindowsPath
            if "\\" in original_text or "\\" in old_root_text
            else PurePosixPath
        )
        try:
            relative = stored_path_type(original_text).relative_to(
                stored_path_type(old_root_text)
            )
        except (ValueError, OSError):
            pass
        else:
            relocated = manifest_path.resolve().parent.joinpath(*relative.parts)
            if relocated.is_file():
                return relocated
    if original.is_file():
        return original
    return original


def verify_payload(
    manifest: dict[str, Any], manifest_path: Path, include_inputs: bool = False
) -> dict[str, Any]:
    errors: list[dict[str, str]] = []
    signatures = manifest.get("output_signatures")
    if manifest.get("status") != "complete":
        errors.append({"artifact": "run_manifest", "reason": "run_not_complete"})
    if not isinstance(signatures, dict) or not signatures:
        errors.append({"artifact": "run_manifest", "reason": "missing_output_signatures"})
        signatures = {}
    signatures = dict(signatures)
    if include_inputs:
        input_signature = manifest.get("input_signature")
        if isinstance(input_signature, dict):
            signatures["inputs/video"] = input_signature
        else:
            errors.append({"artifact": "inputs/video", "reason": "missing_signature"})
        model_signatures = manifest.get("model_signatures")
        if isinstance(model_signatures, dict):
            for name, signature in model_signatures.items():
                signatures[f"inputs/models/{name}"] = signature
        else:
            errors.append({"artifact": "inputs/models", "reason": "missing_signatures"})
        device_signatures = manifest.get("device_input_signatures")
        if isinstance(device_signatures, dict):
            for group in ("automatic", "review"):
                items = device_signatures.get(group)
                if not isinstance(items, list):
                    errors.append({
                        "artifact": f"inputs/device/{group}",
                        "reason": "missing_signatures",
                    })
                    continue
                for index, signature in enumerate(items, start=1):
                    signatures[f"inputs/device/{group}/{index}"] = signature
        else:
            errors.append({"artifact": "inputs/device", "reason": "missing_signatures"})
    verified: list[dict[str, Any]] = []
    for name, signature in signatures.items():
        if not isinstance(signature, dict):
            errors.append({"artifact": str(name), "reason": "invalid_signature"})
            continue
        path = resolve_signed_path(signature, manifest, manifest_path)
        if not path.is_file():
            errors.append({"artifact": str(name), "reason": "missing"})
            continue
        actual_size = path.stat().st_size
        if actual_size != signature.get("size_bytes"):
            errors.append({"artifact": str(name), "reason": "size_mismatch"})
            continue
        actual_hash = content_sha256(path)
        if actual_hash != signature.get("content_sha256"):
            errors.append({"artifact": str(name), "reason": "sha256_mismatch"})
            continue
        verified.append({"artifact": str(name), "path": str(path.resolve())})
    return {
        "schema": "sabervision.mvp.verification.v1",
        "scope": "outputs_and_inputs" if include_inputs else "outputs",
        "ok": not errors,
        "verified_count": len(verified),
        "verified": verified,
        "errors": errors,
    }


def add_audit_metadata(result: dict[str, Any], manifest_path: Path) -> dict[str, Any]:
    """Bind a verification report to the exact manifest that was checked."""
    return {
        **result,
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": content_sha256(manifest_path),
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="校验 SaberVision MVP 运行产物")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--include-inputs", action="store_true")
    args = parser.parse_args()
    try:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8-sig"))
        if not isinstance(manifest, dict):
            raise ValueError("manifest root must be an object")
        result = add_audit_metadata(
            verify_payload(manifest, args.manifest, args.include_inputs),
            args.manifest,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        result = {
            "schema": "sabervision.mvp.verification.v1",
            "scope": "outputs_and_inputs" if args.include_inputs else "outputs",
            "ok": False,
            "verified_count": 0,
            "verified": [],
            "errors": [{"artifact": "run_manifest", "reason": str(exc)}],
        }
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
