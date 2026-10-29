from __future__ import annotations

import hashlib
import importlib.util
import sys
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parent / "verify_run_manifest.py"
SPEC = importlib.util.spec_from_file_location("verify_run_manifest", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class ManifestVerificationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.path = Path(__file__).resolve()
        self.signature = {
            "path": str(self.path),
            "size_bytes": self.path.stat().st_size,
            "content_sha256": hashlib.sha256(self.path.read_bytes()).hexdigest(),
            "hash_mode": "full",
        }

    def test_complete_matching_manifest_passes(self) -> None:
        result = MODULE.verify_payload(
            {"status": "complete", "output_signatures": {"test": self.signature}},
            self.path,
        )
        self.assertTrue(result["ok"])
        self.assertEqual(result["verified_count"], 1)

    def test_hash_mismatch_fails(self) -> None:
        bad = {**self.signature, "content_sha256": "0" * 64}
        result = MODULE.verify_payload(
            {"status": "complete", "output_signatures": {"test": bad}}, self.path
        )
        self.assertFalse(result["ok"])
        self.assertEqual(result["errors"][0]["reason"], "sha256_mismatch")

    def test_unsigned_legacy_manifest_fails_closed(self) -> None:
        result = MODULE.verify_payload({"status": "complete"}, self.path)
        self.assertFalse(result["ok"])
        self.assertEqual(result["errors"][0]["reason"], "missing_output_signatures")

    def test_relocated_run_resolves_relative_output_path(self) -> None:
        manifest_path = self.path.parent / "run_manifest.json"
        signature = {"path": r"C:\old_run\test_verify_run_manifest.py"}
        resolved = MODULE.resolve_signed_path(
            signature, {"output_dir": r"C:\old_run"}, manifest_path
        )
        self.assertEqual(resolved, self.path)

    def test_optional_full_audit_includes_input_model_and_device(self) -> None:
        manifest = {
            "status": "complete",
            "output_signatures": {"test": self.signature},
            "input_signature": self.signature,
            "model_signatures": {"pose": self.signature},
            "device_input_signatures": {
                "automatic": [self.signature], "review": [self.signature]
            },
        }
        result = MODULE.verify_payload(manifest, self.path, include_inputs=True)
        self.assertTrue(result["ok"])
        self.assertEqual(result["scope"], "outputs_and_inputs")
        self.assertEqual(result["verified_count"], 5)

    def test_audit_metadata_binds_report_to_manifest_bytes(self) -> None:
        result = MODULE.add_audit_metadata({"ok": True}, self.path)
        self.assertEqual(result["manifest_path"], str(self.path))
        self.assertEqual(
            result["manifest_sha256"], hashlib.sha256(self.path.read_bytes()).hexdigest()
        )
        self.assertIn("verified_at_utc", result)


if __name__ == "__main__":
    unittest.main()
