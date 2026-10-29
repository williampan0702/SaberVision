from __future__ import annotations

import importlib.util
import hashlib
import io
import json
import sys
import time
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parent / "sabervision_mvp.py"
SPEC = importlib.util.spec_from_file_location("sabervision_mvp", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def signature(path: str, size: int = 100, modified: int = 200) -> dict:
    return {"path": path, "size_bytes": size, "modified_ns": modified}


class FakeArtifact:
    def __init__(
        self, suffix: str, content: str = "", size: int | None = None,
        name: str | None = None,
    ) -> None:
        self.suffix = suffix
        self.name = name or f"artifact{suffix}"
        self.content = content
        self.size = len(content.encode("utf-8")) if size is None else size

    def exists(self) -> bool:
        return True

    def is_file(self) -> bool:
        return True

    def stat(self):
        return type("Stat", (), {"st_size": self.size})()

    def read_text(self, **_kwargs) -> str:
        return self.content

    def open(self, *_args, **_kwargs):
        return io.StringIO(self.content)

    def resolve(self):
        return self

    def __str__(self) -> str:
        return self.name


class FakeAtomicPath:
    def __init__(self, name: str, events: list[tuple]) -> None:
        self.name = name
        self.events = events

    def with_name(self, name: str):
        self.events.append(("with_name", name))
        return FakeAtomicPath(name, self.events)

    def write_text(self, text: str, encoding: str) -> None:
        self.events.append(("write", self.name, text, encoding))

    def replace(self, target) -> None:
        self.events.append(("replace", self.name, target.name))


class ReuseSafetyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.input_signature = signature(r"D:\clips\exchange.mp4")
        self.model_signatures = {
            "fencer": signature(r"D:\models\fencer.pt", 10, 20),
            "pose": signature(r"D:\models\pose.pt", 30, 40),
            "fallback_person_pose": signature(r"D:\models\fallback.pt", 50, 60),
        }
        self.previous = {
            "status": "complete",
            "input_signature": dict(self.input_signature),
            "models": {
                name: value["path"] for name, value in self.model_signatures.items()
            },
            "model_signatures": {
                name: dict(value) for name, value in self.model_signatures.items()
            },
            "implementation_signature": {
                "content_sha256": "pipeline-v1", "file_count": 10
            },
        }

    def test_identical_fingerprints_can_reuse(self) -> None:
        self.assertEqual(
            MODULE.reuse_mismatches(
                self.previous, self.input_signature, self.model_signatures
            ),
            [],
        )

    def test_changed_video_is_rejected(self) -> None:
        current = signature(r"D:\clips\another_exchange.mp4")
        self.assertIn(
            "input_video_path_changed",
            MODULE.reuse_mismatches(self.previous, current, self.model_signatures),
        )

    def test_model_replaced_in_place_is_rejected(self) -> None:
        current_models = {
            name: dict(value) for name, value in self.model_signatures.items()
        }
        current_models["pose"]["size_bytes"] += 1
        self.assertIn(
            "model_pose_size_bytes_changed",
            MODULE.reuse_mismatches(
                self.previous, self.input_signature, current_models
            ),
        )

    def test_content_change_with_same_metadata_is_rejected(self) -> None:
        previous = {
            **self.previous,
            "input_signature": {
                **self.input_signature,
                "content_sha256": "old-content",
            },
        }
        current = {**self.input_signature, "content_sha256": "new-content"}
        self.assertIn(
            "input_video_content_sha256_changed",
            MODULE.reuse_mismatches(previous, current, self.model_signatures),
        )

    def test_incomplete_previous_run_is_rejected(self) -> None:
        self.previous["status"] = "failed"
        self.assertIn(
            "previous_run_not_complete",
            MODULE.reuse_mismatches(
                self.previous, self.input_signature, self.model_signatures
            ),
        )

    def test_legacy_manifest_with_matching_paths_can_reuse(self) -> None:
        legacy = {
            "status": "complete",
            "input_video": r"D:\clips\exchange.mp4",
            "models": {
                name: value["path"] for name, value in self.model_signatures.items()
            },
        }
        self.assertEqual(
            MODULE.reuse_mismatches(
                legacy, self.input_signature, self.model_signatures
            ),
            [],
        )

    def test_changed_pipeline_code_is_rejected(self) -> None:
        current = {"content_sha256": "pipeline-v2", "file_count": 10}
        self.assertIn(
            "pipeline_implementation_changed",
            MODULE.reuse_mismatches(
                self.previous, self.input_signature, self.model_signatures, current
            ),
        )

    def test_legacy_manifest_requires_refresh_when_code_check_is_enabled(self) -> None:
        legacy = {
            "status": "complete",
            "input_video": self.input_signature["path"],
            "models": {
                name: value["path"] for name, value in self.model_signatures.items()
            },
        }
        self.assertIn(
            "previous_manifest_missing_implementation_signature",
            MODULE.reuse_mismatches(
                legacy,
                self.input_signature,
                self.model_signatures,
                {"content_sha256": "pipeline-v1"},
            ),
        )

    def test_valid_json_and_csv_artifacts_are_accepted(self) -> None:
        self.assertIsNone(MODULE.artifact_issue(FakeArtifact(".json", '{"ok": true}')))
        self.assertIsNone(MODULE.artifact_issue(FakeArtifact(".csv", "time,state\n")))

    def test_device_csv_requires_lamp_event_columns(self) -> None:
        issue = MODULE.artifact_issue(FakeArtifact(
            ".csv", "wrong,columns\n", name="device_gated_lamp_events.csv"
        ))
        self.assertEqual(issue, "invalid_csv_schema")

    def test_custom_named_device_csv_still_requires_lamp_columns(self) -> None:
        issue = MODULE.device_evidence_issue(FakeArtifact(
            ".csv", "wrong,columns\n", name="custom_lights.csv"
        ))
        self.assertEqual(issue, "invalid_device_csv_schema")

    def test_custom_named_valid_device_csv_is_accepted(self) -> None:
        issue = MODULE.device_evidence_issue(FakeArtifact(
            ".csv", "state,start_sec,end_sec\nleft,1.0,1.2\n",
            name="custom_lights.csv",
        ))
        self.assertIsNone(issue)

    def test_disabled_optional_evidence_is_never_exposed(self) -> None:
        stale = FakeArtifact(".json", '{"frames": [], "stats": {}}', name="blade.json")
        self.assertFalse(MODULE.optional_artifact_available(stale, enabled=False))
        self.assertTrue(MODULE.optional_artifact_available(stale, enabled=True))

    def test_reused_stage_is_immediately_checkpointed(self) -> None:
        stages: list[dict] = []
        snapshots: list[list[str]] = []
        artifact = FakeArtifact(".json", '{"ok": true}')
        MODULE.run_stage(
            "pose", ["python", "pose.py"], FakeArtifact(""), stages,
            [artifact], reuse=True,
            checkpoint=lambda: snapshots.append([row["status"] for row in stages]),
        )
        self.assertEqual(snapshots, [["reused"]])
        self.assertEqual(stages[0]["status"], "reused")

    def test_truncated_json_is_rejected(self) -> None:
        issue = MODULE.artifact_issue(FakeArtifact(".json", '{"broken":'))
        self.assertEqual(issue, "unreadable_JSONDecodeError")

    def test_wrong_verdict_json_schema_is_rejected(self) -> None:
        issue = MODULE.artifact_issue(
            FakeArtifact(".json", "{}", name="verdict.json")
        )
        self.assertEqual(issue, "invalid_json_schema")

    def test_valid_verdict_json_schema_is_accepted(self) -> None:
        issue = MODULE.artifact_issue(FakeArtifact(
            ".json", '{"events": [], "summary": {}}', name="verdict.json"
        ))
        self.assertIsNone(issue)

    def test_empty_csv_is_rejected(self) -> None:
        self.assertEqual(MODULE.artifact_issue(FakeArtifact(".csv", "")), "empty")

    def test_wrong_pose_csv_schema_is_rejected(self) -> None:
        issue = MODULE.artifact_issue(
            FakeArtifact(".csv", "wrong,columns\n", name="pose.csv")
        )
        self.assertEqual(issue, "invalid_csv_schema")

    def test_implausibly_small_video_is_rejected(self) -> None:
        self.assertEqual(
            MODULE.artifact_issue(FakeArtifact(".mp4", size=256)),
            "video_too_small",
        )

    def test_video_metadata_requires_a_video_stream(self) -> None:
        self.assertEqual(
            MODULE.video_metadata_issue({"streams": [], "format": {"duration": "1"}}),
            "missing_video_stream",
        )

    def test_video_metadata_requires_positive_dimensions(self) -> None:
        payload = {
            "streams": [{"codec_type": "video", "width": 0, "height": 720}],
            "format": {"duration": "1.2"},
        }
        self.assertEqual(
            MODULE.video_metadata_issue(payload), "invalid_video_dimensions"
        )

    def test_valid_video_metadata_is_accepted(self) -> None:
        payload = {
            "streams": [{"codec_type": "video", "width": 1280, "height": 720}],
            "format": {"duration": "1.2"},
        }
        self.assertIsNone(MODULE.video_metadata_issue(payload))

    def test_manifest_json_is_written_then_atomically_replaced(self) -> None:
        events: list[tuple] = []
        MODULE.write_json_atomic(
            FakeAtomicPath("run_manifest.json", events),
            {"status": "running"},
        )
        self.assertEqual(events[0], ("with_name", ".run_manifest.json.tmp"))
        self.assertEqual(events[1][0:2], ("write", ".run_manifest.json.tmp"))
        self.assertEqual(events[1][3], "utf-8")
        self.assertEqual(json.loads(events[1][2])["status"], "running")
        self.assertEqual(
            events[2],
            ("replace", ".run_manifest.json.tmp", "run_manifest.json"),
        )

    def test_finished_manifest_records_timestamp_and_elapsed_time(self) -> None:
        events: list[tuple] = []
        payload = {"status": "failed"}
        MODULE.finish_manifest(
            FakeAtomicPath("run_manifest.json", events), payload, time.perf_counter()
        )
        written = json.loads(events[1][2])
        self.assertIn("finished_at_utc", written)
        self.assertGreaterEqual(written["elapsed_seconds"], 0)
        self.assertEqual(written["status"], "failed")

    def test_file_signature_hashes_small_artifact_fully(self) -> None:
        path = Path(__file__).resolve()
        result = MODULE.file_signature(path)
        self.assertEqual(result["content_sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
        self.assertEqual(result["hash_mode"], "full")


if __name__ == "__main__":
    unittest.main()
