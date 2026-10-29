#!/usr/bin/env python3
"""Extract reviewed double-light events with score-progression outcomes."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--venue", default="tallinn")
    parser.add_argument("--pre-sec", type=float, default=1.4)
    parser.add_argument("--post-sec", type=float, default=0.35)
    args = parser.parse_args()
    clip_dir = args.output_root / "clips"
    log_dir = args.output_root / "logs"
    clip_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    with args.events.open("r", encoding="utf-8-sig", newline="") as handle:
        events = [row for row in csv.DictReader(handle) if row.get("state") == "double" and row.get("score_change_side") in {"left", "right", "no_point"}]
    rows: list[dict[str, str]] = []
    failures: list[str] = []
    for index, event in enumerate(events, 1):
        event_id = int(event["event_id"])
        case_id = f"{args.venue}_double_{event_id:02d}"
        lamp_time = float(event["start_sec"])
        start = max(0.0, lamp_time - args.pre_sec)
        duration = lamp_time + args.post_sec - start
        output = clip_dir / f"{case_id}.mp4"
        command = [
            "ffmpeg", "-y", "-ss", f"{start:.4f}", "-i", str(args.video),
            "-t", f"{duration:.4f}", "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "18", str(output),
        ]
        completed = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace")
        (log_dir / f"{case_id}.log").write_text(completed.stdout + "\n--- STDERR ---\n" + completed.stderr, encoding="utf-8")
        if completed.returncode or not output.exists():
            failures.append(case_id)
            continue
        rows.append({
            "case_id": case_id,
            "venue": args.venue,
            "video": str(output.resolve()),
            "official_outcome": event["score_change_side"],
            "label_basis": "visually_confirmed_double_lamp_plus_score_progression",
            "confidence": event.get("label_confidence", "high"),
            "lamp_time_override": f"{lamp_time - start:.4f}",
            "source_event_id": event["event_id"],
            "source_lamp_time": event["start_sec"],
            "score_change_time": event.get("score_change_time", ""),
            "score_delay_sec": event.get("score_delay_sec", ""),
            "notes": f"Independent {args.venue} cross-venue event; process label not yet expert-reviewed. " + event.get("review_notes", ""),
        })
        print(f"[{index:02d}/{len(events):02d}] {case_id}", flush=True)
    spec = args.output_root / f"double_priority_ground_truth_{args.venue}_v1.csv"
    with spec.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {"events_found": len(events), "clips_created": len(rows), "failures": failures, "spec": str(spec.resolve())}
    (args.output_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
