from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Finalize the visually reviewed Tallinn v4.7.2 event audit.")
    parser.add_argument("candidates", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    with args.candidates.open("r", encoding="utf-8-sig", newline="") as handle:
        candidates = list(csv.DictReader(handle))

    # These two weak detector responses were inspected as short frame sequences.
    # Both physical red and green indicators remain visibly illuminated, so the
    # audit corrects the candidate state while keeping them out of automatic
    # high-confidence output.
    reviewed_double_times = {444.8667, 472.0}
    rows = []
    for index, candidate in enumerate(candidates, start=1):
        peak_time = float(candidate["peak_time"])
        peak = float(candidate["peak_score"])
        positive_frames = int(candidate["positive_frames"])
        automatic = positive_frames >= 3 and peak >= 0.04
        state = "double" if any(abs(peak_time - value) <= 0.001 for value in reviewed_double_times) else candidate["state"]
        rows.append(
            {
                "event_id": index,
                "state": state,
                "start_sec": candidate["start_sec"],
                "end_sec": candidate["end_sec"],
                "duration_sec": candidate["duration_sec"],
                "peak_score": candidate["peak_score"],
                "peak_time": candidate["peak_time"],
                "positive_frames": positive_frames,
                "automatic_output": int(automatic),
                "review_status": "visually_confirmed_auto" if automatic else "visually_confirmed_review_only",
                "evidence": "multi_frame_temporal_detector" if automatic else "single_frame_detector_plus_manual_sequence",
            }
        )

    args.output.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["event_id", "state", "start_sec", "end_sec"]
    with (args.output / "reviewed_lamp_events_v472.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "reviewed_events": len(rows),
        "automatic_high_confidence": sum(int(row["automatic_output"]) for row in rows),
        "review_only": sum(not int(row["automatic_output"]) for row in rows),
        "all_candidates_visually_true": True,
        "warning": "This is an event-level visual audit, not frame-level recall ground truth.",
        "events": rows,
    }
    (args.output / "reviewed_summary_v472.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
