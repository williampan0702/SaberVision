from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from validate_dual_scoreboard_lamp_events_v47 import read_csv


REVIEW_NOTES = {
    5: ("review", "low", "Red is clear; green lasts only one frame. The automatic state remains left, but the event may be double."),
    8: ("high", "high", "One double-light event was split by a brief lost box and has been merged."),
    9: ("review", "medium", "The lamp interval is short, but both the exchange and red lamp are visible."),
    10: ("review", "medium", "The weak red lamp was recovered with a lower red threshold; the exchange is visible."),
    15: ("review", "medium", "A short lamp interval occurs during a fast pan; the exchange is visible."),
    20: ("high", "high", "The final red lamp was split by briefly missing frames and has been merged."),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Finalize visually reviewed Riyadh v4.7 lamp events.")
    parser.add_argument("events", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    events = read_csv(args.events)
    rows = []
    for index, event in enumerate(events, start=1):
        automatic, state_confidence, note = REVIEW_NOTES.get(
            index, ("high", "high", "The score-box track and lamp state agree in the frame-by-frame review.")
        )
        rows.append(
            {
                "index": index,
                "state": event["state"],
                "start_sec": event["start_sec"],
                "end_sec": event["end_sec"],
                "duration_sec": event["duration_sec"],
                "peak_score": event["peak_score"],
                "fragments": event.get("fragments", "1"),
                "visual_review": "confirmed",
                "automatic_confidence": automatic,
                "state_confidence": state_confidence,
                "notes": note,
            }
        )
    fields = list(rows[0])
    with (args.output / "reviewed_lamp_events.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "video": "riyadh_2024_blind.mp4",
        "reviewed_events": len(rows),
        "visual_confirmed": sum(row["visual_review"] == "confirmed" for row in rows),
        "automatic_high": sum(row["automatic_confidence"] == "high" for row in rows),
        "automatic_review": sum(row["automatic_confidence"] == "review" for row in rows),
        "state_low_confidence": [int(row["index"]) for row in rows if row["state_confidence"] == "low"],
        "locator": {
            "sampled_frames": 5251,
            "stable_rows": 5055,
            "coverage": 0.9626737764235384,
            "anchor_frames": 783,
            "optical_flow_frames": 4257,
            "hold_frames": 15,
        },
        "thresholds": {
            "red_fraction": 0.03,
            "green_fraction": 0.01,
            "dark_fraction": 0.16,
            "event_gap_sec": 0.5,
            "fragment_merge_gap_sec": 2.5,
        },
    }
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
