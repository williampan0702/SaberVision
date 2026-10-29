from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from validate_dual_scoreboard_lamp_events_v47 import read_csv


REVIEW_NOTES = {
    5: ("review", "low", "红灯明确；绿灯只持续单帧，自动状态暂保留 left，可能实际为 double。"),
    8: ("high", "high", "同一次双灯被短暂丢框拆成两段，已合并。"),
    9: ("review", "medium", "亮灯持续较短，但交锋与红灯均可见。"),
    10: ("review", "medium", "红灯较弱，降低红色阈值后恢复；交锋可见。"),
    15: ("review", "medium", "快速平移中短亮灯；交锋可见。"),
    20: ("high", "high", "终局红灯被短暂缺帧拆分，已合并。"),
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
            index, ("high", "high", "逐帧总览中裁判器框与灯光一致。")
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
