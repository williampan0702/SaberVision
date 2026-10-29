from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def merge_candidates(rows: list[dict[str, str]]) -> list[dict]:
    merged: list[dict] = []
    for row in sorted(rows, key=lambda item: float(item["start_sec"])):
        item = {
            "state": row["state"],
            "start_sec": float(row["start_sec"]),
            "end_sec": float(row["end_sec"]),
            "peak_score": float(row["peak_score"]),
            "peak_time": float(row["peak_time"]),
            "positive_frames": int(row["positive_frames"]),
            "fragments": 1,
        }
        if merged:
            previous = merged[-1]
            gap = item["start_sec"] - previous["end_sec"]
            states = {previous["state"], item["state"]}
            compatible = len(states) == 1 or "double" in states or (
                states == {"left", "right"} and gap <= 0.8
            )
            if 0.0 <= gap <= 2.5 and compatible:
                previous["state"] = "double" if "double" in states or states == {"left", "right"} else previous["state"]
                previous["end_sec"] = max(previous["end_sec"], item["end_sec"])
                if item["peak_score"] > previous["peak_score"]:
                    previous["peak_score"] = item["peak_score"]
                    previous["peak_time"] = item["peak_time"]
                previous["positive_frames"] += item["positive_frames"]
                previous["fragments"] += 1
                continue
        merged.append(item)
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description="Finalize the Tallinn v4.7.3 interpolated-track lamp audit.")
    parser.add_argument("review_candidates", type=Path)
    parser.add_argument("score_changes", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    events = merge_candidates(read_csv(args.review_candidates))
    # One additional red-lamp response at 578.6667 s is too short to create a
    # stable runtime event.  A 2-second frame sequence and the 581.0 s left
    # score update both confirm it, so retain it only as a reviewed event.
    events.append(
        {
            "state": "left",
            "start_sec": 578.6667,
            "end_sec": 578.6667,
            "peak_score": 0.029442,
            "peak_time": 578.6667,
            "positive_frames": 1,
            "fragments": 1,
        }
    )
    events.sort(key=lambda item: item["start_sec"])

    changes = read_csv(args.score_changes)
    used_changes: set[int] = set()
    output_rows = []
    for event_id, event in enumerate(events, start=1):
        matches = []
        for index, change in enumerate(changes):
            if index in used_changes:
                continue
            change_time = float(change["time_sec"])
            if event["start_sec"] - 0.5 <= change_time <= event["end_sec"] + 6.0:
                matches.append((abs(change_time - event["end_sec"]), index, change))
        score_change_time = ""
        score_change_side = ""
        score_delay = ""
        if matches:
            _distance, change_index, change = min(matches)
            used_changes.add(change_index)
            score_change_time = change["time_sec"]
            score_change_side = change["side"]
            score_delay = f"{float(change['time_sec']) - event['end_sec']:.4f}"
        automatic = event["positive_frames"] >= 3 and event["peak_score"] >= 0.035
        output_rows.append(
            {
                "event_id": event_id,
                "state": event["state"],
                "start_sec": f"{event['start_sec']:.4f}",
                "end_sec": f"{event['end_sec']:.4f}",
                "duration_sec": f"{max(0.0, event['end_sec'] - event['start_sec']):.4f}",
                "peak_score": f"{event['peak_score']:.5f}",
                "peak_time": f"{event['peak_time']:.4f}",
                "positive_frames": event["positive_frames"],
                "fragments": event["fragments"],
                "automatic_output": int(automatic),
                "review_status": "visually_confirmed_auto" if automatic else "visually_confirmed_review_only",
                "score_change_time": score_change_time,
                "score_change_side": score_change_side,
                "score_delay_sec": score_delay,
            }
        )

    unmatched_changes = [
        change for index, change in enumerate(changes) if index not in used_changes
    ]
    args.output.mkdir(parents=True, exist_ok=True)
    fields = list(output_rows[0])
    with (args.output / "reviewed_lamp_events_v473.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output_rows)
    summary = {
        "reviewed_true_events": len(output_rows),
        "automatic_high_confidence": sum(int(row["automatic_output"]) for row in output_rows),
        "review_only": sum(not int(row["automatic_output"]) for row in output_rows),
        "score_changes_total": len(changes),
        "score_changes_matched": len(used_changes),
        "score_changes_unmatched": len(unmatched_changes),
        "unmatched_score_changes": unmatched_changes,
        "warning": "Score-overlay changes are a weak audit index, not complete frame-level lamp ground truth.",
        "events": output_rows,
    }
    (args.output / "reviewed_summary_v473.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
