from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read_events(path: Path, source: str) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    events = []
    for row in rows:
        events.append(
            {
                "state": row["state"].lower(),
                "start_sec": float(row["start_sec"]),
                "end_sec": float(row["end_sec"]),
                "peak_score": float(row.get("peak_score") or 0.0),
                "sources": {source},
            }
        )
    return events


def overlaps(first: dict, second: dict, margin: float) -> bool:
    return (
        first["start_sec"] <= second["end_sec"] + margin
        and second["start_sec"] <= first["end_sec"] + margin
    )


def consolidate(events: list[dict], fragment_gap: float) -> list[dict]:
    result = []
    for event in sorted(events, key=lambda item: (item["start_sec"], item["end_sec"])):
        if result:
            previous = result[-1]
            gap = event["start_sec"] - previous["end_sec"]
            if previous["state"] == event["state"] and gap <= fragment_gap:
                previous["end_sec"] = max(previous["end_sec"], event["end_sec"])
                previous["peak_score"] = max(previous["peak_score"], event["peak_score"])
                previous["sources"].update(event["sources"])
                continue
            if gap <= fragment_gap and (
                previous["state"] == "double"
                or event["state"] == "double"
                or {previous["state"], event["state"]} == {"left", "right"}
            ):
                previous["state"] = "double"
                previous["end_sec"] = max(previous["end_sec"], event["end_sec"])
                previous["peak_score"] = max(previous["peak_score"], event["peak_score"])
                previous["sources"].update(event["sources"])
                continue
        result.append(event)
    return result


def merge(
    primary: list[dict], fallback: list[dict], overlap_margin: float, fragment_gap: float
) -> list[dict]:
    # The colour-checked fallback sees actual lamp brightness directly, so its
    # boundaries win when a device-gated event overlaps it. The gated event
    # contributes confirmation but cannot extend a stale track past lamp-off.
    result = [dict(event) for event in fallback]
    for event in primary:
        match = next(
            (
                existing
                for existing in result
                if (existing["state"] == event["state"] or existing["state"] == "double")
                and overlaps(existing, event, overlap_margin)
            ),
            None,
        )
        if match is None:
            result.append(dict(event))
            continue
        match["sources"].update(event["sources"])
        match["peak_score"] = max(match["peak_score"], event["peak_score"])
    return consolidate(result, fragment_gap)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fuse device-gated and colour-checked fallback lamp events.")
    parser.add_argument("primary_events", type=Path)
    parser.add_argument("fallback_events", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--overlap-margin", type=float, default=0.15)
    parser.add_argument("--fragment-gap", type=float, default=0.65)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    primary = read_events(args.primary_events, "device_gated")
    fallback_source = (
        "physical_temporal"
        if "physical_temporal" in args.fallback_events.name
        else "edge_fallback"
    )
    fallback = read_events(args.fallback_events, fallback_source)
    fused = merge(primary, fallback, args.overlap_margin, args.fragment_gap)
    rows = []
    for index, event in enumerate(fused, 1):
        rows.append(
            {
                "event_id": index,
                "state": event["state"],
                "start_sec": f"{event['start_sec']:.4f}",
                "end_sec": f"{event['end_sec']:.4f}",
                "duration_sec": f"{event['end_sec'] - event['start_sec']:.4f}",
                "peak_score": f"{event['peak_score']:.5f}",
                "sources": "+".join(sorted(event["sources"])),
            }
        )
    fields = ["event_id", "state", "start_sec", "end_sec", "duration_sec", "peak_score", "sources"]
    with (args.output / "fused_events.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "primary_events": len(primary),
        "fallback_events": len(fallback),
        "fused_events": rows,
        "boundary_rule": "colour-checked fallback boundaries win on compatible overlap",
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
