from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def overlap(a, b, tolerance: float = 0.8) -> bool:
    return float(a["start_sec"]) <= float(b["end_sec"]) + tolerance and float(a["end_sec"]) >= float(b["start_sec"]) - tolerance


def center_state(event: dict[str, str]) -> str:
    return event.get("center_anchor_state", "none") or "none"


def confirmed_state(event: dict[str, str]) -> str:
    return event.get("multiview_confirmed_state", "none") or "none"


def raw_eligible(event: dict[str, str]) -> bool:
    regions = set(filter(None, event["supporting_regions"].split(";")))
    duration = float(event["duration_sec"])
    return center_state(event) != "none" and confirmed_state(event) != "none" and (
        (len(regions) == 3 and duration >= 0.35)
        or (len(regions) >= 2 and duration >= 0.80)
    )


def weak_cross_confirmed(physical: dict[str, str], regional: dict[str, str]) -> bool:
    """Allow two complementary weak sensors to confirm one short real hit.

    This is intentionally limited to a three-view layout, at least two centre
    colour observations in total, and a non-trivial physical-model response.
    Side-panel-only colour changes therefore remain ineligible.
    """
    regions = set(filter(None, regional.get("supporting_regions", "").split(";")))
    centre_frames = int(regional.get("center_left_frames") or 0) + int(
        regional.get("center_right_frames") or 0
    )
    return (
        len(regions) == 3
        and float(regional.get("duration_sec") or 0.0) >= 0.12
        and centre_frames >= 2
        and float(physical.get("peak_score") or 0.0) >= 0.035
    )


def combined_state(states: set[str]) -> str:
    states.discard("")
    states.discard("none")
    if "double" in states or {"left", "right"} <= states:
        return "double"
    return next(iter(states), "none")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fuse physical events with centre-confirmed three-region evidence.")
    parser.add_argument("physical", type=Path)
    parser.add_argument("regional", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--overlap-tolerance", type=float, default=0.80)
    parser.add_argument("--strong-physical", type=float, default=0.12)
    parser.add_argument("--merge-gap", type=float, default=0.60)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    physical = read_csv(args.physical)
    regional = read_csv(args.regional)
    accepted_atoms = []
    rejected = []

    for event in physical:
        related = [item for item in regional if overlap(event, item, args.overlap_tolerance)]
        eligible_related = [item for item in related if raw_eligible(item)]
        weak_cross_related = [item for item in related if weak_cross_confirmed(event, item)]
        supported_related = eligible_related + [
            item for item in weak_cross_related if item not in eligible_related
        ]
        colour_supported = bool(supported_related)
        strong = float(event.get("peak_score") or 0.0) >= args.strong_physical
        if not colour_supported and not strong:
            rejected.append({"source": "physical", "reason": "no_centre_confirmed_colour_and_not_strong", **event})
            continue
        regions = set()
        starts = [float(event["start_sec"])]
        ends = [float(event["end_sec"])]
        states = {item["state"] for item in supported_related}
        # Weak physical detections help establish that an event exists, but
        # direction comes from centre-colour evidence.  Only a strong physical
        # event may add or override a side state.
        if strong or weak_cross_related or not states:
            states.add(event["state"])
        for item in related:
            regions.update(filter(None, item["supporting_regions"].split(";")))
            if item in supported_related:
                starts.append(float(item["start_sec"]))
                ends.append(float(item["end_sec"]))
        accepted_atoms.append({
            "state": combined_state(states),
            "start_sec": min(starts),
            "end_sec": max(ends),
            "peak_score": float(event.get("peak_score") or 0.0),
            "sources": {"physical_lamp", *(["three_region_colour"] if related else [])},
            "regions": regions,
            "physical_count": 1,
            "regional_count": len(supported_related),
        })

    for event in regional:
        if not raw_eligible(event):
            corroborated = any(
                overlap(event, physical_event, args.overlap_tolerance)
                and weak_cross_confirmed(physical_event, event)
                for physical_event in physical
            )
            if corroborated:
                continue
            reason = "no_centre_confirmation" if center_state(event) == "none" else "insufficient_duration_or_views"
            rejected.append({"source": "regional", "reason": reason, **event})
            continue
        if any(overlap(event, physical_event, args.overlap_tolerance) for physical_event in physical):
            continue
        accepted_atoms.append({
            "state": event["state"],
            "start_sec": float(event["start_sec"]),
            "end_sec": float(event["end_sec"]),
            "peak_score": 0.0,
            "sources": {"three_region_colour"},
            "regions": set(filter(None, event["supporting_regions"].split(";"))),
            "physical_count": 0,
            "regional_count": 1,
        })

    accepted_atoms.sort(key=lambda item: item["start_sec"])
    merged = []
    for atom in accepted_atoms:
        if merged and atom["start_sec"] - merged[-1]["end_sec"] <= args.merge_gap:
            current = merged[-1]
            current["end_sec"] = max(current["end_sec"], atom["end_sec"])
            current["state"] = combined_state({current["state"], atom["state"]})
            current["peak_score"] = max(current["peak_score"], atom["peak_score"])
            current["sources"].update(atom["sources"])
            current["regions"].update(atom["regions"])
            current["physical_count"] += atom["physical_count"]
            current["regional_count"] += atom["regional_count"]
        else:
            merged.append(atom)

    output = []
    for index, item in enumerate(merged, start=1):
        output.append({
            "event_id": index,
            "state": item["state"],
            "start_sec": f"{item['start_sec']:.4f}",
            "end_sec": f"{item['end_sec']:.4f}",
            "duration_sec": f"{item['end_sec'] - item['start_sec']:.4f}",
            "peak_score": f"{item['peak_score']:.5f}",
            "sources": ";".join(sorted(item["sources"])),
            "supporting_regions": ";".join(sorted(item["regions"])),
            "physical_candidates": item["physical_count"],
            "regional_candidates": item["regional_count"],
        })
    output_fields = [
        "event_id", "state", "start_sec", "end_sec", "duration_sec", "peak_score",
        "sources", "supporting_regions", "physical_candidates", "regional_candidates",
    ]
    with (args.output / "final_events.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0]) if output else output_fields)
        writer.writeheader()
        writer.writerows(output)
    rejected_fields = sorted({key for item in rejected for key in item}) or [
        "source", "reason", "state", "start_sec", "end_sec", "duration_sec"
    ]
    with (args.output / "rejected_candidates.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rejected_fields)
        writer.writeheader()
        writer.writerows(rejected)
    summary = {
        "physical_candidates": len(physical),
        "regional_candidates": len(regional),
        "center_confirmed_regional_candidates": sum(center_state(item) != "none" for item in regional),
        "eligible_regional_candidates": sum(raw_eligible(item) for item in regional),
        "final_events": len(output),
        "rejected_candidates": len(rejected),
        "rules": {
            "direction": "centre confirms at least one colour; dim red may use multi-view aggregate while missing green requires strong bilateral remote peaks; weak physical state cannot override centre-supported colour",
            "raw": "centre-confirmed plus 3 views >=0.35s, or centre-confirmed plus >=2 views >=0.80s",
            "physical": "strong physical event, or overlap with centre-confirmed colour",
            "side_only": "always rejected",
            "weak_cross_sensor": "three views + >=2 centre observations + overlapping physical peak >=0.035",
        },
    }
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
