from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import median


VALID_STATES = {"left", "right", "double"}


def _finite_float(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def normalize_state(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    aliases = {
        "l": "left", "left_on": "left",
        "r": "right", "right_on": "right",
        "both": "double", "double_light": "double",
    }
    text = aliases.get(text, text)
    return text if text in VALID_STATES else None


def _first_float(row: dict, names: tuple[str, ...], default=None):
    for name in names:
        value = row.get(name)
        if value in (None, ""):
            continue
        result = _finite_float(value)
        if result is not None:
            return result
    return default


def load_device_events(
    path: Path,
    authority: str,
    source_name: str | None = None,
) -> list[dict]:
    if source_name is None:
        # A per-exchange file is often just named timeline.csv/final_events.csv,
        # while validation fixtures such as mixed_final_events.csv carry their
        # identity in the stem.  Keep independent files as independent sources.
        source_name = (
            path.parent.name
            if path.stem in {"timeline", "final_events", "review_events"}
            else path.stem
        )
    events = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row_index, row in enumerate(reader, 1):
            state = normalize_state(
                row.get("state") or row.get("side") or row.get("decision")
            )
            if state is None:
                continue
            start = _first_float(
                row, ("start_sec", "start_time", "detected_time", "time_sec")
            )
            if start is None:
                continue
            if start < 0:
                continue
            end = _first_float(
                row, ("end_sec", "end_time"), default=start
            )
            if end is None or end < start:
                end = start
            row_authority = authority
            if str(row.get("confidence_tier", "")).strip().lower() == "review":
                row_authority = "review"
            channel_text = row.get("sources") or row.get("regions") or ""
            channels = [
                item.strip() for item in str(channel_text).split(";")
                if item.strip()
            ]
            events.append({
                "source": source_name,
                "source_file": str(path),
                "source_row": row_index,
                "authority": row_authority,
                "state": state,
                "start_sec": round(start, 4),
                "end_sec": round(end, 4),
                "channels": channels,
                "review_reason": row.get("review_reason") or None,
            })
    return events


def select_action_sequence(
    action_summary: dict | None,
    sequence_key: str | None,
) -> tuple[str | None, dict | None]:
    if not action_summary:
        return None, None
    sequences = action_summary.get("sequences")
    if not isinstance(sequences, dict) or not sequences:
        return None, None
    if sequence_key:
        if sequence_key in sequences:
            return sequence_key, sequences[sequence_key]
        matches = [key for key in sequences if sequence_key in key]
        if len(matches) == 1:
            return matches[0], sequences[matches[0]]
        raise KeyError(
            f"Action sequence key {sequence_key!r} matched {len(matches)} entries"
        )
    if len(sequences) == 1:
        key = next(iter(sequences))
        return key, sequences[key]
    raise ValueError("--action-sequence-key is required for a multi-sequence summary")


def summarize_action(sequence: dict | None) -> dict:
    if not sequence:
        return {"available": False, "left": None, "right": None}
    result = {"available": True}
    for side in ("left", "right"):
        athlete = (sequence.get("athletes") or {}).get(side) or {}
        result[side] = {
            "action_status": athlete.get("action_status"),
            "movement": athlete.get("dominant_pre_contact_movement"),
            "lunge_present": athlete.get("lunge_present"),
            "lunge_evidence_status": athlete.get("lunge_evidence_status"),
            "requires_resolution": athlete.get(
                "requires_downstream_lunge_resolution", False
            ),
        }
    result["phase_start_frame"] = sequence.get("phase_start_frame")
    result["contact_frame"] = sequence.get("contact_frame")
    return result


def summarize_blade(
    blade_payload: dict | None,
    event_start: float,
    pre_seconds: float = 2.0,
    post_seconds: float = 0.25,
) -> dict:
    if not blade_payload:
        return {"available": False, "window": None, "left": None, "right": None}
    lo = event_start - pre_seconds
    hi = event_start + post_seconds
    frames = []
    for frame in blade_payload.get("frames", []):
        if not isinstance(frame, dict):
            continue
        frame_time = _finite_float(frame.get("time_sec"))
        if frame_time is not None and lo <= frame_time <= hi:
            frames.append((frame, frame_time))
    result = {
        "available": True,
        "window": {"start_sec": round(lo, 4), "end_sec": round(hi, 4)},
    }
    for side in ("left", "right"):
        records = []
        for frame, frame_time in frames:
            record = frame.get(side)
            if not isinstance(record, dict):
                continue
            guard = record.get("guard")
            tip = record.get("tip")
            if not guard or not tip or len(guard) < 2 or len(tip) < 2:
                continue
            tip_x = _finite_float(tip[0])
            guard_x = _finite_float(guard[0])
            if tip_x is None or guard_x is None:
                continue
            toward = tip_x > guard_x if side == "left" else tip_x < guard_x
            records.append({
                "time_sec": frame_time,
                "source": record.get("source"),
                "toward_opponent": toward,
                "angle_deg": record.get("angle_deg"),
                "length_px": record.get("length_px"),
            })
        geometry = [record for record in records if record["source"] == "geometry"]
        temporal = [record for record in records if record["source"] == "temporal"]
        toward_count = sum(record["toward_opponent"] for record in records)
        if geometry:
            visibility = "clear_geometry"
        elif temporal:
            visibility = "temporal_only"
        else:
            visibility = "absent"
        angles = [
            angle
            for record in records
            if (angle := _finite_float(record.get("angle_deg"))) is not None
        ]
        result[side] = {
            "visibility": visibility,
            "observations": len(records),
            "geometry_observations": len(geometry),
            "temporal_observations": len(temporal),
            "toward_opponent_observations": toward_count,
            "median_angle_deg": round(median(angles), 3) if angles else None,
            "first_seen_sec": (
                round(min(record["time_sec"] for record in records), 4)
                if records else None
            ),
            "last_seen_sec": (
                round(max(record["time_sec"] for record in records), 4)
                if records else None
            ),
            "capability": "direction_only_not_contact_or_priority",
        }
    return result


def summarize_blade_timing(
    timing_payload: dict | None,
    event_start: float,
    pre_seconds: float = 2.0,
    post_seconds: float = 0.25,
) -> dict:
    """Select conservative blade/Pose timing candidates around one lamp event."""
    if not timing_payload:
        return {"available": False, "window": None}
    lo = event_start - pre_seconds
    hi = event_start + post_seconds
    result = {
        "available": True,
        "window": {"start_sec": round(lo, 4), "end_sec": round(hi, 4)},
        "attack_extension_start": {},
    }
    for side in ("left", "right"):
        extension = (timing_payload.get("attack_extension_start") or {}).get(side) or {}
        start = _finite_float(extension.get("start_sec"))
        end = _finite_float(extension.get("end_sec"))
        in_window = start is not None and lo <= start <= hi
        result["attack_extension_start"][side] = {
            "state": extension.get("state") if in_window else "not_in_event_window",
            "start_sec": start if in_window else None,
            "end_sec": end if in_window else None,
        }

    def within(items: list[dict] | None) -> list[dict]:
        selected = []
        for item in items or []:
            if not isinstance(item, dict):
                continue
            item_time = _finite_float(item.get("time_sec"))
            if item_time is not None and lo <= item_time <= hi:
                selected.append(item)
        return selected

    tips = timing_payload.get("tip_approach_candidates") or {}
    result["tip_approach_candidates"] = {
        side: within(tips.get(side)) for side in ("left", "right")
    }
    result["blade_proximity_candidates"] = within(
        timing_payload.get("blade_proximity_candidates")
    )
    result["possible_contact_windows"] = within(
        timing_payload.get("possible_contact_windows")
    )
    result["possible_parry_candidates"] = within(
        timing_payload.get("possible_parry_candidates")
    )
    result["capability"] = "candidate_timing_only_not_final_priority"
    return result


def _events_share_exchange(a: dict, b: dict, tolerance: float) -> bool:
    return not (
        a["end_sec"] + tolerance < b["start_sec"]
        or b["end_sec"] + tolerance < a["start_sec"]
    )


def cluster_events(events: list[dict], tolerance: float = 0.35) -> list[list[dict]]:
    clusters: list[list[dict]] = []
    for event in sorted(events, key=lambda item: (item["start_sec"], item["end_sec"])):
        matching = [
            index for index, cluster in enumerate(clusters)
            if any(_events_share_exchange(event, member, tolerance) for member in cluster)
        ]
        if not matching:
            clusters.append([event])
            continue
        target = matching[0]
        clusters[target].append(event)
        for extra in reversed(matching[1:]):
            clusters[target].extend(clusters.pop(extra))
    return clusters


def source_state(events: list[dict], double_onset_max_gap: float = 0.25) -> str:
    states = {event["state"] for event in events}
    if "double" in states:
        return "double"
    if states == {"left", "right"}:
        left_events = [event for event in events if event["state"] == "left"]
        right_events = [event for event in events if event["state"] == "right"]
        simultaneous_pair = any(
            abs(left["start_sec"] - right["start_sec"]) <= double_onset_max_gap
            and left["start_sec"] <= right["end_sec"]
            and right["start_sec"] <= left["end_sec"]
            for left in left_events
            for right in right_events
        )
        if simultaneous_pair:
            return "double"
        earliest = min(events, key=lambda event: (event["start_sec"], event["end_sec"]))
        return earliest["state"]
    return next(iter(states))


def physical_double_subset_resolution(
    by_source: dict[str, list[dict]], source_results: dict[str, str], maximum_start_gap: float = 0.25
) -> bool:
    """Allow a direct physical double when another detector sees one subset lamp."""
    double_sources = [
        source for source, state in source_results.items()
        if state == "double" and source.startswith("physical_temporal")
    ]
    other_sources = [source for source in source_results if source not in double_sources]
    if len(double_sources) != 1 or not other_sources:
        return False
    if any(source_results[source] not in {"left", "right"} for source in other_sources):
        return False
    physical_start = min(event["start_sec"] for event in by_source[double_sources[0]])
    return all(
        abs(min(event["start_sec"] for event in by_source[source]) - physical_start) <= maximum_start_gap
        for source in other_sources
    )


def explicit_double_with_colour_support(
    by_source: dict[str, list[dict]],
    source_results: dict[str, str],
    maximum_detection_lag: float = 0.75,
) -> bool:
    """Accept an explicit double when another source sees both colours nearby.

    A physical lamp may be temporarily hidden by a fencer, so its detected
    colour onsets can differ even though a gated source sees the double at once.
    The supporting source must still observe both colours near that explicit
    onset; a single-colour subset is not enough for this rule.
    """
    double_sources = [
        source for source, state in source_results.items()
        if state == "double"
        and any(event["state"] == "double" for event in by_source[source])
    ]
    for double_source in double_sources:
        double_start = min(event["start_sec"] for event in by_source[double_source])
        supporting_sources = [source for source in source_results if source != double_source]
        if not supporting_sources:
            continue
        supported = True
        for source in supporting_sources:
            events = by_source[source]
            for colour in ("left", "right"):
                nearby = any(
                    event["state"] in {colour, "double"}
                    and event["end_sec"] >= double_start
                    and event["start_sec"] <= double_start + maximum_detection_lag
                    for event in events
                )
                if not nearby:
                    supported = False
                    break
            if not supported:
                break
        if supported:
            return True
    return False


def resolve_device_cluster(cluster: list[dict]) -> dict:
    automatic = [event for event in cluster if event["authority"] == "automatic"]
    review = [event for event in cluster if event["authority"] != "automatic"]
    by_source = defaultdict(list)
    for event in automatic:
        by_source[event["source"]].append(event)
    source_results = {
        source: source_state(events) for source, events in by_source.items()
    }
    unique_results = set(source_results.values())
    resolved_start = min(event["start_sec"] for event in cluster)
    resolution_rule = None
    if not source_results:
        status = "review_required" if review else "no_device_result"
        state = None
        resolution_rule = "no_automatic_source"
    elif len(unique_results) == 1:
        status = "resolved_by_any_clear_source"
        state = next(iter(unique_results))
        resolution_rule = "automatic_sources_agree"
    elif physical_double_subset_resolution(by_source, source_results):
        status = "resolved_by_any_clear_source"
        state = "double"
        resolution_rule = "physical_near_simultaneous_double_over_single_subset"
        resolved_start = min(
            event["start_sec"]
            for source, events in by_source.items()
            if source.startswith("physical_temporal")
            for event in events
            if event["state"] == "double"
        )
    elif explicit_double_with_colour_support(by_source, source_results):
        status = "resolved_by_any_clear_source"
        state = "double"
        resolution_rule = "explicit_double_with_cross_source_colour_support"
        resolved_start = min(
            event["start_sec"]
            for source, events in by_source.items()
            if source_results[source] == "double"
            for event in events
            if event["state"] == "double"
        )
    else:
        status = "automatic_source_conflict"
        state = None
        resolution_rule = "unresolved_automatic_source_conflict"
    return {
        "start_sec": round(resolved_start, 4),
        "end_sec": round(max(event["end_sec"] for event in cluster), 4),
        "touch_status": status,
        "touch_state": state,
        "resolution_rule": resolution_rule,
        "automatic_source_results": source_results,
        "automatic_events": automatic,
        "review_events": review,
    }


def priority_context(
    touch_state: str | None,
    action: dict,
    blade: dict,
    blade_timing: dict | None = None,
) -> dict:
    if touch_state in {"left", "right"}:
        return {
            "status": "resolved_by_single_light",
            "awarded_side": touch_state,
            "candidate_side": touch_state,
            "reason": "A reliable single-light event does not require action voting.",
        }
    if touch_state != "double":
        return {
            "status": "not_resolved",
            "awarded_side": None,
            "candidate_side": None,
            "reason": "No resolved device event is available.",
        }
    left = action.get("left") or {}
    right = action.get("right") or {}
    candidate = None
    reason = "Double light requires richer attack/parry/contact timing evidence."
    if left.get("lunge_present") is True and right.get("lunge_present") is False:
        candidate = "left"
        reason = "Only the left side has confirmed lunge geometry."
    elif right.get("lunge_present") is True and left.get("lunge_present") is False:
        candidate = "right"
        reason = "Only the right side has confirmed lunge geometry."
    elif left.get("movement") == "advance" and right.get("movement") == "retreat":
        candidate = "left"
        reason = "Left advances while right retreats."
    elif right.get("movement") == "advance" and left.get("movement") == "retreat":
        candidate = "right"
        reason = "Right advances while left retreats."
    return {
        "status": "context_candidate_not_final" if candidate else "unresolved_double",
        "awarded_side": None,
        "candidate_side": candidate,
        "reason": reason,
        "blade_limitation": (
            "Blade timing entries are candidates only. Empty entries mean not "
            "observed in direct geometry frames, not proof that no contact/parry occurred."
        ),
        "blade_available": bool(blade.get("available")),
        "blade_timing_available": bool((blade_timing or {}).get("available")),
    }


def fuse(
    action_summary: dict | None,
    action_sequence_key: str | None,
    blade_payload: dict | None,
    device_events: list[dict],
    merge_tolerance: float = 0.35,
    blade_timing_payload: dict | None = None,
) -> dict:
    selected_key, sequence = select_action_sequence(
        action_summary, action_sequence_key
    )
    action = summarize_action(sequence)
    fused_events = []
    for event_index, cluster in enumerate(
        cluster_events(device_events, merge_tolerance), 1
    ):
        device = resolve_device_cluster(cluster)
        blade = summarize_blade(blade_payload, device["start_sec"])
        blade_timing = summarize_blade_timing(
            blade_timing_payload, device["start_sec"]
        )
        priority = priority_context(
            device["touch_state"], action, blade, blade_timing
        )
        fused_events.append({
            "event_id": event_index,
            "start_sec": device["start_sec"],
            "end_sec": device["end_sec"],
            "touch_status": device["touch_status"],
            "touch_state": device["touch_state"],
            "awarded_side": priority["awarded_side"],
            "priority": priority,
            "action": action,
            "blade": blade,
            "blade_timing": blade_timing,
            "device_evidence": {
                "resolution_rule": device["resolution_rule"],
                "automatic_source_results": device["automatic_source_results"],
                "automatic_events": device["automatic_events"],
                "review_events": device["review_events"],
            },
        })
    return {
        "schema_version": "1.0",
        "action_sequence_key": selected_key,
        "policy": {
            "device_result": (
                "One reliable automatic source is sufficient; no confidence vote or N-of-3 rule."
            ),
            "automatic_conflict": "Do not guess; send to review.",
            "review_event": "Never promote a review-only event to an automatic result.",
            "action_and_blade": (
                "Context for double-light priority; they do not override a clear lamp result."
            ),
        },
        "source_availability": {
            "action": action.get("available", False),
            "blade": bool(blade_payload),
            "blade_timing": bool(blade_timing_payload),
            "device_events": len(device_events),
        },
        "events": fused_events,
    }


def write_event_csv(payload: dict, path: Path) -> None:
    rows = []
    for event in payload["events"]:
        rows.append({
            "event_id": event["event_id"],
            "start_sec": event["start_sec"],
            "end_sec": event["end_sec"],
            "touch_status": event["touch_status"],
            "touch_state": event["touch_state"],
            "awarded_side": event["awarded_side"],
            "priority_status": event["priority"]["status"],
            "priority_candidate_side": event["priority"].get("candidate_side"),
            "automatic_sources": ";".join(
                sorted(event["device_evidence"]["automatic_source_results"])
            ),
            "review_events": len(event["device_evidence"]["review_events"]),
        })
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        fieldnames = [
            "event_id", "start_sec", "end_sec", "touch_status", "touch_state",
            "awarded_side", "priority_status", "priority_candidate_side",
            "automatic_sources", "review_events",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fuse synchronized SaberVision evidence without confidence voting."
    )
    parser.add_argument("--action-summary", type=Path)
    parser.add_argument("--action-sequence-key")
    parser.add_argument("--blade-json", type=Path)
    parser.add_argument("--blade-timing-json", type=Path)
    parser.add_argument("--device-final", type=Path, nargs="*", default=[])
    parser.add_argument("--device-review", type=Path, nargs="*", default=[])
    parser.add_argument("--merge-tolerance", type=float, default=0.35)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()

    action_summary = read_json(args.action_summary) if args.action_summary else None
    blade_payload = read_json(args.blade_json) if args.blade_json else None
    blade_timing_payload = (
        read_json(args.blade_timing_json) if args.blade_timing_json else None
    )
    device_events = []
    for path in args.device_final:
        device_events.extend(load_device_events(path, "automatic"))
    for path in args.device_review:
        device_events.extend(load_device_events(path, "review"))
    payload = fuse(
        action_summary,
        args.action_sequence_key,
        blade_payload,
        device_events,
        args.merge_tolerance,
        blade_timing_payload,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_event_csv(payload, args.output_csv)
    print(json.dumps({
        "events": len(payload["events"]),
        "resolved": sum(
            event["touch_status"] == "resolved_by_any_clear_source"
            for event in payload["events"]
        ),
        "conflicts": sum(
            event["touch_status"] == "automatic_source_conflict"
            for event in payload["events"]
        ),
        "output_json": str(args.output_json),
        "output_csv": str(args.output_csv),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
