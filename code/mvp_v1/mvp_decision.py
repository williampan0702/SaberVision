from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
FUSION_DIR = HERE.parent / "fusion_v1"
if str(FUSION_DIR) not in sys.path:
    sys.path.insert(0, str(FUSION_DIR))

from fuse_synchronized_evidence import (  # noqa: E402
    fuse,
    load_device_events,
    read_json,
)


SIDES = ("left", "right")
LAMP_SPEED_WINDOW_SEC = 0.35
LAMP_SPEED_CONSISTENCY_WINDOW_SEC = 0.75
LAMP_SPEED_MARGIN_BODY_HEIGHTS_PER_SEC = 0.78315
# Double-light automatic awards use a deliberately larger safety margin than
# candidate display.  The lower margin remains useful supporting evidence, but
# it is not strong enough to survive parry/riposte counterexamples on its own.
# A 1.70 automatic threshold preserves every previously correct automatic
# decision while leaving a useful buffer above the 1.558 cross-venue no-point
# stress case.  Candidate display keeps the lower 0.78315 support threshold.
LAMP_SPEED_AUTOMATIC_MARGIN_BODY_HEIGHTS_PER_SEC = 1.70
LAMP_SPEED_MINIMUM_SAMPLES = 5


def _finite_float(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _candidate(side: str, source: str, reason: str, decisive: bool = False) -> dict:
    return {
        "side": side,
        "source": source,
        "reason": reason,
        "decisive_for_prototype": decisive,
    }


def priority_candidates(event: dict) -> list[dict]:
    action = event.get("action") or {}
    timing = event.get("blade_timing") or {}
    result: list[dict] = []

    left = action.get("left") or {}
    right = action.get("right") or {}
    if left.get("movement") == "advance" and right.get("movement") == "retreat":
        result.append(_candidate("left", "action_movement", "left advances while right retreats"))
    elif right.get("movement") == "advance" and left.get("movement") == "retreat":
        result.append(_candidate("right", "action_movement", "right advances while left retreats"))

    if left.get("lunge_present") is True and right.get("lunge_present") is False:
        result.append(_candidate("left", "action_lunge", "only left has confirmed lunge geometry"))
    elif right.get("lunge_present") is True and left.get("lunge_present") is False:
        result.append(_candidate("right", "action_lunge", "only right has confirmed lunge geometry"))

    lamp_speed = action.get("lamp_anchored_speed") or {}
    if lamp_speed.get("state") == "candidate" and lamp_speed.get("candidate_side") in SIDES:
        side = lamp_speed["candidate_side"]
        result.append(
            _candidate(
                side,
                "lamp_anchored_speed",
                lamp_speed.get("reason") or f"{side} has a large inward-speed advantage immediately before the lamps",
            )
        )

    starts = timing.get("attack_extension_start") or {}
    observed = {}
    for side in SIDES:
        record = starts.get(side) or {}
        start = _finite_float(record.get("start_sec"))
        if record.get("state") == "candidate" and start is not None:
            observed[side] = start
    if len(observed) == 1:
        side = next(iter(observed))
        result.append(_candidate(side, "attack_extension_timing", f"only {side} has sustained extension onset"))
    elif len(observed) == 2:
        delta = observed["right"] - observed["left"]
        if abs(delta) >= 0.15:
            side = "left" if delta > 0 else "right"
            result.append(
                _candidate(
                    side,
                    "attack_extension_timing",
                    f"{side} extension begins {abs(delta):.3f}s earlier",
                )
            )

    # Direct blade observations are exposed to the user, but the current blade
    # model is not reliable enough to turn 2-D proximity into a priority award.
    return result


def lamp_anchored_speed_evidence(
    pose_payload: dict | None,
    event_start_sec: float,
    window_sec: float = LAMP_SPEED_WINDOW_SEC,
    margin: float = LAMP_SPEED_MARGIN_BODY_HEIGHTS_PER_SEC,
) -> dict:
    """Measure final inward speed around the actual lamp onset.

    This is calibrated as supporting evidence only.  It cannot distinguish an
    attack from a parry-riposte or a simultaneous action by itself.
    """
    if not pose_payload:
        return {"state": "unavailable", "candidate_side": None}
    event_start_sec = _finite_float(event_start_sec)
    if event_start_sec is None:
        return {"state": "unavailable", "candidate_side": None}
    lo = event_start_sec - max(window_sec, LAMP_SPEED_CONSISTENCY_WINDOW_SEC)
    values: dict[str, list[float]] = {"left": [], "right": []}
    recent_values: dict[str, list[float]] = {"left": [], "right": []}
    for frame in pose_payload.get("frames", []):
        time_sec = _finite_float(frame.get("time_sec"))
        if time_sec is None or not lo <= time_sec <= event_start_sec:
            continue
        for side in SIDES:
            features = ((frame.get(side) or {}).get("features") or {})
            speed_value = _finite_float(features.get("inward_speed_body_heights_per_sec"))
            if speed_value is None:
                continue
            values[side].append(speed_value)
            if event_start_sec - window_sec <= time_sec:
                recent_values[side].append(speed_value)
    counts = {side: len(items) for side, items in recent_values.items()}
    if min(counts.values()) < LAMP_SPEED_MINIMUM_SAMPLES:
        return {
            "state": "insufficient_samples",
            "candidate_side": None,
            "window_sec": window_sec,
            "sample_counts": counts,
        }
    means = {side: statistics.fmean(items) for side, items in recent_values.items()}
    delta = means["left"] - means["right"]
    consistency_counts = {side: len(items) for side, items in values.items()}
    consistency_means = {
        side: statistics.fmean(items) if items else None for side, items in values.items()
    }
    consistency_delta = (
        float(consistency_means["left"]) - float(consistency_means["right"])
        if all(consistency_means[side] is not None for side in SIDES)
        else None
    )
    candidate_side = None
    if delta > margin:
        candidate_side = "left"
    elif delta < -margin:
        candidate_side = "right"
    state = "candidate" if candidate_side else "balanced_or_ambiguous"
    direction_consistent = (
        consistency_delta is not None
        and delta * consistency_delta > 0
        and min(consistency_counts.values()) >= LAMP_SPEED_MINIMUM_SAMPLES
    )
    reason = (
        f"{candidate_side} inward-speed advantage is {abs(delta):.3f} body-heights/s "
        f"in the {window_sec:.2f}s before lamp onset"
        if candidate_side
        else f"final inward-speed difference {abs(delta):.3f} is below the supporting margin"
    )
    return {
        "state": state,
        "candidate_side": candidate_side,
        "reason": reason,
        "window_sec": window_sec,
        "margin_body_heights_per_sec": margin,
        "mean_inward_speed": {side: round(value, 4) for side, value in means.items()},
        "left_minus_right": round(delta, 4),
        "sample_counts": counts,
        "consistency_window_sec": LAMP_SPEED_CONSISTENCY_WINDOW_SEC,
        "consistency_mean_inward_speed": {
            side: round(float(value), 4) if value is not None else None
            for side, value in consistency_means.items()
        },
        "consistency_left_minus_right": round(consistency_delta, 4) if consistency_delta is not None else None,
        "consistency_sample_counts": consistency_counts,
        "direction_consistent_075": direction_consistent,
        "capability": "supporting_only_not_parry_or_simultaneous_resolution",
    }


def adjudicate_event(event: dict) -> dict:
    touch_status = event.get("touch_status")
    touch_state = event.get("touch_state")
    if touch_status != "resolved_by_any_clear_source":
        return {
            "status": "review_required",
            "awarded_side": None,
            "point_awarded": None,
            "reason": "Automatic lamp sources are missing or conflict.",
            "priority_candidates": [],
        }
    if touch_state in {"left", "right"}:
        return {
            "status": "final_single_light",
            "awarded_side": touch_state,
            "point_awarded": True,
            "reason": "One clear automatic lamp source is sufficient; no confidence vote is used.",
            "priority_candidates": [],
        }
    if touch_state != "double":
        return {
            "status": "review_required",
            "awarded_side": None,
            "point_awarded": None,
            "reason": "The lamp state is not usable.",
            "priority_candidates": [],
        }

    simultaneous = event.get("simultaneous_evidence") or {}
    if simultaneous.get("state") == "confirmed":
        return {
            "status": "final_simultaneous_no_point",
            "awarded_side": None,
            "point_awarded": False,
            "reason": "Confirmed simultaneous double light; neither side receives a point.",
            "priority_candidates": [],
        }

    candidates = priority_candidates(event)
    timing = [item for item in candidates if item["source"] == "attack_extension_timing"]
    speed = [item for item in candidates if item["source"] == "lamp_anchored_speed"]
    lamp_speed = (event.get("action") or {}).get("lamp_anchored_speed") or {}
    speed_value = _finite_float(lamp_speed.get("left_minus_right"))
    speed_delta = abs(speed_value) if speed_value is not None else 0.0
    consistency_value = _finite_float(lamp_speed.get("consistency_left_minus_right"))
    consistency_speed_delta = (
        abs(consistency_value) if consistency_value is not None else 0.0
    )
    strong_speed = (
        speed
        if speed_delta >= LAMP_SPEED_AUTOMATIC_MARGIN_BODY_HEIGHTS_PER_SEC
        and consistency_speed_delta >= LAMP_SPEED_AUTOMATIC_MARGIN_BODY_HEIGHTS_PER_SEC
        and lamp_speed.get("direction_consistent_075") is True
        else []
    )
    if len({item["side"] for item in strong_speed}) == 1:
        side = strong_speed[0]["side"]
        return {
            "status": "prototype_priority_award_strong_speed",
            "awarded_side": side,
            "point_awarded": True,
            "reason": "Very strong lamp-anchored speed advantage exceeds the automatic margin in both 0.35s and 0.75s windows.",
            "priority_candidates": candidates,
            "automatic_speed_margin_body_heights_per_sec": LAMP_SPEED_AUTOMATIC_MARGIN_BODY_HEIGHTS_PER_SEC,
        }
    conflicting_timing = {item["side"] for item in timing + speed}
    if len(conflicting_timing) > 1:
        return {
            "status": "review_required_conflicting_priority",
            "awarded_side": None,
            "point_awarded": None,
            "reason": "Lamp-anchored speed and attack-extension timing disagree; review is required.",
            "priority_candidates": candidates,
        }
    all_candidate_sides = {item["side"] for item in candidates}
    if len(all_candidate_sides) > 1:
        return {
            "status": "review_required_conflicting_priority",
            "awarded_side": None,
            "point_awarded": None,
            "reason": "Available priority context points to different sides; review is required.",
            "priority_candidates": candidates,
        }
    return {
        "status": "review_required_double_light",
        "awarded_side": None,
        "point_awarded": None,
        "reason": "Double light lacks corroborated priority or confirmed simultaneous evidence.",
        "priority_candidates": candidates,
        "automatic_speed_margin_body_heights_per_sec": LAMP_SPEED_AUTOMATIC_MARGIN_BODY_HEIGHTS_PER_SEC,
    }


def build_mvp_payload(fused: dict) -> dict:
    events = []
    for event in fused.get("events", []):
        item = dict(event)
        verdict = adjudicate_event(item)
        item["verdict"] = verdict
        item["awarded_side"] = verdict["awarded_side"]
        events.append(item)
    return {
        "schema_version": "sabervision_mvp_1.0",
        "product_status": "offline_prototype_not_official_referee_system",
        "policy": {
            "single_light": "Any one clear automatic lamp source resolves the touch; no confidence/N-of-3 voting.",
            "double_light": "Movement, lunge and extension onset alone never decide priority. A very strong lamp-anchored speed advantage must exceed 1.70 body-heights/s in both 0.35s and 0.75s windows with the same direction; otherwise review.",
            "simultaneous": "A confirmed simultaneous double light produces no point for either side; absence of confirmation remains review.",
            "blade": "Visible guard/tip, contact and parry candidates are supporting evidence only at current model quality.",
            "safety": "Missing evidence is not converted into negative evidence.",
        },
        "source_availability": fused.get("source_availability"),
        "events": events,
        "summary": {
            "events": len(events),
            "automatic_awards": sum(
                event["verdict"]["awarded_side"] in SIDES for event in events
            ),
            "review_required": sum(
                event["verdict"]["point_awarded"] is None for event in events
            ),
            "simultaneous_no_point": sum(
                event["verdict"]["point_awarded"] is False for event in events
            ),
        },
    }


def write_csv(payload: dict, path: Path) -> None:
    rows = []
    for event in payload["events"]:
        verdict = event["verdict"]
        rows.append(
            {
                "event_id": event["event_id"],
                "start_sec": event["start_sec"],
                "end_sec": event["end_sec"],
                "touch_state": event.get("touch_state"),
                "touch_status": event.get("touch_status"),
                "verdict_status": verdict["status"],
                "awarded_side": verdict["awarded_side"],
                "point_awarded": verdict["point_awarded"],
                "verdict_reason": verdict["reason"],
                "priority_sources": ";".join(
                    item["source"] for item in verdict["priority_candidates"]
                ),
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [
            "event_id", "start_sec", "end_sec", "touch_state", "touch_status",
            "verdict_status", "awarded_side", "point_awarded", "verdict_reason", "priority_sources",
        ])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build conservative SaberVision MVP verdicts.")
    parser.add_argument("--action-summary", type=Path)
    parser.add_argument("--action-sequence-key")
    parser.add_argument("--pose-json", type=Path)
    parser.add_argument("--blade-json", type=Path)
    parser.add_argument("--blade-timing-json", type=Path)
    parser.add_argument("--device-final", type=Path, nargs="*", default=[])
    parser.add_argument("--device-review", type=Path, nargs="*", default=[])
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()

    action = read_json(args.action_summary) if args.action_summary else None
    blade = read_json(args.blade_json) if args.blade_json else None
    timing = read_json(args.blade_timing_json) if args.blade_timing_json else None
    device_events = []
    for path in args.device_final:
        device_events.extend(load_device_events(path, "automatic"))
    for path in args.device_review:
        device_events.extend(load_device_events(path, "review"))
    fused = fuse(
        action,
        args.action_sequence_key,
        blade,
        device_events,
        blade_timing_payload=timing,
    )
    pose_payload = read_json(args.pose_json) if args.pose_json else None
    for event in fused.get("events", []):
        action_context = event.setdefault("action", {})
        action_context["lamp_anchored_speed"] = lamp_anchored_speed_evidence(
            pose_payload, float(event["start_sec"])
        )
    payload = build_mvp_payload(fused)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_csv(payload, args.output_csv)
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
