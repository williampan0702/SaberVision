from __future__ import annotations

import argparse
import json
from pathlib import Path

from camera_stabilized_movement import stabilized_movement_features
from evaluate_action_rules import find_exchange_phase, load_json
from summarize_action_sequence_v7 import summarize_one as summarize_v7


CAMERA_MOVEMENT_THRESHOLD = 0.15
CAMERA_PAIR_MINIMUM = 0.10
CAMERA_PAIR_STRONG_MINIMUM = 0.18
LEGACY_STRONG_MOVEMENT = 0.25
MINIMUM_CAMERA_COVERAGE = 0.65
MINIMUM_CAMERA_INLIER_RATIO = 0.50

LOW_DYNAMIC_MAX_SPEED = 0.40
LOW_DYNAMIC_MAX_INWARD_NET = 0.35


def _number(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _movement_from_net(value: float, threshold: float) -> str:
    if value >= threshold:
        return "advance"
    if value <= -threshold:
        return "retreat"
    return "steady"


def _camera_is_reliable(camera: dict) -> bool:
    return bool(
        camera.get("status") == "ok"
        and _number(camera.get("transform_coverage"))
        >= MINIMUM_CAMERA_COVERAGE
        and _number(camera.get("median_inlier_ratio"))
        >= MINIMUM_CAMERA_INLIER_RATIO
    )


def apply_camera_stabilized_movement(
    sequence: dict, camera: dict
) -> None:
    reliable = _camera_is_reliable(camera)
    camera_nets = {
        side: _number(
            camera.get("athletes", {})
            .get(side, {})
            .get("inward_net_body_heights")
        )
        for side in ("left", "right")
    }
    proposed: dict[str, tuple[str, str]] = {}
    for side in ("left", "right"):
        athlete = sequence["athletes"][side]
        legacy_label = athlete.get("dominant_pre_contact_movement")
        legacy_net = _number(athlete.get("event_features", {}).get("inward_net"))
        athlete["legacy_pre_contact_movement"] = legacy_label
        athlete["camera_stabilized_movement"] = camera.get(
            "athletes", {}
        ).get(side)
        if athlete.get("action_status") != "ok" or not reliable:
            proposed[side] = (legacy_label, "legacy_unstabilized")
            continue
        # Preserve a strong, previously validated screen-space displacement.
        # This fallback matters when a fixed broadcast overlay dominates the
        # optical-flow registration even after masking.
        if legacy_net >= LEGACY_STRONG_MOVEMENT:
            proposed[side] = ("advance", "legacy_strong_displacement")
        elif legacy_net <= -LEGACY_STRONG_MOVEMENT:
            proposed[side] = ("retreat", "legacy_strong_displacement")
        else:
            proposed[side] = (
                _movement_from_net(
                    camera_nets[side], CAMERA_MOVEMENT_THRESHOLD
                ),
                "camera_stabilized_displacement",
            )

    # A small but consistent positive residual on both athletes is useful when
    # zoom makes each individual displacement fall just below the threshold.
    # Requiring both signs to agree avoids turning a static advance/retreat pair
    # into two advances.
    if (
        reliable
        and min(camera_nets.values()) >= CAMERA_PAIR_MINIMUM
        and max(camera_nets.values()) >= CAMERA_PAIR_STRONG_MINIMUM
        and all(
            _number(
                sequence["athletes"][side]
                .get("event_features", {})
                .get("inward_net")
            )
            > -LEGACY_STRONG_MOVEMENT
            for side in ("left", "right")
        )
    ):
        for side in ("left", "right"):
            if sequence["athletes"][side].get("action_status") == "ok":
                proposed[side] = (
                    "advance",
                    "camera_stabilized_bilateral_closure",
                )

    for side, (label, source) in proposed.items():
        athlete = sequence["athletes"][side]
        if athlete.get("action_status") != "ok":
            continue
        athlete["dominant_pre_contact_movement"] = label
        athlete["movement_source"] = source
    sequence["camera_motion"] = {
        key: value for key, value in camera.items() if key != "athletes"
    }


def _attack_strength(athlete: dict) -> float:
    event = athlete.get("event_features", {})
    return (
        _number(event.get("max_speed"))
        + 0.35 * max(0.0, _number(event.get("inward_net")))
        + 0.08 * int(athlete.get("geometry_candidate_frames") or 0)
    )


def _set_lunge_negative(athlete: dict, state: str, source: str) -> None:
    athlete["lunge_present"] = False
    athlete["lunge_evidence_status"] = state
    athlete["lunge_source"] = source
    athlete["requires_downstream_lunge_resolution"] = False


def apply_v4_box_lunge_guards(sequence: dict) -> None:
    # Full-body v4 crops make static fencing guards geometrically cleaner.  A
    # posture is not a lunge unless it is accompanied by enough inward motion.
    for athlete in sequence["athletes"].values():
        if (
            athlete.get("action_status") == "ok"
            and athlete.get("lunge_evidence_status")
            == "confirmed_geometry"
        ):
            event = athlete.get("event_features", {})
            window = athlete.get("temporal_window_evidence", {}).get(
                "best_window"
            ) or {}
            maxima = window.get("maxima") or {}
            minima = window.get("minima") or {}
            single_frame_strong_posture = bool(
                int(athlete.get("geometry_candidate_frames") or 0) == 1
                and _number(maxima.get("stance")) >= 0.70
                and _number(maxima.get("weapon_elbow")) >= 160.0
                and _number(minima.get("front_knee"), 180.0) <= 135.0
            )
            if (
                not bool(athlete.get("explosive_attack_event", False))
                and _number(event.get("max_speed")) < LOW_DYNAMIC_MAX_SPEED
                and _number(event.get("inward_net"))
                < LOW_DYNAMIC_MAX_INWARD_NET
                and not single_frame_strong_posture
            ):
                _set_lunge_negative(
                    athlete,
                    "low_dynamic_posture_negative",
                    "geometry_without_sufficient_inward_motion",
                )

    left = sequence["athletes"]["left"]
    right = sequence["athletes"]["right"]

    # If a one-frame, shallow response is detected opposite a much stronger
    # multi-frame attack, keep the strong attack and suppress the response.
    if left.get("lunge_present") is True and right.get("lunge_present") is True:
        strengths = {"left": _attack_strength(left), "right": _attack_strength(right)}
        strong_side = max(strengths, key=strengths.get)
        weak_side = "right" if strong_side == "left" else "left"
        strong = sequence["athletes"][strong_side]
        weak = sequence["athletes"][weak_side]
        weak_event = weak.get("event_features", {})
        if (
            strengths[strong_side] >= 2.5
            and strengths[strong_side] >= 2.0 * max(strengths[weak_side], 0.01)
            and int(strong.get("geometry_candidate_frames") or 0) >= 5
            and int(weak.get("geometry_candidate_frames") or 0) <= 1
            and _number(weak_event.get("max_stance")) < 0.70
        ):
            _set_lunge_negative(
                weak,
                "dominated_response_negative",
                "weak_single_frame_response_opposite_strong_attack",
            )

    # Rarely a one-frame base rule fires on the advancing athlete while the
    # opponent has the coherent four-frame lunge.  Resolve that pair using the
    # temporal leg-coherence evidence instead of accepting the noisier frame.
    for confirmed_side, supported_side in (("left", "right"), ("right", "left")):
        confirmed = sequence["athletes"][confirmed_side]
        supported = sequence["athletes"][supported_side]
        confirmed_window = confirmed.get("temporal_window_evidence", {}).get(
            "best_window"
        ) or {}
        supported_window = supported.get("temporal_window_evidence", {}).get(
            "best_window"
        ) or {}
        if (
            confirmed.get("lunge_present") is True
            and confirmed.get("lunge_evidence_status") == "confirmed_geometry"
            and not bool(confirmed_window.get("leg_coherent_in_one_frame"))
            and supported.get("lunge_evidence_status")
            == "supported_window_geometry"
            and bool(supported_window.get("leg_coherent_in_one_frame"))
            and bool(supported_window.get("supported"))
            and _number(confirmed.get("event_features", {}).get("inward_net"))
            >= 0.50
            and _number(supported.get("event_features", {}).get("inward_net"))
            >= 0.50
        ):
            _set_lunge_negative(
                confirmed,
                "pairwise_incoherent_step_negative",
                "single_frame_geometry_lost_pairwise_coherence_test",
            )
            supported["lunge_present"] = True
            supported["lunge_evidence_status"] = "pairwise_coherent_lunge"
            supported["lunge_source"] = "coherent_temporal_geometry_wins_pair"
            supported["requires_downstream_lunge_resolution"] = False
            break


def summarize_one(path: Path) -> dict:
    sequence = summarize_v7(path)
    data = load_json(path)
    start_frame, contact_frame = find_exchange_phase(data)
    camera = stabilized_movement_features(
        data,
        pose_json_path=path,
        start_frame=start_frame,
        contact_frame=contact_frame,
    )
    apply_camera_stabilized_movement(sequence, camera)
    apply_v4_box_lunge_guards(sequence)
    return sequence


def main() -> None:
    parser = argparse.ArgumentParser(
        description="v8 action summary with camera motion and v4 crop guards."
    )
    parser.add_argument("--pose-dir", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = []
    for pose_dir in args.pose_dir:
        paths.extend(sorted(pose_dir.glob("*.json")))
    sequences = {path.stem: summarize_one(path) for path in paths}
    evidence_counts: dict[str, int] = {}
    movement_source_counts: dict[str, int] = {}
    for sequence in sequences.values():
        for athlete in sequence["athletes"].values():
            evidence = athlete.get("lunge_evidence_status", "missing")
            evidence_counts[evidence] = evidence_counts.get(evidence, 0) + 1
            source = athlete.get("movement_source", "unresolved")
            movement_source_counts[source] = movement_source_counts.get(source, 0) + 1
    payload = {
        "schema_version": "8.0-candidate",
        "movement_policy": {
            "camera_threshold": CAMERA_MOVEMENT_THRESHOLD,
            "legacy_strong_threshold": LEGACY_STRONG_MOVEMENT,
            "minimum_camera_coverage": MINIMUM_CAMERA_COVERAGE,
            "minimum_camera_inlier_ratio": MINIMUM_CAMERA_INLIER_RATIO,
        },
        "lunge_policy": {
            "low_dynamic_max_speed": LOW_DYNAMIC_MAX_SPEED,
            "low_dynamic_max_inward_net": LOW_DYNAMIC_MAX_INWARD_NET,
            "pairwise_dominance_guard": True,
            "pairwise_temporal_coherence_resolution": True,
        },
        "movement_source_counts": movement_source_counts,
        "lunge_evidence_counts": evidence_counts,
        "sequences": sequences,
        "limitations": [
            "Reference action labels are AI visual engineering labels, not expert referee ground truth.",
            "Camera compensation estimates 2-D background motion and does not reconstruct full 3-D piste motion.",
            "A final fencing decision must still combine synchronized weapon and scorebox evidence.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "sequences": len(sequences),
                "movement_source_counts": movement_source_counts,
                "lunge_evidence_counts": evidence_counts,
                "output": str(args.output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
