from __future__ import annotations

import argparse
import bisect
import csv
import json
from pathlib import Path

import cv2
import numpy as np


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def load_confirmed_track(path: Path):
    rows = []
    for row in read_csv(path):
        if row.get("valid") != "1" or not row.get("x1"):
            continue
        rows.append(
            (
                float(row["time_sec"]),
                tuple(round(float(row[key])) for key in ("x1", "y1", "x2", "y2")),
            )
        )
    rows.sort()
    return rows, [row[0] for row in rows]


def nearest_confirmed_box(track, times, timestamp: float, max_gap: float = 0.35):
    index = bisect.bisect_left(times, timestamp)
    candidates = []
    if index < len(track):
        candidates.append(track[index])
    if index > 0:
        candidates.append(track[index - 1])
    if not candidates:
        return None
    nearest = min(candidates, key=lambda item: abs(item[0] - timestamp))
    return nearest[1] if abs(nearest[0] - timestamp) <= max_gap else None


def measure(frame: np.ndarray, box: tuple[int, int, int, int]) -> dict[str, float]:
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = box
    x1, x2 = max(0, x1), min(width, x2)
    y1, y2 = max(0, y1), min(height, y2)
    if x2 <= x1 or y2 <= y1:
        return {"red_fraction": 0.0, "green_fraction": 0.0, "dark_fraction": 0.0, "lamp_row": 0.0}
    crop = frame[y1:y2, x1:x2]
    crop_h, crop_w = crop.shape[:2]
    band_y1 = max(0, round(crop_h * 0.20))
    band_y2 = min(crop_h, round(crop_h * 0.60))
    split = max(1, round(crop_w * 0.50))
    left_band = crop[band_y1:band_y2, :split]
    right_band = crop[band_y1:band_y2, split:]

    def color_fraction(region: np.ndarray, color: str) -> float:
        if region.size == 0:
            return 0.0
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        hue, saturation, value = cv2.split(hsv)
        vivid = (saturation >= 105) & (value >= 125)
        if color == "red":
            mask = vivid & ((hue <= 12) | (hue >= 168))
        else:
            mask = vivid & (hue >= 36) & (hue <= 88)
        return float(mask.mean())

    def filled_lamp_blob_fraction(region: np.ndarray, color: str) -> float:
        """Return the strongest compact lamp block, excluding score digits."""
        if region.size == 0:
            return 0.0
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        hue, saturation, value = cv2.split(hsv)
        vivid = (saturation >= 105) & (value >= 125)
        if color == "red":
            mask = vivid & ((hue <= 12) | (hue >= 168))
        else:
            mask = vivid & (hue >= 36) & (hue <= 88)
        count, _labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask.astype(np.uint8), 8
        )
        region_height, region_width = region.shape[:2]
        region_area = max(1, region_height * region_width)
        best = 0.0
        for index in range(1, count):
            _x, _y, blob_width, blob_height, area = stats[index]
            area_fraction = float(area) / region_area
            width_fraction = float(blob_width) / max(1, region_width)
            height_fraction = float(blob_height) / max(1, region_height)
            fill = float(area) / max(1, blob_width * blob_height)
            center_y = float(centroids[index][1]) / max(1, region_height)
            # The lamp block is wide, compact and above the large score
            # digits.  The red timer digits can be similarly bright, so the
            # red block uses a tighter upper-position rule.
            maximum_center_y = 0.24 if color == "red" else 0.72
            minimum_area = 0.010 if color == "red" else 0.008
            minimum_width = 0.15 if color == "red" else 0.10
            if (
                area_fraction >= minimum_area
                and width_fraction >= minimum_width
                and height_fraction >= 0.14
                and fill >= 0.26
                and center_y <= maximum_center_y
            ):
                best = max(best, area_fraction)
        return best

    # The scoring unit is a compact black cabinet.  This rejects a held or
    # mis-inferred box that lands on a silver fencer during a camera pan.
    cabinet = crop[round(crop_h * 0.15) : round(crop_h * 0.95)]
    gray = cv2.cvtColor(cabinet, cv2.COLOR_BGR2GRAY)
    dark_fraction = float((gray <= 92).mean()) if gray.size else 0.0

    # The dual-scoreboard locator deliberately keeps a tall centre box so it
    # survives camera zooms.  In modern broadcasts the physical scoring unit
    # often sits in the lower half of that box, below the older 20%-60% lamp
    # band.  Measure a second lower band instead of enlarging the locator or
    # relaxing the colour thresholds.  The local dark-cabinet test keeps this
    # recovery selective when a fencer crosses the centre.
    # Keep this above the large score digits at the very bottom.  The true
    # red/green lamp blocks sit alongside the timer in roughly this slice.
    lower_y1 = max(0, round(crop_h * 0.50))
    lower_y2 = crop_h
    lower_band = crop[lower_y1:lower_y2]
    lower_split = max(1, round(crop_w * 0.50))
    lower_left_band = lower_band[:, :lower_split]
    lower_right_band = lower_band[:, lower_split:]
    lower_gray = cv2.cvtColor(lower_band, cv2.COLOR_BGR2GRAY) if lower_band.size else np.empty((0, 0), dtype=np.uint8)
    lower_dark_fraction = float((lower_gray <= 92).mean()) if lower_gray.size else 0.0
    dark_fraction = max(dark_fraction, lower_dark_fraction)
    full_band = crop[band_y1:band_y2]
    band_h, band_w = full_band.shape[:2]
    band_hsv = cv2.cvtColor(full_band, cv2.COLOR_BGR2HSV)
    band_hue, band_sat, band_value = cv2.split(band_hsv)
    coloured_lamp = (band_sat >= 90) & (band_value >= 130) & (
        (band_hue <= 12) | (band_hue >= 168) | ((band_hue >= 36) & (band_hue <= 88))
    )
    white_lamp = (band_sat <= 90) & (band_value >= 165)
    bright_mask = ((coloured_lamp | white_lamp).astype(np.uint8) * 255)
    count, _labels, stats, centroids = cv2.connectedComponentsWithStats(bright_mask, 8)
    components = []
    minimum_area = max(3, round(band_h * band_w * 0.002))
    maximum_area = max(minimum_area + 1, round(band_h * band_w * 0.18))
    for index in range(1, count):
        _x, _y, component_w, component_h, area = stats[index]
        if not minimum_area <= area <= maximum_area:
            continue
        if component_w < band_w * 0.035 or component_h < band_h * 0.06:
            continue
        components.append(
            {
                "x": float(_x),
                "y": float(_y),
                "w": float(component_w),
                "h": float(component_h),
                "area": float(area),
                "cx": float(centroids[index][0]),
                "cy": float(centroids[index][1]),
            }
        )
    best_cluster = []
    best_cluster_score = float("-inf")
    for component in components:
        aligned = [
            other
            for other in components
            if abs(other["cy"] - component["cy"]) <= band_h * 0.14
        ]
        aligned.sort(key=lambda item: item["cx"])
        if len(aligned) < 3:
            continue
        # The physical unit has four similarly sized horizontal lamps.  Score
        # clusters by four-lamp likeness, horizontal span and area consistency;
        # this is more selective than merely counting bright background text.
        for window_size in range(3, min(6, len(aligned)) + 1):
            for start_index in range(0, len(aligned) - window_size + 1):
                cluster = aligned[start_index : start_index + window_size]
                span = max(item["x"] + item["w"] for item in cluster) - min(item["x"] for item in cluster)
                span_ratio = span / max(1.0, band_w)
                if not 0.34 <= span_ratio <= 1.02:
                    continue
                areas = np.array([item["area"] for item in cluster], dtype=np.float32)
                area_consistency = float(areas.min() / max(1.0, areas.max()))
                count_score = 2.5 - abs(window_size - 4) * 0.65
                score = count_score + span_ratio * 2.0 + area_consistency * 1.5
                if score > best_cluster_score:
                    best_cluster_score = score
                    best_cluster = cluster

    best_aligned = len(best_cluster)
    row_red_fraction = 0.0
    row_green_fraction = 0.0
    row_span_fraction = 0.0
    if best_cluster:
        row_x1 = max(0, round(min(item["x"] for item in best_cluster) - band_w * 0.025))
        row_x2 = min(band_w, round(max(item["x"] + item["w"] for item in best_cluster) + band_w * 0.025))
        row_y1 = max(0, round(min(item["y"] for item in best_cluster) - band_h * 0.05))
        row_y2 = min(band_h, round(max(item["y"] + item["h"] for item in best_cluster) + band_h * 0.05))
        row_mid_x = round((row_x1 + row_x2) * 0.5)
        row_red_fraction = color_fraction(full_band[row_y1:row_y2, row_x1:row_mid_x], "red")
        row_green_fraction = color_fraction(full_band[row_y1:row_y2, row_mid_x:row_x2], "green")
        row_span_fraction = (row_x2 - row_x1) / max(1.0, band_w)
    return {
        "red_fraction": row_red_fraction,
        "green_fraction": row_green_fraction,
        "broad_red_fraction": color_fraction(left_band, "red"),
        "broad_green_fraction": color_fraction(right_band, "green"),
        "lower_red_lamp_blob": filled_lamp_blob_fraction(lower_left_band, "red"),
        "lower_green_lamp_blob": filled_lamp_blob_fraction(lower_right_band, "green"),
        "lower_green_fraction": color_fraction(lower_right_band, "green"),
        "dark_fraction": dark_fraction,
        "lower_dark_fraction": lower_dark_fraction,
        "lamp_row": float(best_aligned),
        "lamp_row_span": row_span_fraction,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate dual-scoreboard lamp events using cabinet and true lamp colours.")
    parser.add_argument("input", type=Path)
    parser.add_argument("device_track", type=Path)
    parser.add_argument("candidate_events", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--red-threshold", type=float, default=0.025)
    parser.add_argument("--green-threshold", type=float, default=0.018)
    parser.add_argument("--dark-threshold", type=float, default=0.16)
    parser.add_argument("--analysis-fps", type=float, default=15.0)
    parser.add_argument("--required-lamp-components", type=int, default=3)
    parser.add_argument("--occluded-model-threshold", type=float, default=0.09)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    track, times = load_confirmed_track(args.device_track)
    events = read_csv(args.candidate_events)
    cap = cv2.VideoCapture(str(args.input))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {args.input}")

    audit_rows = []
    accepted = []
    for event in events:
        start = float(event["start_sec"])
        end = float(event["end_sec"])
        samples = []
        timestamp = start
        while timestamp <= end + 1e-6:
            box = nearest_confirmed_box(track, times, timestamp)
            if box is None:
                timestamp += 1.0 / args.analysis_fps
                continue
            cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000.0)
            ok, frame = cap.read()
            if not ok:
                timestamp += 1.0 / args.analysis_fps
                continue
            samples.append((timestamp, measure(frame, box)))
            timestamp += 1.0 / args.analysis_fps
        structured_samples = [
            sample for sample in samples
            if sample[1]["lamp_row"] >= args.required_lamp_components
        ]
        peak_red = max((sample[1]["red_fraction"] for sample in structured_samples), default=0.0)
        peak_green = max((sample[1]["green_fraction"] for sample in structured_samples), default=0.0)
        peak_dark = max((sample[1]["dark_fraction"] for sample in samples), default=0.0)
        peak_lamp_row = max((sample[1]["lamp_row"] for sample in samples), default=0.0)
        red_on = peak_red >= args.red_threshold
        green_on = peak_green >= args.green_threshold
        cabinet_visible = (
            peak_dark >= args.dark_threshold
            and peak_lamp_row >= args.required_lamp_components
        )
        state = "double" if red_on and green_on else "left" if red_on else "right" if green_on else "none"
        candidate_peak = float(event.get("peak_score") or 0.0)
        if cabinet_visible and state != "none":
            decision = "accepted_color"
        elif cabinet_visible and candidate_peak >= args.occluded_model_threshold:
            # The lamp row is still physically present, but a passing fencer can
            # hide the coloured pixels for most of a short event.  In that case
            # only a strong temporal YOLO onset may preserve the candidate state.
            state = event["state"]
            decision = "accepted_model_fallback"
        else:
            decision = "blocked"
        audit_rows.append(
            {
                "candidate_start_sec": event["start_sec"],
                "candidate_end_sec": event["end_sec"],
                "candidate_state": event["state"],
                "validated_state": state,
                "confirmed_samples": len(samples),
                "structured_samples": len(structured_samples),
                "peak_red_fraction": f"{peak_red:.6f}",
                "peak_green_fraction": f"{peak_green:.6f}",
                "peak_dark_fraction": f"{peak_dark:.6f}",
                "peak_lamp_components": round(peak_lamp_row),
                "decision": decision,
            }
        )
        if decision.startswith("accepted"):
            accepted.append({**event, "state": state})
    cap.release()

    audit_fields = list(audit_rows[0]) if audit_rows else [
        "candidate_start_sec", "candidate_end_sec", "candidate_state", "validated_state",
        "confirmed_samples", "structured_samples", "peak_red_fraction", "peak_green_fraction",
        "peak_dark_fraction", "peak_lamp_components", "decision",
    ]
    with (args.output / "event_validation_audit.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=audit_fields)
        writer.writeheader()
        writer.writerows(audit_rows)
    event_fields = ["state", "start_sec", "end_sec", "duration_sec", "peak_score"]
    with (args.output / "validated_events.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=event_fields)
        writer.writeheader()
        writer.writerows(accepted)
    summary = {
        "candidate_events": len(events),
        "validated_events": len(accepted),
        "blocked_events": len(events) - len(accepted),
        "thresholds": {
            "red_fraction": args.red_threshold,
            "green_fraction": args.green_threshold,
            "dark_fraction": args.dark_threshold,
            "required_lamp_components": args.required_lamp_components,
            "occluded_model_threshold": args.occluded_model_threshold,
        },
    }
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"summary": summary, "audit": audit_rows}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
