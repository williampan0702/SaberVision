from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


WORKSPACE = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = Path(r"D:\SaberVision_data\SaberVision_lamp_state_training_v1\stage44_lit_lamp_candidates_v43")
KNOWN_ROOT = Path(r"D:\SaberVision_data\SaberVision_training_sources\cross_venue_v4\known_event_clips")
SINT_ROOT = Path(r"D:\SaberVision_data\SaberVision_training_sources\single_lamp_supplement\verified_fullres_clips")
BLIND_ROOT = Path(r"D:\SaberVision_data\SaberVision_training_sources\blind_holdout_v4_1\sources")
STAGE1_MANIFEST = Path(r"D:\SaberVision_data\SaberVision_referee_dataset_stage1_v1\stage1_final_v3\training_pool_manifest.csv")
SEED = 20260827


@dataclass(frozen=True)
class Event:
    name: str
    venue: str
    source: Path
    start: float
    end: float
    active: tuple[str, ...]
    # Tight normalized rectangles around each physical emitter. Red and green
    # are separate so score digits inside the same device are never positives.
    red_rois: tuple[tuple[float, float, float, float], ...]
    green_rois: tuple[tuple[float, float, float, float], ...]
    split: str


EVENTS = (
    Event("madrid_left", "madrid_spain", KNOWN_ROOT / "madrid_spain_left_0464p895.mp4", 1.70, 2.42,
          ("red_lamp",), ((0.660, 0.345, 0.715, 0.405),), (), "train"),
    Event("madrid_double", "madrid_spain", KNOWN_ROOT / "madrid_spain_double_0210p976.mp4", 0.50, 2.50,
          ("red_lamp", "green_lamp"), ((0.490, 0.345, 0.525, 0.410),),
          ((0.515, 0.345, 0.555, 0.410),), "train"),
    Event("lima_left", "lima_peru", KNOWN_ROOT / "lima_peru_left_0310p745.mp4", 0.50, 2.00,
          ("red_lamp",), ((0.350, 0.345, 0.455, 0.405), (0.410, 0.810, 0.455, 0.880)),
          (), "train"),
    Event("lima_right", "lima_peru", KNOWN_ROOT / "lima_peru_right_1654p043.mp4", 2.25, 2.50,
          ("green_lamp",), (), ((0.500, 0.345, 0.620, 0.405), (0.495, 0.810, 0.540, 0.880)),
          "train"),
    Event("lima_double", "lima_peru", KNOWN_ROOT / "lima_peru_double_2901p916.mp4", 1.25, 2.50,
          ("red_lamp", "green_lamp"), ((0.350, 0.345, 0.455, 0.405), (0.410, 0.810, 0.455, 0.880)),
          ((0.500, 0.345, 0.620, 0.405), (0.495, 0.810, 0.540, 0.880)), "train"),
    Event("genova_right", "genova_italy", KNOWN_ROOT / "genova_italy_right_0567p200.mp4", 1.70, 2.48,
          ("green_lamp",), (), ((0.070, 0.300, 0.120, 0.405), (0.535, 0.325, 0.620, 0.425),
                                 (0.970, 0.295, 1.000, 0.410)), "val"),
    # The verified Sint-Niklaas clips are already tight event cuts. Candidate
    # color must still be present in the physical top lamp bar to count.
    Event("sint_left_0026", "sint_niklaas_belgium", SINT_ROOT / "sint_left_0026p6.mp4", 0.0, 9.0,
          ("red_lamp",), ((0.400, 0.250, 0.490, 0.305),), (), "train"),
    Event("sint_left_0162", "sint_niklaas_belgium", SINT_ROOT / "sint_left_0162p4.mp4", 0.0, 9.0,
          ("red_lamp",), ((0.400, 0.250, 0.490, 0.305),), (), "train"),
    Event("sint_left_0188", "sint_niklaas_belgium", SINT_ROOT / "sint_left_0188p2.mp4", 0.0, 9.0,
          ("red_lamp",), ((0.400, 0.250, 0.490, 0.305),), (), "train"),
    Event("sint_left_0200", "sint_niklaas_belgium", SINT_ROOT / "sint_left_0200p2.mp4", 0.0, 9.0,
          ("red_lamp",), ((0.400, 0.250, 0.490, 0.305),), (), "train"),
    Event("sint_right_0015", "sint_niklaas_belgium", SINT_ROOT / "sint_right_0015p2.mp4", 0.0, 9.0,
          ("green_lamp",), (), ((0.825, 0.245, 0.920, 0.305),), "train"),
    Event("sint_right_0174", "sint_niklaas_belgium", SINT_ROOT / "sint_right_0174p6.mp4", 0.0, 9.0,
          ("green_lamp",), (), ((0.595, 0.245, 0.680, 0.305),), "train"),
    Event("sint_right_0308", "sint_niklaas_belgium", SINT_ROOT / "sint_right_0308p8.mp4", 0.0, 9.0,
          ("green_lamp",), (), ((0.610, 0.245, 0.690, 0.305),), "train"),
    Event("bali_double_274", "bali_2025", BLIND_ROOT / "bali_2025_blind_10m00s_15m00s.mp4", 274.10, 275.10,
          ("red_lamp", "green_lamp"), ((0.405, 0.355, 0.455, 0.500),),
          ((0.495, 0.355, 0.545, 0.500),), "train"),
    Event("bali_double_287", "bali_2025", BLIND_ROOT / "bali_2025_blind_10m00s_15m00s.mp4", 287.10, 288.80,
          ("red_lamp", "green_lamp"), ((0.405, 0.355, 0.455, 0.500),),
          ((0.495, 0.355, 0.545, 0.500),), "train"),
    Event("wuxi_double_080", "wuxi_2025", BLIND_ROOT / "wuxi_2025_blind_04m00s_09m00s.mp4", 79.45, 80.90,
          ("red_lamp", "green_lamp"),
          ((0.000, 0.245, 0.035, 0.325), (0.075, 0.245, 0.145, 0.325),
           (0.205, 0.245, 0.275, 0.325), (0.335, 0.245, 0.405, 0.325)),
          ((0.475, 0.245, 0.550, 0.325), (0.605, 0.245, 0.680, 0.325),
           (0.735, 0.245, 0.815, 0.325), (0.865, 0.245, 0.955, 0.325)), "val"),
)


def read_image(path: Path) -> np.ndarray | None:
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None


def write_jpeg(path: Path, image: np.ndarray, quality: int = 94) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError(f"Could not encode {path}")
    encoded.tofile(str(path))


def color_mask(image: np.ndarray, label: str) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    if label == "red_lamp":
        mask = (((h <= 12) | (h >= 168)) & (s >= 75) & (v >= 135)).astype(np.uint8) * 255
    elif label == "green_lamp":
        mask = ((h >= 34) & (h <= 98) & (s >= 58) & (v >= 125)).astype(np.uint8) * 255
    else:
        raise ValueError(label)
    kernel = np.ones((3, 3), np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def components(image: np.ndarray, label: str) -> list[tuple[int, int, int, int, int]]:
    mask = color_mask(image, label)
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    h, w = image.shape[:2]
    result = []
    for x, y, bw, bh, area in stats[1:]:
        if area < max(4, round(w * h * 0.000006)):
            continue
        if bw < 2 or bh < 2 or bw > w * 0.42 or bh > h * 0.30:
            continue
        fill = area / max(bw * bh, 1)
        if fill < 0.10:
            continue
        result.append((int(x), int(y), int(x + bw), int(y + bh), int(area)))
    return result


def inside_rois(box: tuple[int, int, int, int, int], rois, width: int, height: int) -> bool:
    x1, y1, x2, y2, _ = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    return any(rx1 * width <= cx <= rx2 * width and ry1 * height <= cy <= ry2 * height
               for rx1, ry1, rx2, ry2 in rois)


def roi_candidate(image: np.ndarray, label: str, roi) -> tuple[int, int, int, int, int] | None:
    """Return one color component constrained to one manually reviewed emitter."""
    h, w = image.shape[:2]
    rx1, ry1, rx2, ry2 = roi
    x1, y1 = max(0, round(rx1 * w)), max(0, round(ry1 * h))
    x2, y2 = min(w, round(rx2 * w)), min(h, round(ry2 * h))
    if x2 <= x1 or y2 <= y1:
        return None
    mask = color_mask(image[y1:y2, x1:x2], label)
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if count <= 1:
        return None
    component = max(stats[1:], key=lambda item: item[cv2.CC_STAT_AREA])
    cx, cy, bw, bh, area = map(int, component)
    if area < max(5, round(w * h * 0.000006)) or bw < 2 or bh < 2:
        return None
    return x1 + cx, y1 + cy, x1 + cx + bw, y1 + cy + bh, area


def candidate_crop(image: np.ndarray, box: tuple[int, int, int, int, int], size: int = 128) -> np.ndarray:
    x1, y1, x2, y2, _ = box
    h, w = image.shape[:2]
    bw, bh = x2 - x1, y2 - y1
    radius = max(18, int(max(bw, bh) * 1.65))
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    sx1, sy1 = max(0, cx - radius), max(0, cy - radius)
    sx2, sy2 = min(w, cx + radius), min(h, cy + radius)
    crop = image[sy1:sy2, sx1:sx2]
    scale = size / max(crop.shape[:2])
    nw, nh = max(1, round(crop.shape[1] * scale)), max(1, round(crop.shape[0] * scale))
    resized = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
    canvas = np.zeros((size, size, 3), np.uint8)
    ox, oy = (size - nw) // 2, (size - nh) // 2
    canvas[oy:oy + nh, ox:ox + nw] = resized
    return canvas


def iter_video_frames(event: Event, sample_fps: float = 8.0):
    cap = cv2.VideoCapture(str(event.source))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {event.source}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
    start, end = max(0.0, event.start), min(event.end, duration)
    cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000.0)
    step = 1.0 / sample_fps
    timestamp = start
    while timestamp <= end + 1e-6:
        cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000.0)
        ok, frame = cap.read()
        if ok:
            yield timestamp, frame
        timestamp += step
    cap.release()


def save_candidate(output: Path, split: str, label: str, venue: str, event: str,
                   timestamp: float, serial: int, image: np.ndarray, box, source: Path,
                   review_status: str) -> dict:
    token = hashlib.sha1(f"{source}|{timestamp:.4f}|{label}|{box}".encode()).hexdigest()[:12]
    name = f"{venue}__{event}__t{timestamp:08.3f}__{serial:02d}__{token}.jpg".replace(".", "p", 1)
    target = output / split / label / name
    write_jpeg(target, candidate_crop(image, box))
    x1, y1, x2, y2, area = box
    return {
        "split": split, "label": label, "venue": venue, "event": event,
        "timestamp_sec": f"{timestamp:.3f}", "source": str(source), "crop_path": str(target),
        "x1": x1, "y1": y1, "x2": x2, "y2": y2, "mask_area": area,
        "review_status": review_status,
    }


def event_candidates(output: Path) -> list[dict]:
    rows = []
    for event in EVENTS:
        if not event.source.exists():
            raise FileNotFoundError(event.source)
        for timestamp, frame in iter_video_frames(event):
            h, w = frame.shape[:2]
            for color in ("red_lamp", "green_lamp"):
                found = components(frame, color)
                rois = event.red_rois if color == "red_lamp" else event.green_rois
                positives = []
                if color in event.active:
                    positives = [box for roi in rois if (box := roi_candidate(frame, color, roi)) is not None]
                # Candidate components inside a reviewed active emitter are
                # represented by the positive crop above, not duplicated as negatives.
                negatives = [b for b in found if not (color in event.active and inside_rois(b, rois, w, h))]
                # Keep all reviewed physical-lamp candidates, but cap incidental
                # graphics per frame so a single broadcast does not dominate.
                for serial, box in enumerate(positives):
                    rows.append(save_candidate(output, event.split, color, event.venue, event.name,
                                               timestamp, serial, frame, box, event.source,
                                               "roi_and_state_verified"))
                negatives.sort(key=lambda b: b[4], reverse=True)
                for serial, box in enumerate(negatives[:5]):
                    rows.append(save_candidate(output, event.split, "background", event.venue, event.name,
                                               timestamp, serial, frame, box, event.source,
                                               "outside_physical_lamp_roi"))
    return rows


def hard_negative_candidates(output: Path, limit_per_venue: int = 90) -> list[dict]:
    if not STAGE1_MANIFEST.exists():
        return []
    with STAGE1_MANIFEST.open("r", encoding="utf-8-sig", newline="") as handle:
        source_rows = [r for r in csv.DictReader(handle) if r["selected_role"] in {"hard_negative", "transition"}]
    rng = random.Random(SEED)
    by_venue = defaultdict(list)
    for row in source_rows:
        by_venue[row["venue"]].append(row)
    rows = []
    for venue, items in sorted(by_venue.items()):
        if venue in {"genova_italy", "wuxi_2025"}:
            continue
        rng.shuffle(items)
        kept = 0
        for source_row in items:
            image = read_image(Path(source_row["output_frame"]))
            if image is None:
                continue
            proposals = components(image, "red_lamp") + components(image, "green_lamp")
            rng.shuffle(proposals)
            for serial, box in enumerate(proposals[:6]):
                rows.append(save_candidate(output, "train", "background", venue, "stage1_hardneg",
                                           float(source_row["timestamp_sec"]), serial, image, box,
                                           Path(source_row["output_frame"]), "stage1_hard_negative"))
                kept += 1
                if kept >= limit_per_venue:
                    break
            if kept >= limit_per_venue:
                break
    return rows


def write_qa(output: Path, rows: list[dict]) -> None:
    rng = random.Random(SEED)
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["split"], row["label"])].append(row)
    for (split, label), items in grouped.items():
        rng.shuffle(items)
        items = items[:64]
        tile, columns = 144, 8
        canvas = np.zeros((((len(items) + columns - 1) // columns) * tile, columns * tile, 3), np.uint8)
        for idx, row in enumerate(items):
            image = read_image(Path(row["crop_path"]))
            image = cv2.resize(image, (tile, tile), interpolation=cv2.INTER_AREA)
            cv2.rectangle(image, (0, 0), (tile, 20), (0, 0, 0), -1)
            cv2.putText(image, row["venue"][:12], (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.34,
                        (255, 255, 255), 1, cv2.LINE_AA)
            y, x = divmod(idx, columns)
            canvas[y * tile:(y + 1) * tile, x * tile:(x + 1) * tile] = image
        write_jpeg(output / "qa" / f"{split}__{label}.jpg", canvas, 91)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build v4.3 physical lit-lamp candidate classifier data.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise RuntimeError(f"Output must be new or empty: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    rows = event_candidates(args.output)
    rows.extend(hard_negative_candidates(args.output))
    fields = list(rows[0])
    with (args.output / "manifest.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    write_qa(args.output, rows)
    counts = Counter((r["split"], r["label"], r["venue"]) for r in rows)
    summary = {
        "total_candidates": len(rows),
        "split_rule": "venue isolated: train=madrid/lima/sint/bali plus training hard negatives; val=genova/wuxi",
        "classes": ["background", "green_lamp", "red_lamp"],
        "counts": {"/".join(key): value for key, value in sorted(counts.items())},
        "diagnostic_note": "Bali and Wuxi were already used for v4.2 failure analysis and are no longer final blind holdouts.",
    }
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
