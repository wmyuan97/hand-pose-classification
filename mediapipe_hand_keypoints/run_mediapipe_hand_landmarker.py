"""Detect up to two hands per image with MediaPipe Hand Landmarker.

This script intentionally uses only the local .task model declared in the
Configuration section.  Download that model once before running this script;
the script never downloads or updates model assets itself.
"""

from __future__ import annotations

import logging
import sys
from collections import Counter, defaultdict
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from tqdm import tqdm


# =============================================================================
# Configuration
# =============================================================================
INPUT_ROOT = Path(r"D:\uottawa\phd\SHUEE\MV_gloria_data")
OUTPUT_ROOT = INPUT_ROOT / "mediapipe_hand_keypoints"
EXCEL_OUTPUT = OUTPUT_ROOT / "mediapipe_hand_keypoints.xlsx"
OVERLAY_ROOT = OUTPUT_ROOT / "mediapipe_hand_overlay"
DETECTION_LOG_OUTPUT = OUTPUT_ROOT / "mediapipe_detection_log.csv"
MODEL_PATH = OUTPUT_ROOT / "models" / "hand_landmarker.task"

# MediaPipe Hand Landmarker parameters. Thresholds are fixed when a task is
# created, so the two task objects below are each created once and reused for
# all images; no model is recreated inside the per-image loop.
NUM_HANDS = 2
INITIAL_HAND_DETECTION_CONFIDENCE = 0.50
INITIAL_HAND_PRESENCE_CONFIDENCE = 0.50
INITIAL_TRACKING_CONFIDENCE = 0.50
RETRY_HAND_DETECTION_CONFIDENCE = 0.30
RETRY_HAND_PRESENCE_CONFIDENCE = 0.30
RETRY_TRACKING_CONFIDENCE = 0.30

# A retry is triggered only when the initial pass has no geometrically valid
# hand near image centre. Crop fractions are finite, centred, and are slices
# of the original raster: no resize or aspect-ratio distortion is performed.
CENTER_MAX_DISTANCE_NORMALIZED = 0.42
CENTER_CROP_FRACTIONS = (0.80, 0.90)
KEEP_VALID_NON_TARGET_HANDS = True

# Automatic geometry checks. These are deliberately exposed so they can be
# calibrated for a later study without modifying the detection algorithm.
PALM_CENTER_INDICES = (0, 5, 9, 13, 17)
OUT_OF_BOUNDS_TOLERANCE_NORMALIZED = 0.02
MIN_IN_FRAME_KEYPOINT_FRACTION = 0.90
MIN_HAND_BBOX_DIAGONAL_NORMALIZED = 0.06
MAX_HAND_BBOX_DIAGONAL_NORMALIZED = 1.35
MIN_PALM_SPREAD_NORMALIZED = 0.025
MIN_MEDIAN_BONE_LENGTH_NORMALIZED = 0.006
MIN_BONE_TO_MEDIAN_RATIO = 0.03
MAX_BONE_TO_MEDIAN_RATIO = 12.0

# MediaPipe's documented handedness convention assumes mirrored (selfie) input.
# The study images are ordinary, non-mirrored camera photographs, so its labels
# are swapped to report anatomical left/right.  Set True only for mirrored input.
INPUT_IMAGE_IS_MIRRORED = False

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
LEFT_COLOR_BGR = (255, 80, 0)       # blue in BGR
RIGHT_COLOR_BGR = (0, 90, 255)      # red in BGR
LANDMARK_RADIUS = 3
CONNECTION_THICKNESS = 2
DRAW_LANDMARK_INDICES = True
INDEX_FONT_SCALE = 0.35
INDEX_FONT_THICKNESS = 1
INDEX_TEXT_OFFSET = (4, -4)


# Official MediaPipe hand landmark connection topology (indices 0--20).
HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
)


@dataclass
class HandCandidate:
    """One MediaPipe hand mapped into original-image pixel coordinates."""

    side: str
    score: float
    points: np.ndarray
    center_distance: float
    geometry_ok: bool
    center_ok: bool
    quality: float
    check_flags: tuple[str, ...]


@dataclass
class DetectionAttempt:
    """All usable candidates produced by one inference pass."""

    method: str
    candidates: list[HandCandidate]
    crop_fraction: float | None = None
    diagnostic_flags: tuple[str, ...] = ()


def configure_logging() -> None:
    """Send concise progress and duplicate-name warnings to the console."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def discover_images(input_root: Path) -> list[Path]:
    """Find images below immediate ``data_*`` folders and sort by name then path."""
    data_dirs = sorted(
        (path for path in input_root.iterdir() if path.is_dir() and path.name.casefold().startswith("data_")),
        key=lambda path: path.name.casefold(),
    )
    images: list[Path] = []
    for data_dir in data_dirs:
        images.extend(
            path
            for path in data_dir.rglob("*")
            if path.is_file() and path.suffix.casefold() in SUPPORTED_EXTENSIONS
        )

    images.sort(key=lambda path: (path.name.casefold(), str(path.resolve()).casefold()))

    duplicate_names = {
        filename: count
        for filename, count in Counter(path.name.casefold() for path in images).items()
        if count > 1
    }
    for filename, count in sorted(duplicate_names.items()):
        logging.warning("Duplicate filename retained %d times: %s", count, filename)
    return images


def make_columns() -> list[str]:
    """Return the fixed Excel schema (MediaPipe exposes no per-landmark score)."""
    columns = [
        "filename",
        "detection_status",
        "detection_method",
        "retry_count",
        "center_distance",
        "left_hand_score",
        "right_hand_score",
    ]
    for hand in ("left", "right"):
        for index in range(21):
            columns.extend((f"{hand}_kp{index}_x", f"{hand}_kp{index}_y"))
    return columns


def empty_record(filename: str, columns: list[str]) -> dict[str, Any]:
    """Create a row whose coordinate and score fields are numeric NaN values."""
    record: dict[str, Any] = {column: np.nan for column in columns}
    record["filename"] = filename
    return record


def model_handedness_to_anatomical(label: str) -> str | None:
    """Convert a MediaPipe handedness label to the requested anatomical side."""
    normalized = label.casefold()
    if normalized not in {"left", "right"}:
        return None
    if INPUT_IMAGE_IS_MIRRORED:
        return normalized
    return "right" if normalized == "left" else "left"


def landmark_to_pixel(landmark: Any, width: int, height: int) -> tuple[float, float]:
    """Convert normalized MediaPipe coordinates to original-image pixel coordinates."""
    return landmark.x * width, landmark.y * height


def draw_hand(image_bgr: np.ndarray, points: np.ndarray, side: str) -> None:
    """Draw one hand using the official topology without changing image dimensions."""
    color = LEFT_COLOR_BGR if side == "left" else RIGHT_COLOR_BGR

    for first, second in HAND_CONNECTIONS:
        x1, y1 = points[first]
        x2, y2 = points[second]
        if not np.isfinite((x1, y1, x2, y2)).all():
            continue
        cv2.line(
            image_bgr,
            (round(x1), round(y1)),
            (round(x2), round(y2)),
            color,
            CONNECTION_THICKNESS,
            cv2.LINE_AA,
        )
    for index, (x, y) in enumerate(points):
        if not np.isfinite((x, y)).all():
            continue
        cv2.circle(image_bgr, (round(x), round(y)), LANDMARK_RADIUS, color, -1, cv2.LINE_AA)
        if DRAW_LANDMARK_INDICES:
            cv2.putText(
                image_bgr,
                str(index),
                (round(x) + INDEX_TEXT_OFFSET[0], round(y) + INDEX_TEXT_OFFSET[1]),
                cv2.FONT_HERSHEY_SIMPLEX,
                INDEX_FONT_SCALE,
                color,
                INDEX_FONT_THICKNESS,
                cv2.LINE_AA,
            )


def write_overlay(image_bgr: np.ndarray, input_path: Path, input_root: Path) -> None:
    """Save the overlay below OUTPUT_ROOT, preserving the source subfolder layout."""
    output_path = OVERLAY_ROOT / input_path.relative_to(input_root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), image_bgr):
        raise OSError(f"OpenCV could not write overlay: {output_path}")


def assess_candidate(points: np.ndarray, side: str, score: float, width: int, height: int) -> HandCandidate:
    """Check one hand's range, centrality, scale, and skeleton geometry."""
    flags: list[str] = []
    normalized = points / np.array([width, height], dtype=float)
    if not np.isfinite(normalized).all():
        flags.append("non_finite_coordinate")

    strictly_in_frame = (
        (normalized[:, 0] >= 0.0)
        & (normalized[:, 0] <= 1.0)
        & (normalized[:, 1] >= 0.0)
        & (normalized[:, 1] <= 1.0)
    )
    tolerance = OUT_OF_BOUNDS_TOLERANCE_NORMALIZED
    within_tolerance = (
        (normalized[:, 0] >= -tolerance)
        & (normalized[:, 0] <= 1.0 + tolerance)
        & (normalized[:, 1] >= -tolerance)
        & (normalized[:, 1] <= 1.0 + tolerance)
    )
    if not bool(within_tolerance.all()):
        flags.append("keypoint_far_out_of_range")
    if float(strictly_in_frame.mean()) < MIN_IN_FRAME_KEYPOINT_FRACTION:
        flags.append("too_many_out_of_frame_keypoints")

    palm = normalized[list(PALM_CENTER_INDICES)]
    palm_center = palm.mean(axis=0)
    center_distance = float(np.linalg.norm(palm_center - np.array([0.5, 0.5])))

    span = normalized.max(axis=0) - normalized.min(axis=0)
    bbox_diagonal = float(np.linalg.norm(span))
    if bbox_diagonal < MIN_HAND_BBOX_DIAGONAL_NORMALIZED:
        flags.append("keypoints_too_concentrated")
    if bbox_diagonal > MAX_HAND_BBOX_DIAGONAL_NORMALIZED:
        flags.append("keypoints_too_dispersed")

    palm_pair_distances = [
        np.linalg.norm(palm[first] - palm[second])
        for first, second in ((0, 1), (1, 2), (2, 3), (3, 4), (0, 4))
    ]
    if max(palm_pair_distances) < MIN_PALM_SPREAD_NORMALIZED:
        flags.append("palm_too_concentrated")

    bone_lengths = np.array(
        [np.linalg.norm(normalized[first] - normalized[second]) for first, second in HAND_CONNECTIONS],
        dtype=float,
    )
    median_bone = float(np.median(bone_lengths))
    if median_bone < MIN_MEDIAN_BONE_LENGTH_NORMALIZED:
        flags.append("skeleton_bones_too_short")
    elif (
        float(bone_lengths.min() / median_bone) < MIN_BONE_TO_MEDIAN_RATIO
        or float(bone_lengths.max() / median_bone) > MAX_BONE_TO_MEDIAN_RATIO
    ):
        flags.append("implausible_bone_length_ratio")

    geometry_ok = not flags
    center_ok = geometry_ok and center_distance <= CENTER_MAX_DISTANCE_NORMALIZED
    # Score remains the dominant ranking signal; centre proximity breaks ties.
    quality = float(score) + 0.20 * max(0.0, 1.0 - center_distance / CENTER_MAX_DISTANCE_NORMALIZED)
    if not geometry_ok:
        quality -= 1.0
    # A small extrapolation around an image edge can occur even in a good hand
    # fit. It is never exported as a pixel coordinate: those individual points
    # remain blank, while the hand is retained only when enough points are in
    # range for the configurable reliability check above.
    export_points = points.copy()
    export_points[~strictly_in_frame] = np.nan
    return HandCandidate(
        side=side,
        score=score,
        points=export_points,
        center_distance=center_distance,
        geometry_ok=geometry_ok,
        center_ok=center_ok,
        quality=quality,
        check_flags=tuple(flags),
    )


def run_detection(
    landmarker: Any,
    image_bgr: np.ndarray,
    original_width: int,
    original_height: int,
    method: str,
    offset: tuple[int, int] = (0, 0),
    crop_fraction: float | None = None,
) -> DetectionAttempt:
    """Run one inference and map its landmarks back to the original image."""
    image_height, image_width = image_bgr.shape[:2]
    diagnostics: list[str] = []
    try:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        result = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb))
    except Exception as error:
        logging.exception("MediaPipe inference failed during %s", method)
        return DetectionAttempt(method, [], crop_fraction, (f"inference_error:{type(error).__name__}",))

    candidates: list[HandCandidate] = []
    for landmarks, categories in zip(result.hand_landmarks, result.handedness):
        if len(landmarks) != 21 or not categories:
            diagnostics.append("invalid_hand_result")
            continue
        category = categories[0]
        side = model_handedness_to_anatomical(category.category_name)
        if side is None:
            diagnostics.append("invalid_handedness")
            continue
        points = np.array(
            [landmark_to_pixel(landmark, image_width, image_height) for landmark in landmarks],
            dtype=float,
        )
        points += np.array(offset, dtype=float)
        candidates.append(assess_candidate(points, side, float(category.score), original_width, original_height))
    return DetectionAttempt(method, candidates, crop_fraction, tuple(diagnostics))


def center_crop(image_bgr: np.ndarray, fraction: float) -> tuple[np.ndarray, tuple[int, int]]:
    """Return an unscaled crop centred on the source image and its x/y offset."""
    height, width = image_bgr.shape[:2]
    crop_width = max(1, round(width * fraction))
    crop_height = max(1, round(height * fraction))
    x0 = (width - crop_width) // 2
    y0 = (height - crop_height) // 2
    return image_bgr[y0:y0 + crop_height, x0:x0 + crop_width], (x0, y0)


def best_center_candidate(attempt: DetectionAttempt) -> HandCandidate | None:
    """Choose the highest-quality valid hand that is near the image centre."""
    candidates = [candidate for candidate in attempt.candidates if candidate.center_ok]
    return max(candidates, key=lambda candidate: candidate.quality, default=None)


def select_output_hands(attempt: DetectionAttempt, target: HandCandidate) -> dict[str, HandCandidate]:
    """Keep the centre target and, optionally, one non-duplicate valid second hand."""
    selected = {target.side: target}
    if not KEEP_VALID_NON_TARGET_HANDS:
        return selected
    by_side: dict[str, list[HandCandidate]] = defaultdict(list)
    for candidate in attempt.candidates:
        if candidate.geometry_ok:
            by_side[candidate.side].append(candidate)
    for side, candidates in by_side.items():
        if side not in selected:
            selected[side] = max(candidates, key=lambda candidate: candidate.quality)
    return selected


def process_image(
    image_path: Path,
    initial_landmarker: Any,
    retry_landmarker: Any,
    columns: list[str],
) -> tuple[dict[str, Any], dict[str, Any], str]:
    """Use original, lower-threshold, then centred-crop inference when needed."""
    record = empty_record(image_path.name, columns)
    log_record: dict[str, Any] = {
        "filename": image_path.name,
        "source_relative_path": str(image_path.relative_to(INPUT_ROOT)),
        "original_candidate_count": 0,
        "original_reliable_center": False,
        "retry_reason": "",
        "attempted_methods": "",
        "crop_fraction": np.nan,
    }
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        record["detection_status"] = "failed:image_read_error"
        record["detection_method"] = "none"
        record["retry_count"] = 0
        log_record.update(record)
        return record, log_record, "unreliable"

    try:
        height, width = image_bgr.shape[:2]
        attempts = [run_detection(initial_landmarker, image_bgr, width, height, "original")]
        original = attempts[0]
        original_target = best_center_candidate(original)
        log_record["original_candidate_count"] = len(original.candidates)
        log_record["original_reliable_center"] = original_target is not None

        if original_target is None:
            if not original.candidates:
                retry_reason = "no_hand_detected"
            elif any(candidate.geometry_ok for candidate in original.candidates):
                retry_reason = "all_valid_hands_outside_center"
            else:
                retry_reason = "center_or_geometry_check_failed"
            log_record["retry_reason"] = retry_reason

            lower_attempt = run_detection(retry_landmarker, image_bgr, width, height, "lower_threshold")
            attempts.append(lower_attempt)
            if best_center_candidate(lower_attempt) is None:
                for fraction in CENTER_CROP_FRACTIONS:
                    cropped_image, offset = center_crop(image_bgr, fraction)
                    crop_attempt = run_detection(
                        retry_landmarker,
                        cropped_image,
                        width,
                        height,
                        "center_crop",
                        offset,
                        fraction,
                    )
                    attempts.append(crop_attempt)
                    if best_center_candidate(crop_attempt) is not None:
                        break

        reliable_attempts = [
            (attempt, target)
            for attempt in attempts
            if (target := best_center_candidate(attempt)) is not None
        ]
        retry_count = len(attempts) - 1
        log_record["attempted_methods"] = ";".join(attempt.method for attempt in attempts)

        if reliable_attempts:
            # Original output is retained whenever it was already reliable; a
            # retry result is selected only when it has a stronger automatic
            # reliability score, never merely because it ran later.
            final_attempt, target = max(reliable_attempts, key=lambda item: item[1].quality)
            selected_hands = select_output_hands(final_attempt, target)
            record["detection_method"] = final_attempt.method
            record["retry_count"] = retry_count
            record["center_distance"] = target.center_distance
            record["detection_status"] = "reliable_center_hand"
            if len(selected_hands) == 2:
                record["detection_status"] += ";secondary_hand_retained"

            for side, candidate in selected_hands.items():
                record[f"{side}_hand_score"] = candidate.score
                for index, (x, y) in enumerate(candidate.points):
                    record[f"{side}_kp{index}_x"] = x
                    record[f"{side}_kp{index}_y"] = y
                draw_hand(image_bgr, candidate.points, side)

            log_record["crop_fraction"] = final_attempt.crop_fraction if final_attempt.crop_fraction else np.nan
            outcome = "original_success" if final_attempt.method == "original" else "retry_success"
        else:
            all_candidates = [candidate for attempt in attempts for candidate in attempt.candidates]
            record["detection_status"] = "uncertain" if all_candidates else "failed"
            record["detection_method"] = "none"
            record["retry_count"] = retry_count
            outcome = "unreliable"

        # Even no-hand images get an unchanged copy in the overlay tree, so
        # every readable source image has a directly corresponding output file.
        write_overlay(image_bgr, image_path, INPUT_ROOT)
        log_record.update(
            {
                "detection_status": record["detection_status"],
                "detection_method": record["detection_method"],
                "retry_count": record["retry_count"],
                "center_distance": record["center_distance"],
                "final_hand_count": len(selected_hands) if reliable_attempts else 0,
            }
        )
        return record, log_record, outcome
    except Exception as error:  # keep remaining images running after one failure
        record["detection_status"] = f"inference_or_overlay_error: {type(error).__name__}"
        record["detection_method"] = "none"
        record["retry_count"] = 0
        logging.exception("Failed to process %s", image_path)
        log_record.update(record)
        return record, log_record, "unreliable"


def main() -> int:
    configure_logging()
    if not INPUT_ROOT.is_dir():
        logging.error("Input root does not exist: %s", INPUT_ROOT)
        return 2
    if not MODEL_PATH.is_file():
        logging.error("Local MediaPipe model is missing: %s", MODEL_PATH)
        logging.error("Download it once from the official MediaPipe model URL recorded in MODEL_PROVENANCE.md.")
        return 2

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    columns = make_columns()
    images = discover_images(INPUT_ROOT)
    if not images:
        logging.warning("No supported images found under data_* folders in %s", INPUT_ROOT)

    def make_options(detection_threshold: float, presence_threshold: float, tracking_threshold: float) -> Any:
        return mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL_PATH)),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_hands=NUM_HANDS,
            min_hand_detection_confidence=detection_threshold,
            min_hand_presence_confidence=presence_threshold,
            min_tracking_confidence=tracking_threshold,
        )

    records: list[dict[str, Any]] = []
    log_records: list[dict[str, Any]] = []
    outcomes: Counter[str] = Counter()
    # Two fixed configurations are necessary because MediaPipe thresholds are
    # construction-time options. Each is initialized once, never per image.
    with ExitStack() as stack:
        initial_landmarker = stack.enter_context(
            mp.tasks.vision.HandLandmarker.create_from_options(
                make_options(
                    INITIAL_HAND_DETECTION_CONFIDENCE,
                    INITIAL_HAND_PRESENCE_CONFIDENCE,
                    INITIAL_TRACKING_CONFIDENCE,
                )
            )
        )
        retry_landmarker = stack.enter_context(
            mp.tasks.vision.HandLandmarker.create_from_options(
                make_options(
                    RETRY_HAND_DETECTION_CONFIDENCE,
                    RETRY_HAND_PRESENCE_CONFIDENCE,
                    RETRY_TRACKING_CONFIDENCE,
                )
            )
        )
        for image_path in tqdm(images, desc="MediaPipe hand detection", unit="image"):
            record, log_record, outcome = process_image(
                image_path,
                initial_landmarker,
                retry_landmarker,
                columns,
            )
            records.append(record)
            log_records.append(log_record)
            outcomes[outcome] += 1

    frame = pd.DataFrame(records, columns=columns)
    # Explicitly retain numeric values (and write missing keypoints as blank cells).
    for column in ("retry_count", "center_distance", "left_hand_score", "right_hand_score", *columns[7:]):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame.to_excel(EXCEL_OUTPUT, index=False, engine="openpyxl")

    log_columns = [
        "filename",
        "source_relative_path",
        "detection_status",
        "detection_method",
        "retry_count",
        "center_distance",
        "original_candidate_count",
        "original_reliable_center",
        "retry_reason",
        "attempted_methods",
        "crop_fraction",
        "final_hand_count",
    ]
    pd.DataFrame(log_records, columns=log_columns).to_csv(DETECTION_LOG_OUTPUT, index=False, encoding="utf-8-sig")

    retry_images = sum(int(record.get("retry_count", 0)) > 0 for record in log_records)
    logging.info(
        "Images: %d | initial reliable centre: %d | retried: %d | retry successes: %d | uncertain/failed: %d",
        len(images),
        outcomes["original_success"],
        retry_images,
        outcomes["retry_success"],
        outcomes["unreliable"],
    )
    logging.info("Excel written: %s", EXCEL_OUTPUT)
    logging.info("Overlays written below: %s", OVERLAY_ROOT)
    logging.info("Detection log written: %s", DETECTION_LOG_OUTPUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
