"""Classify ABD, ADD, and OPP from existing 2-D MediaPipe hand keypoints.

The program does not run MediaPipe.  It reads the already generated keypoint
spreadsheet, derives 2-D thumb/palm features, and evaluates RBF-SVMs with a
jointly stratified four-fold cross-validation protocol.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import cv2
import matplotlib
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =============================================================================
# Configuration
# =============================================================================
DATA_ROOT = Path(r"D:\uottawa\phd\SHUEE\MV_gloria_data")
KEYPOINT_FILE = DATA_ROOT / "mediapipe_hand_keypoints" / "mediapipe_hand_keypoints.xlsx"
GROUND_TRUTH_FILE = DATA_ROOT / "hand_ground_truth.xlsx"
# This independent script and its existing output files reside together.
OUTPUT_DIR = Path(__file__).resolve().parent
FEATURE_OUTPUT = OUTPUT_DIR / "thumb_features.xlsx"
PREDICTION_OUTPUT = OUTPUT_DIR / "svm_predictions.xlsx"
OVERALL_METRICS_OUTPUT = OUTPUT_DIR / "overall_metrics.xlsx"
GROUP_METRICS_OUTPUT = OUTPUT_DIR / "group_metrics.xlsx"
CONFUSION_DIR = OUTPUT_DIR / "confusion_matrices"
VALIDATION_LOG = OUTPUT_DIR / "validation_log.txt"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
CLASS_ORDER = ("ABD", "ADD", "OPP")
CAMERA_ANGLE_ORDER = ("F", "45", "90")
OCCLUSION_ORDER = ("NONE", "HAND", "PAPER")
RANDOM_SEED = 20261123
N_SPLITS = 4

# Target-hand and 2-D skeleton plausibility checks.
PALM_INDICES = (0, 5, 9, 13, 17)
REQUIRED_TARGET_INDICES = (0, 5, 9, 13, 17)
MIN_VALID_KEYPOINTS_FOR_TARGET = 12
TARGET_CENTER_MAX_DISTANCE = 0.47
AMBIGUOUS_CENTER_DISTANCE_MARGIN = 0.008
MIN_HAND_BBOX_DIAGONAL_NORMALIZED = 0.025
MAX_HAND_BBOX_DIAGONAL_NORMALIZED = 1.35
MIN_PALM_WIDTH_NORMALIZED = 0.008
MIN_MEDIAN_BONE_LENGTH_NORMALIZED = 0.006
MAX_BONE_TO_MEDIAN_RATIO = 12.0

# Unified feature set: every eligible image uses all F1--F20.  F1--F8 are
# retained from the previous script; F9--F20 add independent thumb/index/middle
# geometry while intentionally avoiding the unstable thumb-base points K1/K2.
FEATURE_COLUMNS = tuple(f"F{index}" for index in range(1, 21))
MIN_TRAIN_SAMPLES = 6
MIN_TRAIN_CLASSES = 2
MIN_AVAILABLE_FEATURES_FOR_SVM = 12
PCA_VARIANCE_TO_RETAIN = 0.95
SVM_C = 1.0
SVM_GAMMA = "scale"

# Official MediaPipe hand topology used only for 2-D plausibility screening.
HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
)


def configure_logging() -> None:
    """Write reproducibility and validation messages to console and a file."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(VALIDATION_LOG, encoding="utf-8", mode="w")],
    )


def normalized_text(value: Any) -> str:
    """Return a stripped uppercase label while retaining a blank missing value."""
    if pd.isna(value):
        return ""
    return str(value).strip().upper()


def display_image_size(path: Path) -> tuple[int, int]:
    """Return width/height in the visual orientation used by OpenCV MediaPipe input."""
    with Image.open(path) as image:
        width, height = image.size
        if image.getexif().get(274) in {5, 6, 7, 8}:
            width, height = height, width
    return width, height


def discover_images() -> tuple[dict[str, Path], list[str]]:
    """Map image filename to source file and report duplicate-name ambiguities."""
    image_paths = [
        path
        for directory in DATA_ROOT.iterdir()
        if directory.is_dir() and directory.name.casefold().startswith("data_")
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.casefold() in IMAGE_EXTENSIONS
    ]
    by_name: dict[str, list[Path]] = {}
    for path in image_paths:
        by_name.setdefault(path.name, []).append(path)
    duplicates = sorted(name for name, paths in by_name.items() if len(paths) != 1)
    return {name: paths[0] for name, paths in by_name.items() if len(paths) == 1}, duplicates


def points_from_row(row: pd.Series, side: str) -> np.ndarray:
    """Read 21 x/y points from a MediaPipe spreadsheet row."""
    points = np.full((21, 2), np.nan, dtype=float)
    for index in range(21):
        x_column = f"{side}_kp{index}_x"
        y_column = f"{side}_kp{index}_y"
        if x_column in row and y_column in row:
            points[index] = (pd.to_numeric(row[x_column], errors="coerce"), pd.to_numeric(row[y_column], errors="coerce"))
    return points


def finite_indices(points: np.ndarray, indices: tuple[int, ...]) -> bool:
    """Return whether every requested keypoint has two finite coordinates."""
    return bool(np.isfinite(points[list(indices)]).all())


def point_to_segment_distance(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    """Compute Euclidean distance from a point to a finite 2-D line segment."""
    vector = end - start
    squared = float(np.dot(vector, vector))
    if squared <= np.finfo(float).eps:
        return np.nan
    fraction = float(np.clip(np.dot(point - start, vector) / squared, 0.0, 1.0))
    return float(np.linalg.norm(point - (start + fraction * vector)))


def skeleton_plausibility(points: np.ndarray, width: int, height: int) -> tuple[bool, str, float]:
    """Screen a hand independently before it competes as the central target."""
    valid = np.isfinite(points).all(axis=1)
    if int(valid.sum()) < MIN_VALID_KEYPOINTS_FOR_TARGET:
        return False, "too_few_keypoints", np.nan
    if not finite_indices(points, REQUIRED_TARGET_INDICES):
        return False, "missing_required_thumb_or_palm_keypoint", np.nan

    normalized = points[valid] / np.array([width, height], dtype=float)
    if not ((normalized >= 0.0).all() and (normalized <= 1.0).all()):
        return False, "keypoint_outside_image", np.nan
    bbox_diagonal = float(np.linalg.norm(normalized.max(axis=0) - normalized.min(axis=0)))
    if not MIN_HAND_BBOX_DIAGONAL_NORMALIZED <= bbox_diagonal <= MAX_HAND_BBOX_DIAGONAL_NORMALIZED:
        return False, "implausible_hand_extent", np.nan

    palm = points[list(PALM_INDICES)] / np.array([width, height], dtype=float)
    palm_width = float(np.linalg.norm(palm[4] - palm[1]))
    if palm_width < MIN_PALM_WIDTH_NORMALIZED:
        return False, "palm_too_concentrated", np.nan

    bone_lengths = np.array(
        [
            np.linalg.norm((points[first] - points[second]) / np.array([width, height], dtype=float))
            for first, second in HAND_CONNECTIONS
            if valid[first] and valid[second]
        ],
        dtype=float,
    )
    median_bone = float(np.median(bone_lengths))
    if median_bone < MIN_MEDIAN_BONE_LENGTH_NORMALIZED:
        return False, "skeleton_bones_too_short", np.nan
    if float(bone_lengths.max() / median_bone) > MAX_BONE_TO_MEDIAN_RATIO:
        return False, "implausible_bone_ratio", np.nan

    palm_center = palm.mean(axis=0)
    center_distance = float(np.linalg.norm(palm_center - np.array([0.5, 0.5])))
    return True, "ok", center_distance


def select_target_hand(row: pd.Series, width: int, height: int) -> tuple[str, np.ndarray | None, float, str]:
    """Select the nearest plausible centre hand; ties are explicitly uncertain."""
    candidates: list[tuple[str, np.ndarray, float, float]] = []
    for side in ("left", "right"):
        points = points_from_row(row, side)
        valid, reason, center_distance = skeleton_plausibility(points, width, height)
        if not valid or center_distance > TARGET_CENTER_MAX_DISTANCE:
            continue
        hand_score = pd.to_numeric(row.get(f"{side}_hand_score", np.nan), errors="coerce")
        score = float(hand_score) if pd.notna(hand_score) else 0.0
        candidates.append((side, points, center_distance, score))
    if not candidates:
        return "uncertain", None, np.nan, "no_plausible_center_hand"

    candidates.sort(key=lambda item: (item[2], -item[3], item[0]))
    if len(candidates) > 1 and abs(candidates[0][2] - candidates[1][2]) < AMBIGUOUS_CENTER_DISTANCE_MARGIN:
        return "uncertain", None, np.nan, "two_hands_equally_central"
    side, points, distance, _ = candidates[0]
    return side, points, distance, "ok"


def local_palm_frame(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float] | None:
    """Build an index-to-pinky / wrist-to-MCP local 2-D frame for either hand."""
    if not finite_indices(points, PALM_INDICES):
        return None
    wrist = points[0]
    palm_center = points[list(PALM_INDICES)].mean(axis=0)
    across = points[17] - points[5]  # index MCP -> pinky MCP: same anatomical direction on both sides
    across_norm = float(np.linalg.norm(across))
    if across_norm <= np.finfo(float).eps:
        return None
    forward_raw = points[[5, 9, 13, 17]].mean(axis=0) - wrist
    forward_orthogonal = forward_raw - np.dot(forward_raw, across / across_norm) * (across / across_norm)
    forward_norm = float(np.linalg.norm(forward_orthogonal))
    if forward_norm <= np.finfo(float).eps:
        return None
    x_axis = across / across_norm
    y_axis = forward_orthogonal / forward_norm
    scale = (across_norm + float(np.linalg.norm(points[9] - wrist))) / 2.0
    if scale <= np.finfo(float).eps:
        return None
    return palm_center, x_axis, y_axis, scale


def normalized_angle(first: np.ndarray, second: np.ndarray) -> float:
    """Return the unsigned 2-D angle in [0, 1], normalized by pi."""
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denominator <= np.finfo(float).eps:
        return np.nan
    cosine = float(np.clip(np.dot(first, second) / denominator, -1.0, 1.0))
    return float(np.arccos(cosine) / np.pi)


def minimum_segment_distance(point: np.ndarray, points: np.ndarray, segments: tuple[tuple[int, int], ...]) -> float:
    """Find the closest distance to any valid segment in a finger or palm chain."""
    distances = [
        point_to_segment_distance(point, points[start], points[end])
        for start, end in segments
        if finite_indices(points, (start, end))
    ]
    return float(min(distances)) if distances else np.nan


def point_inside_palm_hull(point: np.ndarray, points: np.ndarray) -> float:
    """Return a binary 2-D palm-overlap feature, or NaN when the hull is absent."""
    if not finite_indices(points, PALM_INDICES):
        return np.nan
    hull = cv2.convexHull(points[list(PALM_INDICES)].astype(np.float32))
    return float(cv2.pointPolygonTest(hull, tuple(point.astype(float)), False) >= 0)


def extract_features(points: np.ndarray | None) -> tuple[dict[str, float], str]:
    """Calculate F1--F20 from scale-normalized 2-D palm/thumb geometry."""
    features = {name: np.nan for name in FEATURE_COLUMNS}
    if points is None:
        return features, "no_target_hand"
    frame = local_palm_frame(points)
    if frame is None:
        return features, "invalid_local_palm_frame"
    palm_center, x_axis, y_axis, scale = frame

    def present(*indices: int) -> bool:
        return finite_indices(points, tuple(indices))

    if present(4):
        thumb_tip_relative = points[4] - palm_center
        features["F1"] = float(np.dot(thumb_tip_relative, x_axis) / scale)
        features["F2"] = float(np.dot(thumb_tip_relative, y_axis) / scale)
    if present(3):
        features["F3"] = float(np.dot(points[3] - palm_center, x_axis) / scale)
    if present(4, 5):
        features["F4"] = float(np.linalg.norm(points[4] - points[5]) / scale)
    if present(3, 5):
        features["F5"] = float(np.linalg.norm(points[3] - points[5]) / scale)
    if present(3, 4, 5, 6):
        features["F6"] = normalized_angle(points[4] - points[3], points[6] - points[5])
    if present(4, 5, 6):
        features["F7"] = point_to_segment_distance(points[4], points[5], points[6]) / scale
    if present(4, *PALM_INDICES):
        features["F8"] = point_inside_palm_hull(points[4], points)

    # Distances from thumb tip/joint to index and middle joints/tips.
    if present(4, 8):
        features["F9"] = float(np.linalg.norm(points[4] - points[8]) / scale)
    if present(4, 12):
        features["F10"] = float(np.linalg.norm(points[4] - points[12]) / scale)
    if present(3, 6):
        features["F11"] = float(np.linalg.norm(points[3] - points[6]) / scale)
    if present(3, 10):
        features["F12"] = float(np.linalg.norm(points[3] - points[10]) / scale)

    # Thumb direction relative to the index and middle rays; no K1/K2 needed.
    if present(3, 4, 5, 8):
        features["F13"] = normalized_angle(points[4] - points[3], points[8] - points[5])
    if present(3, 4, 9, 12):
        features["F14"] = normalized_angle(points[4] - points[3], points[12] - points[9])
    if present(3):
        features["F15"] = float(np.dot(points[3] - palm_center, y_axis) / scale)

    # Non-redundant spatial separation from the distal index/middle skeletons,
    # palm centre, and palm edge; all values use the same palm scale.
    if present(4):
        features["F16"] = minimum_segment_distance(points[4], points, ((6, 7), (7, 8))) / scale
        features["F17"] = minimum_segment_distance(points[4], points, ((10, 11), (11, 12))) / scale
        features["F18"] = float(np.linalg.norm(points[4] - palm_center) / scale)
        features["F19"] = minimum_segment_distance(
            points[4],
            points,
            ((0, 5), (5, 9), (9, 13), (13, 17), (17, 0)),
        ) / scale
    if present(3, *PALM_INDICES):
        features["F20"] = point_inside_palm_hull(points[3], points)

    missing = [name for name, value in features.items() if not np.isfinite(value)]
    return features, "ok" if not missing else "missing:" + ",".join(missing)


def load_feature_table() -> tuple[pd.DataFrame, list[str]]:
    """Join ground truth and existing keypoints by filename and derive features."""
    keypoints = pd.read_excel(KEYPOINT_FILE)
    ground_truth = pd.read_excel(GROUND_TRUTH_FILE, dtype=str)
    messages: list[str] = []
    for table_name, frame in (("keypoints", keypoints), ("ground_truth", ground_truth)):
        duplicates = frame["filename"].duplicated(keep=False)
        if duplicates.any():
            messages.append(f"ERROR: duplicate filenames in {table_name}: {int(duplicates.sum())}")
        else:
            messages.append(f"OK: no duplicate filenames in {table_name}")
    keypoints = keypoints.drop_duplicates("filename", keep=False).set_index("filename", drop=False)
    ground_truth = ground_truth.drop_duplicates("filename", keep=False).copy()
    for column in ("final_class", "camera_angle", "occlusion_type"):
        ground_truth[column] = ground_truth[column].map(normalized_text)

    image_map, duplicate_images = discover_images()
    if duplicate_images:
        messages.append(f"ERROR: duplicate source-image filenames: {len(duplicate_images)}")
    missing_keypoints = sorted(set(ground_truth.filename) - set(keypoints.index))
    extra_keypoints = sorted(set(keypoints.index) - set(ground_truth.filename))
    missing_images = sorted(set(ground_truth.filename) - set(image_map))
    messages.extend(
        [
            f"ground_truth rows: {len(ground_truth)}",
            f"missing keypoint records: {len(missing_keypoints)}",
            f"unmatched extra keypoint records: {len(extra_keypoints)}",
            f"missing source images: {len(missing_images)}",
        ]
    )

    rows: list[dict[str, Any]] = []
    for _, label_row in ground_truth.iterrows():
        filename = label_row["filename"]
        result: dict[str, Any] = label_row.to_dict()
        result.update(
            {
                "target_hand": "uncertain",
                "target_selection_reason": "",
                "target_center_distance": np.nan,
                "feature_validity": "",
            }
        )
        result.update({feature: np.nan for feature in FEATURE_COLUMNS})
        if filename not in keypoints.index:
            result["target_selection_reason"] = "missing_keypoint_record"
            result["feature_validity"] = "missing_keypoint_record"
            rows.append(result)
            continue
        if filename not in image_map:
            result["target_selection_reason"] = "missing_or_duplicate_source_image"
            result["feature_validity"] = "missing_source_image"
            rows.append(result)
            continue

        keypoint_row = keypoints.loc[filename]
        width, height = display_image_size(image_map[filename])
        side, points, center_distance, target_reason = select_target_hand(keypoint_row, width, height)
        result["target_hand"] = side
        result["target_selection_reason"] = target_reason
        result["target_center_distance"] = center_distance
        if points is not None:
            features, feature_reason = extract_features(points)
            result.update(features)
            result["feature_validity"] = feature_reason
        else:
            result["feature_validity"] = "no_reliable_target_hand"
        rows.append(result)
    return pd.DataFrame(rows), messages


def make_svm_pipeline() -> Pipeline:
    """Create the required train-only imputer/scaler/PCA/RBF-SVM pipeline."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=PCA_VARIANCE_TO_RETAIN, svd_solver="full")),
            ("svc", SVC(kernel="rbf", C=SVM_C, gamma=SVM_GAMMA)),
        ]
    )


def available_feature_count(row: pd.Series, feature_names: tuple[str, ...]) -> int:
    return int(pd.to_numeric(row[list(feature_names)], errors="coerce").notna().sum())


def train_fold_model(training: pd.DataFrame) -> Pipeline | None:
    """Fit one unified PCA + RBF-SVM from this fold's training rows only."""
    candidate = training.loc[
        (training.target_hand != "uncertain")
        & (training.apply(lambda row: available_feature_count(row, FEATURE_COLUMNS) >= MIN_AVAILABLE_FEATURES_FOR_SVM, axis=1))
    ]
    class_count = int(candidate.final_class.nunique())
    if len(candidate) < MIN_TRAIN_SAMPLES or class_count < MIN_TRAIN_CLASSES:
        return None
    model = make_svm_pipeline()
    model.fit(candidate[list(FEATURE_COLUMNS)], candidate.final_class)
    return model


def cross_validated_predictions(features: pd.DataFrame, messages: list[str]) -> pd.DataFrame:
    """Produce one held-out prediction per image using joint 4-fold strata."""
    strata = features.final_class + "|" + features.camera_angle + "|" + features.occlusion_type
    stratum_counts = strata.value_counts().sort_index()
    messages.append("joint-stratum counts: " + "; ".join(f"{key}={value}" for key, value in stratum_counts.items()))
    if len(stratum_counts) != 27 or int(stratum_counts.min()) < N_SPLITS:
        raise ValueError("Joint strata do not support the required four-fold protocol.")

    splitter = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
    prediction_rows: list[dict[str, Any]] = []
    for fold, (train_index, test_index) in enumerate(splitter.split(features, strata), start=1):
        train = features.iloc[train_index].copy()
        test = features.iloc[test_index].copy()
        model = train_fold_model(train)
        for _, row in test.iterrows():
            model_name = ""
            reason = ""
            prediction: str | None = None
            if row.target_hand == "uncertain":
                reason = "no_reliable_target_hand"
            elif model is None:
                reason = "unified_model_training_unavailable"
            elif available_feature_count(row, FEATURE_COLUMNS) < MIN_AVAILABLE_FEATURES_FOR_SVM:
                reason = "insufficient_valid_features"
            else:
                prediction = str(model.predict(pd.DataFrame([row[list(FEATURE_COLUMNS)].to_dict()]))[0])
                model_name = "unified_pca_rbf_svm"
            prediction_rows.append(
                {
                    "filename": row.filename,
                    "true_class": row.final_class,
                    "predicted_class": prediction,
                    "fold": fold,
                    "is_correct": bool(prediction == row.final_class) if prediction is not None else False,
                    "unclassified_reason": reason,
                    "classification_model": model_name,
                    "camera_angle": row.camera_angle,
                    "occlusion_type": row.occlusion_type,
                    "target_hand": row.target_hand,
                    "center_distance": row.target_center_distance,
                }
            )
    predictions = pd.DataFrame(prediction_rows).sort_values("filename", key=lambda series: series.str.casefold()).reset_index(drop=True)
    return predictions


def metric_tables(predictions: pd.DataFrame, group_type: str, group_name: str) -> tuple[dict[str, Any], pd.DataFrame, np.ndarray]:
    """Evaluate a subset using only its held-out predictions, without retraining."""
    classified = predictions.loc[predictions.predicted_class.notna()].copy()
    total = len(predictions)
    if classified.empty:
        matrix = np.zeros((len(CLASS_ORDER), len(CLASS_ORDER)), dtype=int)
        precision = recall = f1 = support = np.zeros(len(CLASS_ORDER), dtype=float)
        accuracy = macro_f1 = np.nan
    else:
        matrix = confusion_matrix(classified.true_class, classified.predicted_class, labels=CLASS_ORDER)
        precision, recall, f1, support = precision_recall_fscore_support(
            classified.true_class,
            classified.predicted_class,
            labels=CLASS_ORDER,
            zero_division=0,
        )
        accuracy = float(accuracy_score(classified.true_class, classified.predicted_class))
        macro_f1 = float(np.mean(f1))
    summary = {
        "group_type": group_type,
        "group": group_name,
        "n_total": total,
        "n_classified": len(classified),
        "n_unclassified": total - len(classified),
        "unclassified_rate": (total - len(classified)) / total if total else np.nan,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
    }
    per_class = pd.DataFrame(
        {
            "group_type": group_type,
            "group": group_name,
            "class": CLASS_ORDER,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support_classified": support.astype(int),
        }
    )
    return summary, per_class, matrix


def save_confusion_matrix(matrix: np.ndarray, title: str, path: Path) -> None:
    """Save a consistently ordered ABD/ADD/OPP confusion-matrix PNG."""
    figure, axis = plt.subplots(figsize=(5, 4))
    image = axis.imshow(matrix, interpolation="nearest", cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set(
        xticks=np.arange(len(CLASS_ORDER)),
        yticks=np.arange(len(CLASS_ORDER)),
        xticklabels=CLASS_ORDER,
        yticklabels=CLASS_ORDER,
        xlabel="Predicted class",
        ylabel="True class",
        title=title,
    )
    threshold = matrix.max() / 2.0 if matrix.size and matrix.max() else 0.0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center", color="white" if matrix[row, column] > threshold else "black")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def save_metrics(predictions: pd.DataFrame) -> None:
    """Write all requested metrics and fixed-order confusion matrices."""
    CONFUSION_DIR.mkdir(parents=True, exist_ok=True)
    overall_summary, overall_per_class, overall_matrix = metric_tables(predictions, "overall", "all")
    save_confusion_matrix(overall_matrix, "Overall confusion matrix", CONFUSION_DIR / "overall.png")

    unclassified = predictions.loc[predictions.predicted_class.isna()].copy()
    error_distribution = (
        predictions.loc[predictions.predicted_class.notna() & ~predictions.is_correct]
        .groupby(["true_class", "predicted_class"], dropna=False)
        .size()
        .reset_index(name="count")
    )
    with pd.ExcelWriter(OVERALL_METRICS_OUTPUT, engine="openpyxl") as writer:
        pd.DataFrame([overall_summary]).to_excel(writer, sheet_name="overall", index=False)
        overall_per_class.to_excel(writer, sheet_name="per_class", index=False)
        unclassified.to_excel(writer, sheet_name="unclassified", index=False)
        error_distribution.to_excel(writer, sheet_name="misclassification_distribution", index=False)

    group_summaries: dict[str, list[dict[str, Any]]] = {"camera": [], "occlusion": []}
    group_classes: dict[str, list[pd.DataFrame]] = {"camera": [], "occlusion": []}
    for group_type, column, levels in (
        ("camera", "camera_angle", CAMERA_ANGLE_ORDER),
        ("occlusion", "occlusion_type", OCCLUSION_ORDER),
    ):
        for level in levels:
            subset = predictions.loc[predictions[column] == level]
            summary, per_class, matrix = metric_tables(subset, group_type, level)
            group_summaries[group_type].append(summary)
            group_classes[group_type].append(per_class)
            save_confusion_matrix(matrix, f"{group_type}: {level}", CONFUSION_DIR / f"{group_type}_{level}.png")
    with pd.ExcelWriter(GROUP_METRICS_OUTPUT, engine="openpyxl") as writer:
        pd.DataFrame(group_summaries["camera"]).to_excel(writer, sheet_name="camera_summary", index=False)
        pd.concat(group_classes["camera"], ignore_index=True).to_excel(writer, sheet_name="camera_per_class", index=False)
        pd.DataFrame(group_summaries["occlusion"]).to_excel(writer, sheet_name="occlusion_summary", index=False)
        pd.concat(group_classes["occlusion"], ignore_index=True).to_excel(writer, sheet_name="occlusion_per_class", index=False)


def main() -> int:
    configure_logging()
    if not KEYPOINT_FILE.is_file() or not GROUND_TRUTH_FILE.is_file():
        logging.error("Required input spreadsheet is missing.")
        return 2
    try:
        features, messages = load_feature_table()
        for message in messages:
            logging.info(message)
        expected_columns = {"filename", "final_class", "camera_angle", "occlusion_type"}
        if not expected_columns.issubset(features.columns):
            raise ValueError("Ground-truth columns required for cross-validation are absent.")
        if not features.final_class.isin(CLASS_ORDER).all():
            raise ValueError("Unexpected final_class label found.")
        if not features.camera_angle.isin(CAMERA_ANGLE_ORDER).all() or not features.occlusion_type.isin(OCCLUSION_ORDER).all():
            raise ValueError("Unexpected camera_angle or occlusion_type label found.")

        features = features.sort_values("filename", key=lambda series: series.str.casefold()).reset_index(drop=True)
        features.to_excel(FEATURE_OUTPUT, index=False, engine="openpyxl")
        logged_messages = len(messages)
        predictions = cross_validated_predictions(features, messages)
        for message in messages[logged_messages:]:
            logging.info(message)
        predictions.to_excel(PREDICTION_OUTPUT, index=False, engine="openpyxl")
        save_metrics(predictions)

        logging.info("Feature rows: %d", len(features))
        logging.info("Classified held-out rows: %d / %d", int(predictions.predicted_class.notna().sum()), len(predictions))
        logging.info("Results written to %s", OUTPUT_DIR)
        return 0
    except Exception:
        logging.exception("SVM workflow failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
