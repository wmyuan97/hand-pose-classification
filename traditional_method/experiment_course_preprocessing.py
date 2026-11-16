from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay

from enhanced_traditional import (
    BASE_COLUMNS,
    candidate_spec,
    evaluate_candidate,
    extract_enhanced_features,
)
from extract_features import extract_features
from segment_hands import MAX_COMPONENT_GAP, MIN_CONTOUR_AREA, resize_for_processing


PROJECT_DIR = Path(__file__).resolve().parents[1]
LABELS_PATH = PROJECT_DIR / "labels.csv"
BASELINE_EVALUATION_PATH = (
    PROJECT_DIR
    / "traditional_method/results/enhanced_traditional_evaluation.csv"
)
BASELINE_RULE_PATH = (
    PROJECT_DIR / "traditional_method/results/rule_based_evaluation.csv"
)

RESULTS_DIR = PROJECT_DIR / "traditional_method/results_course_filters"
MASK_DIR = RESULTS_DIR / "selected_masks"
FEATURES_PATH = RESULTS_DIR / "course_filter_features.csv"
EVALUATION_PATH = RESULTS_DIR / "course_filter_evaluation.csv"
PREDICTIONS_PATH = RESULTS_DIR / "course_filter_predictions.csv"
RULE_PATH = RESULTS_DIR / "course_filter_rule_evaluation.csv"
COMPARISON_PATH = RESULTS_DIR / "preprocessing_comparison.csv"
CONFUSION_PATH = RESULTS_DIR / "course_filter_confusion.png"


def box_gap(box1, box2):
    x1, y1, width1, height1 = box1
    x2, y2, width2, height2 = box2
    right1, bottom1 = x1 + width1, y1 + height1
    right2, bottom2 = x2 + width2, y2 + height2
    horizontal = max(x1 - right2, x2 - right1, 0)
    vertical = max(y1 - bottom2, y2 - bottom1, 0)
    return float(np.hypot(horizontal, vertical))


def keep_connected_components(mask):
    count, component_map, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )
    if count <= 1:
        return np.zeros_like(mask)

    valid = [
        index
        for index in range(1, count)
        if stats[index, cv2.CC_STAT_AREA] >= MIN_CONTOUR_AREA
    ]
    if not valid:
        return np.zeros_like(mask)

    primary = max(valid, key=lambda index: stats[index, cv2.CC_STAT_AREA])
    primary_box = tuple(int(value) for value in stats[primary, :4])
    selected = np.zeros_like(mask)

    for index in valid:
        component_box = tuple(int(value) for value in stats[index, :4])
        if index == primary or box_gap(primary_box, component_box) <= MAX_COMPONENT_GAP:
            selected[component_map == index] = 255
    return selected


def fill_holes(mask):
    # Padding guarantees that pixel (0, 0) is background for flood filling.
    padded = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    flooded = padded.copy()
    flood_mask = np.zeros((padded.shape[0] + 2, padded.shape[1] + 2), np.uint8)
    cv2.floodFill(flooded, flood_mask, (0, 0), 255)
    holes = cv2.bitwise_not(flooded)
    filled = cv2.bitwise_or(padded, holes)
    return filled[1:-1, 1:-1]


def course_segment(image):
    # Course experiment: Gaussian smoothing before colour thresholding.
    smoothed = cv2.GaussianBlur(image, (5, 5), 0)

    ycrcb = cv2.cvtColor(smoothed, cv2.COLOR_BGR2YCrCb)
    ycrcb_mask = cv2.inRange(
        ycrcb,
        np.array([0, 133, 77], dtype=np.uint8),
        np.array([255, 173, 127], dtype=np.uint8),
    )

    hsv = cv2.cvtColor(smoothed, cv2.COLOR_BGR2HSV)
    hsv_mask = cv2.inRange(
        hsv,
        np.array([0, 50, 0], dtype=np.uint8),
        np.array([179, 255, 255], dtype=np.uint8),
    )
    mask = cv2.bitwise_and(ycrcb_mask, hsv_mask)

    # Median filtering removes isolated binary noise while preserving edges.
    mask = cv2.medianBlur(mask, 5)

    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)

    # Explicit connected-component filtering and complete hole filling.
    mask = keep_connected_components(mask)
    return fill_holes(mask)


def rule_features(mask, filename):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {
            "valid_defect_count": 0,
            "palm_radius": 0.0,
            "thumb_detected": False,
            "normalized_thumb_distance": np.nan,
            "thumb_index_angle": np.nan,
            "predicted_pose": "OPP",
            "status": "no_contour",
        }

    contour = max(contours, key=cv2.contourArea)
    distance_map = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    _, palm_radius, _, palm_center = cv2.minMaxLoc(distance_map)
    hull_indices = cv2.convexHull(contour, returnPoints=False)
    defects = (
        cv2.convexityDefects(contour, hull_indices)
        if hull_indices is not None and len(hull_indices) >= 3
        else None
    )

    valid_defects = []
    wrist_limit = palm_center[1] + 0.3 * palm_radius
    if defects is not None:
        for start_i, end_i, far_i, depth in defects.reshape(-1, 4):
            start = tuple(contour[start_i][0])
            end = tuple(contour[end_i][0])
            far = tuple(contour[far_i][0])
            if depth / 256.0 >= 20 and far[1] < wrist_limit:
                valid_defects.append((start, end, far))

    if not valid_defects or palm_radius <= 0:
        return {
            "valid_defect_count": len(valid_defects),
            "palm_radius": palm_radius,
            "thumb_detected": False,
            "normalized_thumb_distance": np.nan,
            "thumb_index_angle": np.nan,
            "predicted_pose": "OPP",
            "status": "no_valid_defects",
        }

    if filename.startswith("L_"):
        start, end, _ = min(valid_defects, key=lambda item: item[2][0])
        thumb_tip = min((start, end), key=lambda point: point[0])
        index_tip = max((start, end), key=lambda point: point[0])
    else:
        start, end, _ = max(valid_defects, key=lambda item: item[2][0])
        thumb_tip = max((start, end), key=lambda point: point[0])
        index_tip = min((start, end), key=lambda point: point[0])

    thumb_vertical_distance = palm_center[1] - thumb_tip[1]
    if thumb_vertical_distance > 1.5 * palm_radius:
        return {
            "valid_defect_count": len(valid_defects),
            "palm_radius": palm_radius,
            "thumb_detected": False,
            "normalized_thumb_distance": np.nan,
            "thumb_index_angle": np.nan,
            "predicted_pose": "OPP",
            "status": "thumb_rejected",
        }

    thumb_vector = np.asarray(thumb_tip, float) - np.asarray(palm_center, float)
    index_vector = np.asarray(index_tip, float) - np.asarray(palm_center, float)
    normalized_distance = float(np.linalg.norm(thumb_vector) / palm_radius)
    denominator = np.linalg.norm(thumb_vector) * np.linalg.norm(index_vector)
    angle = (
        float(
            np.degrees(
                np.arccos(
                    np.clip(np.dot(thumb_vector, index_vector) / denominator, -1, 1)
                )
            )
        )
        if denominator > 1e-8
        else np.nan
    )
    return {
        "valid_defect_count": len(valid_defects),
        "palm_radius": palm_radius,
        "thumb_detected": True,
        "normalized_thumb_distance": normalized_distance,
        "thumb_index_angle": angle,
        "predicted_pose": "ABD" if normalized_distance > 1.6 else "ADD",
        "status": "ok",
    }


def build_feature_table(labels):
    records = []
    rule_records = []
    pose_map = {"abduction": "ABD", "adduction": "ADD", "opposition": "OPP"}

    MASK_DIR.mkdir(parents=True, exist_ok=True)
    for _, row in labels.iterrows():
        image = cv2.imread(str(PROJECT_DIR / row["path"]))
        if image is None:
            raise FileNotFoundError(row["path"])
        image = resize_for_processing(image)
        mask = course_segment(image)
        stem = Path(row["filename"]).stem
        cv2.imwrite(str(MASK_DIR / f"{stem}_selected_mask.png"), mask)

        base = extract_features(mask)
        geometry, hog = extract_enhanced_features(mask, row["hand"])
        record = {
            "filename": row["filename"],
            "hand": row["hand"],
            "pose": row["pose"],
            "view": row["view"],
            "occlusion": row["occlusion"],
            "subject": int(row["index"]),
        }
        record.update({column: base[column] for column in BASE_COLUMNS})
        record.update(geometry)
        for index, value in enumerate(hog):
            record[f"hog_{index:04d}"] = float(value)
        records.append(record)

        rule = rule_features(mask, row["filename"])
        true_pose = pose_map[row["pose"]]
        rule_records.append(
            {
                "filename": row["filename"],
                "hand": row["hand"],
                "view": row["view"],
                "occlusion": row["occlusion"],
                "true_pose": true_pose,
                **rule,
                "correct": true_pose == rule["predicted_pose"],
            }
        )

    features = pd.DataFrame(records)
    rules = pd.DataFrame(rule_records)
    features.to_csv(FEATURES_PATH, index=False)
    rules.to_csv(RULE_PATH, index=False)
    return features, rules


def evaluate_models(data):
    hog_columns = [column for column in data.columns if column.startswith("hog_")]
    evaluations = []
    predictions = []
    methods = [
        "baseline_fixed_svm",
        "enhanced_geometry_svm",
        "enhanced_geometry_random_forest",
        "enhanced_geometry_gradient_boosting",
        "enhanced_geometry_hog_svm",
    ]
    for method in methods:
        columns, pipeline, grid = candidate_spec(method, hog_columns)
        evaluation, prediction = evaluate_candidate(
            data, method, columns, pipeline, grid
        )
        evaluations.append(evaluation)
        predictions.append(prediction)

    evaluation_table = pd.concat(evaluations, ignore_index=True)
    prediction_table = pd.concat(predictions, ignore_index=True)
    evaluation_table.to_csv(EVALUATION_PATH, index=False)
    prediction_table.to_csv(PREDICTIONS_PATH, index=False)

    cv_rows = prediction_table[prediction_table["protocol"] == "stratified_6fold"]
    figure, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()
    for axis, method in zip(axes, methods):
        rows = cv_rows[cv_rows["method"] == method]
        ConfusionMatrixDisplay.from_predictions(
            rows["true_pose"],
            rows["predicted_pose"],
            labels=["abduction", "adduction", "opposition"],
            display_labels=["ABD", "ADD", "OPP"],
            cmap="Blues",
            colorbar=False,
            ax=axis,
        )
        accuracy = rows["correct"].mean()
        axis.set_title(f"{method}\naccuracy={accuracy:.3f}")
    for axis in axes[len(methods) :]:
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(CONFUSION_PATH, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return evaluation_table


def create_comparison(new_evaluation, new_rules):
    baseline = pd.read_csv(BASELINE_EVALUATION_PATH).rename(
        columns={"accuracy": "original_accuracy"}
    )
    new = new_evaluation.rename(columns={"accuracy": "course_filter_accuracy"})
    comparison = baseline.merge(
        new,
        on=["method", "protocol", "sample_count"],
        how="outer",
    )
    comparison["accuracy_change"] = (
        comparison["course_filter_accuracy"] - comparison["original_accuracy"]
    )

    baseline_rules = pd.read_csv(BASELINE_RULE_PATH)
    rule_row = pd.DataFrame(
        [
            {
                "method": "rule_based",
                "protocol": "all_images",
                "sample_count": len(new_rules),
                "original_accuracy": baseline_rules["correct"].mean(),
                "course_filter_accuracy": new_rules["correct"].mean(),
                "accuracy_change": (
                    new_rules["correct"].mean() - baseline_rules["correct"].mean()
                ),
            }
        ]
    )
    comparison = pd.concat([comparison, rule_row], ignore_index=True)
    comparison.to_csv(COMPARISON_PATH, index=False)
    return comparison


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    labels = pd.read_csv(LABELS_PATH)
    if len(labels) != 54:
        raise ValueError("This experiment expects the original 54-image dataset")

    features, rules = build_feature_table(labels)
    evaluation = evaluate_models(features)
    comparison = create_comparison(evaluation, rules)

    print("\nStratified six-fold comparison")
    print(
        comparison[
            comparison["protocol"].isin(["stratified_6fold", "all_images"])
        ][
            [
                "method",
                "original_accuracy",
                "course_filter_accuracy",
                "accuracy_change",
            ]
        ].to_string(index=False)
    )
    print(f"\nComparison: {COMPARISON_PATH}")


if __name__ == "__main__":
    main()
