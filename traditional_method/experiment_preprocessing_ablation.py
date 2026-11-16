from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from enhanced_traditional import (
    BASE_COLUMNS,
    candidate_spec,
    evaluate_candidate,
    extract_enhanced_features,
)
from evaluate_rule_based import POSE_MAP
from experiment_course_preprocessing import (
    fill_holes,
    keep_connected_components,
    rule_features,
)
from extract_features import extract_features
from segment_hands import extract_and_draw_contours, resize_for_processing


PROJECT_DIR = Path(__file__).resolve().parents[1]
LABELS_PATH = PROJECT_DIR / "labels.csv"
BASELINE_PATH = (
    PROJECT_DIR
    / "traditional_method/results/enhanced_traditional_evaluation.csv"
)
BASELINE_RULE_PATH = (
    PROJECT_DIR / "traditional_method/results/rule_based_evaluation.csv"
)
OUTPUT_DIR = PROJECT_DIR / "traditional_method/results_course_filters"
ABLATION_PATH = OUTPUT_DIR / "preprocessing_ablation.csv"

VARIANTS = {
    "gaussian_only": {"gaussian": True},
    "median_only": {"median": True},
    "fill_components_only": {"fill_components": True},
    "gaussian_median": {"gaussian": True, "median": True},
    "all_course_steps": {
        "gaussian": True,
        "median": True,
        "fill_components": True,
    },
}

METHODS = [
    "baseline_fixed_svm",
    "enhanced_geometry_random_forest",
    "enhanced_geometry_gradient_boosting",
]


def segment_variant(image, options):
    working = (
        cv2.GaussianBlur(image, (5, 5), 0)
        if options.get("gaussian")
        else image
    )
    ycrcb = cv2.cvtColor(working, cv2.COLOR_BGR2YCrCb)
    ycrcb_mask = cv2.inRange(
        ycrcb,
        np.array([0, 133, 77], dtype=np.uint8),
        np.array([255, 173, 127], dtype=np.uint8),
    )
    hsv = cv2.cvtColor(working, cv2.COLOR_BGR2HSV)
    hsv_mask = cv2.inRange(
        hsv,
        np.array([0, 50, 0], dtype=np.uint8),
        np.array([179, 255, 255], dtype=np.uint8),
    )
    mask = cv2.bitwise_and(ycrcb_mask, hsv_mask)

    if options.get("median"):
        mask = cv2.medianBlur(mask, 5)

    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)

    if options.get("fill_components"):
        return fill_holes(keep_connected_components(mask))

    _, selected_mask, _, _ = extract_and_draw_contours(image, mask)
    return selected_mask


def build_features(labels, options):
    feature_records = []
    rule_records = []

    for _, row in labels.iterrows():
        image = cv2.imread(str(PROJECT_DIR / row["path"]))
        if image is None:
            raise FileNotFoundError(row["path"])
        image = resize_for_processing(image)
        mask = segment_variant(image, options)

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
        feature_records.append(record)

        rule = rule_features(mask, row["filename"])
        true_pose = POSE_MAP[row["pose"]]
        rule_records.append(true_pose == rule["predicted_pose"])

    return pd.DataFrame(feature_records), float(np.mean(rule_records))


def evaluate_variant(name, data, rule_accuracy):
    hog_columns = [column for column in data.columns if column.startswith("hog_")]
    records = []
    for method in METHODS:
        columns, pipeline, grid = candidate_spec(method, hog_columns)
        evaluation, _ = evaluate_candidate(data, method, columns, pipeline, grid)
        cv_row = evaluation[evaluation["protocol"] == "stratified_6fold"].iloc[0]
        records.append(
            {
                "variant": name,
                "method": method,
                "accuracy": cv_row["accuracy"],
            }
        )
    records.append(
        {"variant": name, "method": "rule_based", "accuracy": rule_accuracy}
    )
    return records


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    labels = pd.read_csv(LABELS_PATH)
    baseline = pd.read_csv(BASELINE_PATH)
    baseline_cv = baseline[baseline["protocol"] == "stratified_6fold"]
    original_rule = pd.read_csv(BASELINE_RULE_PATH)["correct"].mean()

    records = []
    for method in METHODS:
        row = baseline_cv[baseline_cv["method"] == method].iloc[0]
        records.append(
            {"variant": "original", "method": method, "accuracy": row["accuracy"]}
        )
    records.append(
        {"variant": "original", "method": "rule_based", "accuracy": original_rule}
    )

    for name, options in VARIANTS.items():
        print(f"Evaluating {name}...", flush=True)
        data, rule_accuracy = build_features(labels, options)
        records.extend(evaluate_variant(name, data, rule_accuracy))

    table = pd.DataFrame(records)
    original = table[table["variant"] == "original"][["method", "accuracy"]]
    original = original.rename(columns={"accuracy": "original_accuracy"})
    table = table.merge(original, on="method", how="left")
    table["accuracy_change"] = table["accuracy"] - table["original_accuracy"]
    table.to_csv(ABLATION_PATH, index=False)

    print("\nSix-fold preprocessing ablation")
    print(
        table.pivot(index="variant", columns="method", values="accuracy")
        .round(4)
        .to_string()
    )
    print("\nBest change by method")
    for method, group in table.groupby("method"):
        best = group.loc[group["accuracy"].idxmax()]
        print(
            f"{method}: {best['variant']} = {best['accuracy']:.4f} "
            f"(change {best['accuracy_change']:+.4f})"
        )
    print(f"\nSaved: {ABLATION_PATH}")


if __name__ == "__main__":
    main()
