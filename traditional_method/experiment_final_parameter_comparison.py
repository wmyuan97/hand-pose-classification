"""Compare original and tuned segmentation parameters in the full final pipeline."""

from pathlib import Path

import cv2
import pandas as pd
from sklearn.metrics import f1_score

from enhanced_traditional import (
    BASE_COLUMNS,
    GEOMETRY_COLUMNS,
    candidate_spec,
    evaluate_candidate,
    extract_enhanced_features,
)
from experiment_segmentation_parameter_tuning import (
    BEST_PATH,
    DEFAULTS,
    FEATURE_COLUMNS,
    ImageCache,
    segment,
)
from extract_features import extract_features


PROJECT_DIR = Path(__file__).resolve().parents[1]
LABELS_PATH = PROJECT_DIR / "labels.csv"
ORIGINAL_FEATURES_PATH = (
    PROJECT_DIR / "traditional_method/results/enhanced_features.csv"
)
ORIGINAL_EVALUATION_PATH = (
    PROJECT_DIR / "traditional_method/results/enhanced_traditional_evaluation.csv"
)
ORIGINAL_PREDICTIONS_PATH = (
    PROJECT_DIR / "traditional_method/results/enhanced_traditional_predictions.csv"
)
OUTPUT_DIR = PROJECT_DIR / "traditional_method/results_parameter_tuning"
TUNED_FEATURES_PATH = OUTPUT_DIR / "tuned_full_features.csv"
EVALUATION_PATH = OUTPUT_DIR / "final_parameter_evaluation.csv"
PREDICTIONS_PATH = OUTPUT_DIR / "final_parameter_predictions.csv"
COMPARISON_PATH = OUTPUT_DIR / "final_parameter_comparison.csv"

METHODS = [
    "baseline_fixed_svm",
    "enhanced_geometry_svm",
    "enhanced_geometry_random_forest",
    "enhanced_geometry_gradient_boosting",
    "enhanced_geometry_hog_svm",
]


def tuned_configuration():
    best = pd.read_csv(BEST_PATH).set_index("configuration")
    return {
        name: int(best.loc["coordinate_search_best", name])
        for name in DEFAULTS
    }


def build_tuned_features(labels, config):
    cache = ImageCache(labels)
    items = cache.get(config["target_height"])
    records = []
    for (_, row), item in zip(labels.iterrows(), items):
        mask, component_count = segment(item, config)
        if component_count == 0:
            raise ValueError(f"No selected hand component: {row['filename']}")

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

    table = pd.DataFrame(records)
    table.to_csv(TUNED_FEATURES_PATH, index=False)
    return table


def evaluate_tuned(data):
    hog_columns = [column for column in data.columns if column.startswith("hog_")]
    evaluations = []
    predictions = []
    for method in METHODS:
        print(f"Evaluating tuned parameters: {method}", flush=True)
        columns, pipeline, grid = candidate_spec(method, hog_columns)
        evaluation, prediction = evaluate_candidate(
            data, method, columns, pipeline, grid
        )
        evaluations.append(evaluation)
        predictions.append(prediction)
    return (
        pd.concat(evaluations, ignore_index=True),
        pd.concat(predictions, ignore_index=True),
    )


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    labels = pd.read_csv(LABELS_PATH)
    if len(labels) != 54:
        raise ValueError("Final comparison expects the original 54-image dataset")

    config = tuned_configuration()
    tuned_features = build_tuned_features(labels, config)
    tuned_evaluation, tuned_predictions = evaluate_tuned(tuned_features)

    original_evaluation = pd.read_csv(ORIGINAL_EVALUATION_PATH)
    original_predictions = pd.read_csv(ORIGINAL_PREDICTIONS_PATH)
    original_evaluation.insert(0, "segmentation", "original")
    tuned_evaluation.insert(0, "segmentation", "tuned")
    original_predictions.insert(0, "segmentation", "original")
    tuned_predictions.insert(0, "segmentation", "tuned")

    evaluation = pd.concat(
        [original_evaluation, tuned_evaluation], ignore_index=True
    )
    predictions = pd.concat(
        [original_predictions, tuned_predictions], ignore_index=True
    )

    f1_records = []
    for (segmentation_name, method, protocol), group in predictions.groupby(
        ["segmentation", "method", "protocol"], sort=False
    ):
        f1_records.append(
            {
                "segmentation": segmentation_name,
                "method": method,
                "protocol": protocol,
                "macro_f1": f1_score(
                    group["true_pose"],
                    group["predicted_pose"],
                    labels=["abduction", "adduction", "opposition"],
                    average="macro",
                    zero_division=0,
                ),
            }
        )
    f1_table = pd.DataFrame(f1_records)
    evaluation = evaluation.merge(
        f1_table,
        on=["segmentation", "method", "protocol"],
        how="left",
    )
    evaluation.to_csv(EVALUATION_PATH, index=False)
    predictions.to_csv(PREDICTIONS_PATH, index=False)

    original = evaluation[evaluation["segmentation"] == "original"].rename(
        columns={
            "accuracy": "original_accuracy",
            "macro_f1": "original_macro_f1",
        }
    )
    tuned = evaluation[evaluation["segmentation"] == "tuned"].rename(
        columns={
            "accuracy": "tuned_accuracy",
            "macro_f1": "tuned_macro_f1",
        }
    )
    comparison = original[
        [
            "method",
            "protocol",
            "sample_count",
            "original_accuracy",
            "original_macro_f1",
        ]
    ].merge(
        tuned[
            [
                "method",
                "protocol",
                "sample_count",
                "tuned_accuracy",
                "tuned_macro_f1",
            ]
        ],
        on=["method", "protocol", "sample_count"],
        how="outer",
    )
    comparison["change"] = (
        comparison["tuned_accuracy"] - comparison["original_accuracy"]
    )
    comparison["macro_f1_change"] = (
        comparison["tuned_macro_f1"] - comparison["original_macro_f1"]
    )
    comparison.to_csv(COMPARISON_PATH, index=False)

    print("\nFull final-pipeline comparison")
    print(comparison.to_string(index=False))
    print(f"\nEvaluation: {EVALUATION_PATH}")
    print(f"Predictions: {PREDICTIONS_PATH}")
    print(f"Comparison: {COMPARISON_PATH}")


if __name__ == "__main__":
    main()
