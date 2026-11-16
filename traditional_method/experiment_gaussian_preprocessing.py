from pathlib import Path

import pandas as pd

from enhanced_traditional import candidate_spec, evaluate_candidate
from experiment_preprocessing_ablation import build_features


PROJECT_DIR = Path(__file__).resolve().parents[1]
LABELS_PATH = PROJECT_DIR / "labels.csv"
BASELINE_PATH = (
    PROJECT_DIR
    / "traditional_method/results/enhanced_traditional_evaluation.csv"
)
OUTPUT_DIR = PROJECT_DIR / "traditional_method/results_gaussian_only"
FEATURES_PATH = OUTPUT_DIR / "gaussian_features.csv"
EVALUATION_PATH = OUTPUT_DIR / "gaussian_evaluation.csv"
PREDICTIONS_PATH = OUTPUT_DIR / "gaussian_predictions.csv"
COMPARISON_PATH = OUTPUT_DIR / "gaussian_comparison.csv"

METHODS = [
    "baseline_fixed_svm",
    "enhanced_geometry_svm",
    "enhanced_geometry_random_forest",
    "enhanced_geometry_gradient_boosting",
    "enhanced_geometry_hog_svm",
]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    labels = pd.read_csv(LABELS_PATH)
    data, _ = build_features(labels, {"gaussian": True})
    data.to_csv(FEATURES_PATH, index=False)
    hog_columns = [column for column in data.columns if column.startswith("hog_")]

    evaluations = []
    predictions = []
    for method in METHODS:
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

    baseline = pd.read_csv(BASELINE_PATH).rename(
        columns={"accuracy": "original_accuracy"}
    )
    gaussian = evaluation_table.rename(
        columns={"accuracy": "gaussian_accuracy"}
    )
    comparison = baseline.merge(
        gaussian,
        on=["method", "protocol", "sample_count"],
        how="inner",
    )
    comparison["accuracy_change"] = (
        comparison["gaussian_accuracy"] - comparison["original_accuracy"]
    )
    comparison.to_csv(COMPARISON_PATH, index=False)

    best_method = "enhanced_geometry_gradient_boosting"
    print(
        comparison[comparison["method"] == best_method][
            [
                "protocol",
                "sample_count",
                "original_accuracy",
                "gaussian_accuracy",
                "accuracy_change",
            ]
        ].to_string(index=False)
    )
    print(f"\nSaved: {COMPARISON_PATH}")


if __name__ == "__main__":
    main()
