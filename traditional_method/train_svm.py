from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


PROJECT_DIR = Path(__file__).resolve().parents[1]

RESULTS_DIR = (
    PROJECT_DIR
    / "traditional_method/results"
)

FEATURES_PATH = RESULTS_DIR / "features.csv"

EVALUATION_PATH = (
    RESULTS_DIR
    / "svm_evaluation.csv"
)

PREDICTIONS_PATH = (
    RESULTS_DIR
    / "svm_predictions.csv"
)

CONFUSION_MATRIX_PATH = (
    RESULTS_DIR
    / "svm_confusion_matrices.png"
)

REPORT_PATH = (
    RESULTS_DIR
    / "svm_classification_report.txt"
)


FEATURE_COLUMNS = [
    "normalized_area",
    "normalized_perimeter",
    "aspect_ratio",
    "extent",
    "solidity",
    "circularity",
    "hu_1",
    "hu_2",
    "hu_3",
    "hu_4",
    "hu_5",
    "hu_6",
    "hu_7",
]

CLASS_ORDER = [
    "abduction",
    "adduction",
    "opposition",
]

CLASS_LABELS = [
    "ABD",
    "ADD",
    "OPP",
]


def create_model():
    return Pipeline(
        [
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "svm",
                SVC(
                    kernel="rbf",
                    C=1.0,
                    gamma="scale",
                ),
            ),
        ]
    )


def add_evaluation(
    protocol,
    test_data,
    predictions,
    evaluations,
    prediction_records,
    plot_records,
    report_sections,
):
    true_labels = test_data["pose"]

    accuracy = accuracy_score(
        true_labels,
        predictions,
    )

    balanced_accuracy = (
        balanced_accuracy_score(
            true_labels,
            predictions,
        )
    )

    evaluations.append(
        {
            "protocol": protocol,
            "sample_count": len(test_data),
            "accuracy": accuracy,
            "balanced_accuracy": (
                balanced_accuracy
            ),
        }
    )

    for (_, row), prediction in zip(
        test_data.iterrows(),
        predictions,
    ):
        prediction_records.append(
            {
                "protocol": protocol,
                "filename": row["filename"],
                "hand": row["hand"],
                "view": row["view"],
                "occlusion": row["occlusion"],
                "true_pose": row["pose"],
                "predicted_pose": prediction,
                "correct": (
                    row["pose"] == prediction
                ),
            }
        )

    plot_records.append(
        (
            protocol,
            true_labels.to_numpy(),
            predictions,
            accuracy,
        )
    )

    report = classification_report(
        true_labels,
        predictions,
        labels=CLASS_ORDER,
        target_names=CLASS_LABELS,
        zero_division=0,
    )

    report_sections.append(
        f"{protocol}\n"
        f"{'=' * len(protocol)}\n"
        f"{report}\n"
    )


def main():
    data = pd.read_csv(FEATURES_PATH)

    features = data[FEATURE_COLUMNS]
    labels = data["pose"]

    evaluations = []
    prediction_records = []
    plot_records = []
    report_sections = []

    # 1. 所有108张图片的6折分层交叉验证
    cross_validation = StratifiedKFold(
        n_splits=6,
        shuffle=True,
        random_state=42,
    )

    cv_predictions = cross_val_predict(
        create_model(),
        features,
        labels,
        cv=cross_validation,
    )

    add_evaluation(
        "stratified_6fold",
        data,
        cv_predictions,
        evaluations,
        prediction_records,
        plot_records,
        report_sections,
    )

    # 2. 使用无遮挡图片训练
    none_train = data["occlusion"] == "none"

    for test_occlusion in ["paper", "hand"]:
        test_rows = (
            data["occlusion"]
            == test_occlusion
        )

        model = create_model()

        model.fit(
            data.loc[
                none_train,
                FEATURE_COLUMNS,
            ],
            data.loc[
                none_train,
                "pose",
            ],
        )

        predictions = model.predict(
            data.loc[
                test_rows,
                FEATURE_COLUMNS,
            ]
        )

        add_evaluation(
            (
                "train_none_"
                f"test_{test_occlusion}"
            ),
            data.loc[test_rows],
            predictions,
            evaluations,
            prediction_records,
            plot_records,
            report_sections,
        )

    # 3. 留一个视角测试
    for test_view in [
        "front",
        "45deg",
        "90deg",
    ]:
        test_rows = data["view"] == test_view
        train_rows = ~test_rows

        model = create_model()

        model.fit(
            data.loc[
                train_rows,
                FEATURE_COLUMNS,
            ],
            data.loc[
                train_rows,
                "pose",
            ],
        )

        predictions = model.predict(
            data.loc[
                test_rows,
                FEATURE_COLUMNS,
            ]
        )

        add_evaluation(
            f"holdout_{test_view}",
            data.loc[test_rows],
            predictions,
            evaluations,
            prediction_records,
            plot_records,
            report_sections,
        )

    evaluation_table = pd.DataFrame(
        evaluations
    )

    prediction_table = pd.DataFrame(
        prediction_records
    )

    evaluation_table.to_csv(
        EVALUATION_PATH,
        index=False,
    )

    prediction_table.to_csv(
        PREDICTIONS_PATH,
        index=False,
    )

    REPORT_PATH.write_text(
        "\n".join(report_sections),
        encoding="utf-8",
    )

    # 画6个混淆矩阵
    figure, axes = plt.subplots(
        2,
        3,
        figsize=(15, 10),
    )

    for axis, (
        protocol,
        true_labels,
        predictions,
        accuracy,
    ) in zip(
        axes.flatten(),
        plot_records,
    ):
        ConfusionMatrixDisplay.from_predictions(
            true_labels,
            predictions,
            labels=CLASS_ORDER,
            display_labels=CLASS_LABELS,
            cmap="Blues",
            colorbar=False,
            ax=axis,
        )

        axis.set_title(
            f"{protocol}\n"
            f"accuracy={accuracy:.3f}"
        )

    figure.tight_layout()

    figure.savefig(
        CONFUSION_MATRIX_PATH,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(figure)

    print(evaluation_table.to_string(index=False))
    print(f"Evaluation: {EVALUATION_PATH}")
    print(f"Predictions: {PREDICTIONS_PATH}")
    print(
        "Confusion matrices: "
        f"{CONFUSION_MATRIX_PATH}"
    )
    print(f"Report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
