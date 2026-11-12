from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
ASSET_DIR = PROJECT_DIR / "report_assets"
TRAD_PATH = PROJECT_DIR / "traditional_method/results/enhanced_traditional_evaluation.csv"
AI_PATH = PROJECT_DIR / "ai_method/results/rtmpose_evaluation.csv"
CHART_PATH = ASSET_DIR / "method_comparison.png"


def metric(evaluation, method, protocol):
    row = evaluation[
        (evaluation["method"] == method)
        & (evaluation["protocol"] == protocol)
    ]
    return float(row.iloc[0]["accuracy"])


def cross_subject_average(evaluation, method):
    rows = evaluation[
        (evaluation["method"] == method)
        & evaluation["protocol"].str.startswith("train_other_test_subject_")
    ]
    return float(rows["accuracy"].mean())


def main():
    traditional = pd.read_csv(TRAD_PATH)
    ai = pd.read_csv(AI_PATH)
    methods = [
        ("Baseline\nSVM", traditional, "baseline_fixed_svm"),
        ("Traditional\nRandom Forest", traditional, "enhanced_geometry_random_forest"),
        ("Traditional\nGradient Boosting", traditional, "enhanced_geometry_gradient_boosting"),
        ("RTMPose\nSVM", ai, "rtmpose_keypoints_svm"),
        ("RTMPose\nRandom Forest", ai, "rtmpose_keypoints_random_forest"),
    ]
    cv_values = [metric(table, method, "stratified_6fold") for _, table, method in methods]
    cross_values = [cross_subject_average(table, method) for _, table, method in methods]
    labels = [label for label, _, _ in methods]

    figure, axis = plt.subplots(figsize=(10, 4.8))
    x = list(range(len(methods)))
    width = 0.36
    axis.bar(
        [value - width / 2 for value in x],
        [100 * value for value in cv_values],
        width,
        label="Stratified 6-fold",
        color="#4C78A8",
    )
    axis.bar(
        [value + width / 2 for value in x],
        [100 * value for value in cross_values],
        width,
        label="Cross-subject average",
        color="#F28E2B",
    )
    for offset, values in [(-width / 2, cv_values), (width / 2, cross_values)]:
        for index, value in enumerate(values):
            axis.text(
                index + offset,
                100 * value + 1.2,
                f"{100 * value:.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    axis.set_xticks(x, labels)
    axis.set_ylim(0, 100)
    axis.set_ylabel("Accuracy (%)")
    axis.set_title("Traditional and RTMPose comparison")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(loc="upper left")
    figure.tight_layout()
    figure.savefig(CHART_PATH, dpi=220, bbox_inches="tight")
    plt.close(figure)
    print(CHART_PATH)


if __name__ == "__main__":
    main()
