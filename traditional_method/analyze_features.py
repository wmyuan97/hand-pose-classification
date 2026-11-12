from pathlib import Path

import matplotlib

# 只保存图片，不弹出绘图窗口
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


PROJECT_DIR = Path(__file__).resolve().parents[1]

RESULTS_DIR = (
    PROJECT_DIR
    / "traditional_method/results"
)

FEATURES_PATH = RESULTS_DIR / "features.csv"

BOXPLOT_PATH = (
    RESULTS_DIR
    / "feature_boxplots.png"
)

PCA_PATH = (
    RESULTS_DIR
    / "pca_comparison.png"
)

PCA_SCORES_PATH = (
    RESULTS_DIR
    / "pca_scores.csv"
)

SUMMARY_PATH = (
    RESULTS_DIR
    / "pose_feature_summary.csv"
)


# 用于模型分析的特征
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

# 箱线图只显示容易解释的几何特征
GEOMETRIC_FEATURES = [
    "normalized_area",
    "normalized_perimeter",
    "aspect_ratio",
    "extent",
    "solidity",
    "circularity",
]

POSE_ORDER = [
    "abduction",
    "adduction",
    "opposition",
]

POSE_LABELS = [
    "ABD",
    "ADD",
    "OPP",
]

POSE_COLORS = {
    "abduction": "#d62728",
    "adduction": "#1f77b4",
    "opposition": "#2ca02c",
}

OCCLUSION_ORDER = [
    "none",
    "paper",
    "hand",
]

OCCLUSION_COLORS = {
    "none": "#1f77b4",
    "paper": "#ff7f0e",
    "hand": "#9467bd",
}


def create_boxplots(data):
    figure, axes = plt.subplots(
        2,
        3,
        figsize=(14, 8),
    )

    axes = axes.flatten()

    for axis, feature in zip(
        axes,
        GEOMETRIC_FEATURES,
    ):
        values_by_pose = [
            data.loc[
                data["pose"] == pose,
                feature,
            ].to_numpy()
            for pose in POSE_ORDER
        ]

        plot = axis.boxplot(
            values_by_pose,
            patch_artist=True,
        )

        for box, color in zip(
            plot["boxes"],
            [
                POSE_COLORS[pose]
                for pose in POSE_ORDER
            ],
        ):
            box.set_facecolor(color)
            box.set_alpha(0.65)

        axis.set_xticks([1, 2, 3])
        axis.set_xticklabels(POSE_LABELS)
        axis.set_title(feature)
        axis.set_ylabel("Feature value")
        axis.grid(
            axis="y",
            alpha=0.25,
        )

    figure.suptitle(
        "Geometric feature distributions by pose",
        fontsize=14,
    )

    figure.tight_layout()
    figure.savefig(
        BOXPLOT_PATH,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(figure)


def create_pca_analysis(data):
    features = data[FEATURE_COLUMNS]

    # 将不同量纲的特征统一到相近尺度
    scaler = StandardScaler()
    standardized_features = (
        scaler.fit_transform(features)
    )

    pca = PCA(n_components=2)

    pca_values = pca.fit_transform(
        standardized_features
    )

    pca_table = data[
        [
            "filename",
            "hand",
            "pose",
            "view",
            "occlusion",
        ]
    ].copy()

    pca_table["PC1"] = pca_values[:, 0]
    pca_table["PC2"] = pca_values[:, 1]

    pca_table.to_csv(
        PCA_SCORES_PATH,
        index=False,
    )

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(14, 6),
    )

    # 左图：按照姿势着色
    for pose in POSE_ORDER:
        rows = pca_table["pose"] == pose

        axes[0].scatter(
            pca_table.loc[rows, "PC1"],
            pca_table.loc[rows, "PC2"],
            label=pose,
            color=POSE_COLORS[pose],
            alpha=0.8,
            s=60,
        )

    axes[0].set_title("PCA colored by pose")
    axes[0].set_xlabel("Principal component 1")
    axes[0].set_ylabel("Principal component 2")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    # 右图：按照遮挡条件着色
    for occlusion in OCCLUSION_ORDER:
        rows = (
            pca_table["occlusion"]
            == occlusion
        )

        axes[1].scatter(
            pca_table.loc[rows, "PC1"],
            pca_table.loc[rows, "PC2"],
            label=occlusion,
            color=OCCLUSION_COLORS[occlusion],
            alpha=0.8,
            s=60,
        )

    axes[1].set_title(
        "PCA colored by occlusion"
    )
    axes[1].set_xlabel(
        "Principal component 1"
    )
    axes[1].set_ylabel(
        "Principal component 2"
    )
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    figure.tight_layout()
    figure.savefig(
        PCA_PATH,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(figure)

    explained = (
        pca.explained_variance_ratio_
    )

    print(
        "PCA explained variance: "
        f"PC1={explained[0]:.3f}, "
        f"PC2={explained[1]:.3f}, "
        f"total={explained.sum():.3f}"
    )


def main():
    data = pd.read_csv(FEATURES_PATH)

    missing_columns = [
        column
        for column in FEATURE_COLUMNS
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    if data[FEATURE_COLUMNS].isna().any().any():
        raise ValueError(
            "Feature table contains missing values"
        )

    # 保存每种姿势的均值和标准差
    summary = (
        data.groupby("pose")[FEATURE_COLUMNS]
        .agg(["mean", "std"])
    )

    summary.to_csv(SUMMARY_PATH)

    create_boxplots(data)
    create_pca_analysis(data)

    print(f"Boxplots: {BOXPLOT_PATH}")
    print(f"PCA plot: {PCA_PATH}")
    print(f"PCA scores: {PCA_SCORES_PATH}")
    print(f"Summary: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()