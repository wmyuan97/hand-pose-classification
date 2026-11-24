from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


PROJECT = Path(__file__).resolve().parents[1]
FEATURE_FILE = PROJECT / "traditional_method" / "results" / "features.csv"
FOLD_FILE = PROJECT / "traditional_method" / "traditional_4fold_results" / "predictions.xlsx"
OUTPUT_ROOT = PROJECT / "traditional_method" / "linear_svm_4fold_results"

CLASSES = ["ABD", "ADD", "OPP"]
ANGLES = ["F", "45", "90"]
OCCLUSIONS = ["NONE", "HAND", "PAPER"]

FEATURES = [
    "normalized_area", "normalized_perimeter", "aspect_ratio",
    "extent", "solidity", "circularity",
    *[f"hu_{i}" for i in range(1, 8)],
]


def image_keys(filenames):
    return filenames.fillna("").map(lambda name: Path(name.strip()).stem.casefold())


def load_data():
    data = pd.read_csv(FEATURE_FILE, dtype={"filename": str})
    data = data.rename(columns={
        "pose": "true_class",
        "view": "camera_angle",
        "occlusion": "occlusion_type",
    })

    data["true_class"] = data["true_class"].str.strip().str.upper().replace({
        "ABDUCTION": "ABD", "ADDUCTION": "ADD", "OPPOSITION": "OPP",
    })
    data["camera_angle"] = data["camera_angle"].str.strip().str.upper().replace({
        "FRONT": "F", "45DEG": "45", "90DEG": "90",
    })
    data["occlusion_type"] = data["occlusion_type"].str.strip().str.upper()

    for column, allowed in [
        ("true_class", CLASSES),
        ("camera_angle", ANGLES),
        ("occlusion_type", OCCLUSIONS),
    ]:
        if not data[column].isin(allowed).all():
            raise ValueError(f"{column} 有空值或无法识别的值。")

    fold_table = pd.read_excel(FOLD_FILE, usecols=["filename", "fold"])
    feature_keys = image_keys(data["filename"])
    fold_keys = image_keys(fold_table["filename"])

    if (
        feature_keys.eq("").any()
        or not feature_keys.is_unique
        or not fold_keys.is_unique
        or set(feature_keys) != set(fold_keys)
    ):
        raise ValueError("特征表和原 RBF 预测表的文件名无法一一对应。")

    fold_map = pd.Series(fold_table["fold"].to_numpy(), index=fold_keys)
    data["fold"] = pd.to_numeric(feature_keys.map(fold_map), errors="raise")
    if not data["fold"].isin([1, 2, 3, 4]).all():
        raise ValueError("Fold 编号必须是 1、2、3、4。")
    data["fold"] = data["fold"].astype(int)

    strata = data["true_class"] + "|" + data["camera_angle"] + "|" + data["occlusion_type"]
    balance = pd.crosstab(strata, data["fold"])
    if balance.shape != (27, 4) or not balance.eq(1).all().all():
        raise ValueError("每个拍摄条件在每折必须恰好有一个测试样本。")

    return data


def evaluate_folds(data, use_pca):
    x = data[FEATURES].apply(pd.to_numeric).replace([np.inf, -np.inf], np.nan)
    usable = x.notna().any(axis=1)

    predictions = data[
        ["filename", "true_class", "camera_angle", "occlusion_type", "fold"]
    ].copy()
    predictions["predicted_class"] = pd.Series(pd.NA, index=data.index, dtype="string")
    predictions["unclassified_reason"] = ""
    predictions.loc[~usable, "unclassified_reason"] = "no_valid_features"
    fold_metrics = []

    for fold in range(1, 5):
        held_out = data["fold"].eq(fold)
        train = ~held_out & usable
        test = held_out & usable
        y_train = data.loc[train, "true_class"]

        train_accuracy = np.nan
        test_accuracy = np.nan
        dimensions = np.nan

        if y_train.nunique() < 2:
            predictions.loc[test, "unclassified_reason"] = "insufficient_training_classes"
        else:
            steps = [
                SimpleImputer(strategy="median", keep_empty_features=True),
                StandardScaler(),
            ]
            if use_pca:
                steps.append(PCA(n_components=0.95, svd_solver="full"))
            steps.append(SVC(kernel="linear", C=1.0))

            model = make_pipeline(*steps)
            model.fit(x.loc[train], y_train)
            train_accuracy = model.score(x.loc[train], y_train)
            dimensions = (
                model.named_steps["pca"].n_components_ if use_pca else len(FEATURES)
            )

            if test.any():
                predicted = model.predict(x.loc[test])
                predictions.loc[test, "predicted_class"] = predicted
                test_accuracy = (predicted == data.loc[test, "true_class"].to_numpy()).mean()

        n_classified = predictions.loc[held_out, "predicted_class"].notna().sum()
        fold_metrics.append({
            "fold": fold,
            "n_train": int(train.sum()),
            "n_test": int(held_out.sum()),
            "n_test_classified": int(n_classified),
            "svm_input_dimensions": dimensions,
            "train_accuracy": train_accuracy,
            "test_accuracy": test_accuracy,
        })

        print(
            f"Fold {fold}: 维数 {dimensions}，"
            f"训练准确率 {train_accuracy:.2%}，测试准确率 {test_accuracy:.2%}"
        )

    predictions["is_correct"] = predictions["predicted_class"].eq(predictions["true_class"])
    return predictions, pd.DataFrame(fold_metrics)


def summarize(rows, group_type, group, matrix_path):
    valid = rows.dropna(subset=["predicted_class"])
    n = len(valid)
    y_true = valid["true_class"]
    y_pred = valid["predicted_class"]

    matrix = confusion_matrix(y_true, y_pred, labels=CLASSES)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=CLASSES, zero_division=0
    )

    summary = {
        "group_type": group_type,
        "group": group,
        "n_total": len(rows),
        "n_used_for_metrics": n,
        "n_unclassified": len(rows) - n,
        "unclassified_rate": (len(rows) - n) / len(rows),
        "accuracy": matrix.trace() / n if n else np.nan,
        "macro_f1": f1.mean() if n else np.nan,
    }
    per_class = pd.DataFrame({
        "group_type": group_type,
        "group": group,
        "class": CLASSES,
        "precision": precision if n else np.nan,
        "recall": recall if n else np.nan,
        "f1_score": f1 if n else np.nan,
        "support": support,
    })

    display = ConfusionMatrixDisplay(matrix, display_labels=CLASSES)
    display.plot(cmap="Blues", colorbar=False, values_format="d")
    display.ax_.set_title(f"{group_type} {group} | {n}/{len(rows)} classified")
    display.figure_.savefig(matrix_path, dpi=160, bbox_inches="tight")
    plt.close(display.figure_)

    return summary, per_class


def save_results(predictions, fold_metrics, output):
    matrix_dir = output / "confusion_matrices"
    matrix_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_excel(output / "predictions.xlsx", index=False)

    overall, classes = summarize(predictions, "overall", "all", matrix_dir / "overall.png")
    overall["mean_train_accuracy"] = fold_metrics["train_accuracy"].mean()

    with pd.ExcelWriter(output / "overall_metrics.xlsx", engine="openpyxl") as writer:
        pd.DataFrame([overall]).to_excel(writer, sheet_name="overall", index=False)
        fold_metrics.to_excel(writer, sheet_name="fold_metrics", index=False)
        classes.to_excel(writer, sheet_name="per_class", index=False)

    summaries, class_tables = [], []
    for group_type, column, levels in [
        ("camera", "camera_angle", ANGLES),
        ("occlusion", "occlusion_type", OCCLUSIONS),
    ]:
        for level in levels:
            rows = predictions.loc[predictions[column].eq(level)]
            summary, by_class = summarize(
                rows, group_type, level, matrix_dir / f"{group_type}_{level}.png"
            )
            summaries.append(summary)
            class_tables.append(by_class)

    with pd.ExcelWriter(output / "group_metrics.xlsx", engine="openpyxl") as writer:
        pd.DataFrame(summaries).to_excel(writer, sheet_name="summary", index=False)
        pd.concat(class_tables, ignore_index=True).to_excel(
            writer, sheet_name="per_class", index=False
        )

    print(f"四折平均训练准确率：{overall['mean_train_accuracy']:.2%}")
    print(f"四折合并测试准确率：{overall['accuracy']:.2%}")
    print(f"结果目录：{output}")


def main():
    data = load_data()

    for name, use_pca in [("linear", False), ("linear_pca95", True)]:
        print(f"\n=== {name} ===")
        predictions, fold_metrics = evaluate_folds(data, use_pca)
        save_results(predictions, fold_metrics, OUTPUT_ROOT / name)


if __name__ == "__main__":
    main()
