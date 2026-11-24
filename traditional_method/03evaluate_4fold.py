from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.decomposition import PCA

SCRIPT_DIR = Path(__file__).resolve().parent
FEATURE_FILE = SCRIPT_DIR / "results" / "features.csv"
SHARED_FOLDS = None  # 或 SCRIPT_DIR.parent / "mediapipe_predictions.xlsx"
OUTPUT = SCRIPT_DIR / "traditional_pca_4fold_results"
SEED = 20261123

CLASSES = ["ABD", "ADD", "OPP"]
ANGLES = ["F", "45", "90"]
OCCLUSIONS = ["NONE", "HAND", "PAPER"]
FEATURES = [
    "normalized_area", "normalized_perimeter", "aspect_ratio",
    "extent", "solidity", "circularity",
    *[f"hu_{i}" for i in range(1, 8)],
]


def image_ids(names):
    # 本地和 GitHub 的部分扩展名不同，按文件主名对齐。
    return names.map(lambda name: Path(name.strip()).stem.casefold())


def load_data():
    data = pd.read_csv(FEATURE_FILE, dtype={"filename": str})
    data = data.rename(columns={
        "pose": "true_class",
        "view": "camera_angle",
        "occlusion": "occlusion_type",
    })
    data = data.sort_values(
        "filename", key=lambda s: s.str.casefold()
    ).reset_index(drop=True)

    for column, aliases, allowed in [
        ("true_class", {"ABDUCTION": "ABD", "ADDUCTION": "ADD", "OPPOSITION": "OPP"}, CLASSES),
        ("camera_angle", {"FRONT": "F", "45DEG": "45", "90DEG": "90"}, ANGLES),
        ("occlusion_type", {}, OCCLUSIONS),
    ]:
        data[column] = data[column].astype(str).str.strip().str.upper().replace(aliases)
        if not data[column].isin(allowed).all():
            raise ValueError(f"{column} 包含无法识别的值。")

    if data.filename.isna().any():
        raise ValueError("特征表有空文件名。")
    ids = image_ids(data.filename)
    if ids.eq("").any() or not ids.is_unique:
        raise ValueError("特征表的文件主名必须非空且唯一。")
    return data


def assign_folds(data):
    strata = data.true_class + "|" + data.camera_angle + "|" + data.occlusion_type
    counts = strata.value_counts()
    if len(counts) != 27 or not counts.eq(4).all():
        raise ValueError("预期有 27 个拍摄条件组，每组 4 个样本。")

    if SHARED_FOLDS is None:
        folds = pd.Series(0, index=data.index)
        cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=SEED)
        for fold, (_, test_idx) in enumerate(cv.split(data, strata), start=1):
            folds.iloc[test_idx] = fold
    else:
        if SHARED_FOLDS.suffix.lower() == ".xlsx":
            shared = pd.read_excel(SHARED_FOLDS)
        else:
            shared = pd.read_csv(SHARED_FOLDS)

        fold_column = "test_fold" if "test_fold" in shared else "fold"
        keys = image_ids(shared.filename)
        ids = image_ids(data.filename)
        if not keys.is_unique or set(keys) != set(ids):
            raise ValueError("共享 Fold 名单与特征表的文件主名不能一一对应。")
        fold_map = pd.Series(shared[fold_column].to_numpy(), index=keys)
        folds = pd.to_numeric(ids.map(fold_map), errors="raise")

    if not folds.isin([1, 2, 3, 4]).all():
        raise ValueError("Fold 编号必须为 1、2、3、4。")
    balance = pd.crosstab(strata, folds)
    if balance.shape != (27, 4) or not balance.eq(1).all().all():
        raise ValueError("每个拍摄条件组必须在每折恰好有一个测试样本。")
    return folds.astype(int)


def predict_out_of_fold(data):
    x = data[FEATURES].apply(pd.to_numeric).replace([np.inf, -np.inf], np.nan)
    usable = x.notna().any(axis=1)
    predictions = data[
        ["filename", "true_class", "camera_angle", "occlusion_type", "fold"]
    ].copy()
    predictions["predicted_class"] = pd.Series(pd.NA, index=data.index, dtype="string")
    predictions["unclassified_reason"] = ""
    predictions.loc[~usable, "unclassified_reason"] = "no_valid_features"

    for fold in range(1, 5):
        train = data.fold.ne(fold) & usable
        test = data.fold.eq(fold) & usable
        if data.loc[train, "true_class"].nunique() < 2:
            predictions.loc[test, "unclassified_reason"] = "insufficient_training_classes"
        elif test.any():
            # model = make_pipeline(
            #     SimpleImputer(strategy="median", keep_empty_features=True),
            #     StandardScaler(),
            #     SVC(kernel="rbf", C=1.0, gamma="scale"),
            # )
            model = make_pipeline(
                SimpleImputer(strategy="median", keep_empty_features=True),
                StandardScaler(),
                PCA(n_components=0.95, svd_solver="full"),
                SVC(kernel="rbf", C=1.0, gamma="scale"),
            )

            model.fit(x.loc[train], data.loc[train, "true_class"])
            predictions.loc[test, "predicted_class"] = model.predict(x.loc[test])

        print(f"Fold {fold}: train {train.sum()}/81, test {data.fold.eq(fold).sum()}")

    predictions["is_correct"] = predictions.predicted_class.eq(predictions.true_class)
    return predictions


def evaluate_group(rows, group_type, group, matrix_folder):
    valid = rows.dropna(subset=["predicted_class"])
    n = len(valid)
    matrix = confusion_matrix(valid.true_class, valid.predicted_class, labels=CLASSES)
    precision, recall, f1, support = precision_recall_fscore_support(
        valid.true_class, valid.predicted_class, labels=CLASSES, zero_division=0
    )

    summary = {
        "group_type": group_type,
        "group": group,
        "n_total": len(rows),
        "n_classified": n,
        "n_unclassified": len(rows) - n,
        "unclassified_rate": (len(rows) - n) / len(rows),
        "n_used_for_metrics": n,
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
    filename = "overall.png" if group_type == "overall" else f"{group_type}_{group}.png"
    display.figure_.savefig(matrix_folder / filename, dpi=160, bbox_inches="tight")
    plt.close(display.figure_)
    return summary, per_class


def main():
    data = load_data()
    data["fold"] = assign_folds(data)
    predictions = predict_out_of_fold(data)

    matrix_folder = OUTPUT / "confusion_matrices"
    matrix_folder.mkdir(parents=True, exist_ok=True)
    predictions.to_excel(OUTPUT / "predictions.xlsx", index=False)

    total, classes = evaluate_group(predictions, "overall", "all", matrix_folder)
    with pd.ExcelWriter(OUTPUT / "overall_metrics.xlsx", engine="openpyxl") as writer:
        pd.DataFrame([total]).to_excel(writer, sheet_name="overall", index=False)
        classes.to_excel(writer, sheet_name="per_class", index=False)

    summaries, class_tables = [], []
    for group_type, column, levels in [
        ("camera", "camera_angle", ANGLES),
        ("occlusion", "occlusion_type", OCCLUSIONS),
    ]:
        for level in levels:
            rows = predictions.loc[predictions[column].eq(level)]
            summary, per_class = evaluate_group(rows, group_type, level, matrix_folder)
            summaries.append(summary)
            class_tables.append(per_class)

    with pd.ExcelWriter(OUTPUT / "group_metrics.xlsx", engine="openpyxl") as writer:
        pd.DataFrame(summaries).to_excel(writer, sheet_name="summary", index=False)
        pd.concat(class_tables).to_excel(writer, sheet_name="per_class", index=False)

    print(f"Results: {OUTPUT.resolve()}")


if __name__ == "__main__":
    main()
