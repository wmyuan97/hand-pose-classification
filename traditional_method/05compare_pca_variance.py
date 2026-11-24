"""Compare PCA variance thresholds using the existing four test folds."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


PROJECT = Path(__file__).resolve().parents[1]
FEATURE_FILE = PROJECT / "traditional_method" / "results" / "features.csv"
FOLD_FILE = PROJECT / "traditional_4fold_results" / "predictions.xlsx"
VARIANCE_TARGETS = (0.85, 0.90, 0.95, 0.97, 0.99)
CLASSES = ["ABD", "ADD", "OPP"]
FEATURES = [
    "normalized_area", "normalized_perimeter", "aspect_ratio",
    "extent", "solidity", "circularity",
    *[f"hu_{i}" for i in range(1, 8)],
]


def load_data():
    data = pd.read_csv(FEATURE_FILE, dtype={"filename": str}).set_index("filename")
    fold_table = pd.read_excel(FOLD_FILE, usecols=["filename", "fold"]).set_index("filename")
    if (
        not data.index.is_unique
        or not fold_table.index.is_unique
        or data.index.hasnans
        or set(data.index) != set(fold_table.index)
    ):
        raise ValueError("特征表与原四折名单的文件名必须一一对应。")

    folds = pd.to_numeric(fold_table["fold"].reindex(data.index), errors="raise")
    if folds.value_counts(dropna=False).to_dict() != {1: 27, 2: 27, 3: 27, 4: 27}:
        raise ValueError("原名单必须包含 4 折，每折 27 个测试样本。")

    y = data["pose"].str.strip().str.upper().replace({
        "ABDUCTION": "ABD", "ADDUCTION": "ADD", "OPPOSITION": "OPP",
    })
    if not y.isin(CLASSES).all():
        raise ValueError("姿态标签必须是 ABD、ADD 或 OPP。")

    x = data[FEATURES].apply(pd.to_numeric).replace([np.inf, -np.inf], np.nan)
    if not x.notna().any(axis=1).all():
        raise ValueError("有样本缺少全部 13 个特征，无法比较同一批 108 个样本。")
    return x, y, folds


def evaluate(x, y, folds, variance):
    predictions = pd.Series(index=x.index, dtype="object")
    dimensions = []
    train_accuracies = []

    for fold in (1, 2, 3, 4):
        test = folds.eq(fold)
        train = ~test
        model = make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            StandardScaler(),
            PCA(n_components=variance, svd_solver="full"),
            SVC(kernel="linear", C=1.0),
        )
        # 整个流程只用当前折的训练样本拟合。
        model.fit(x.loc[train], y.loc[train])
        dimensions.append(model.named_steps["pca"].n_components_)
        train_accuracies.append(model.score(x.loc[train], y.loc[train]))
        predictions.loc[test] = model.predict(x.loc[test])

    correct = int(y.eq(predictions).sum())
    return {
        "PCA": f"{variance:.0%}",
        "Fold dimensions": "/".join(map(str, dimensions)),
        "Mean train accuracy": f"{np.mean(train_accuracies):.2%}",
        "Test correct": f"{correct}/{len(y)}",
        "Test accuracy": f"{correct / len(y):.2%}",
        "Test Macro-F1": f"{f1_score(y, predictions, labels=CLASSES, average='macro'):.2%}",
    }


def main():
    x, y, folds = load_data()
    results = [evaluate(x, y, folds, variance) for variance in VARIANCE_TARGETS]
    print(pd.DataFrame(results).to_string(index=False))


if __name__ == "__main__":
    main()
