from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


PROJECT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_DIR / "traditional_method/results"
MASK_DIR = RESULTS_DIR / "selected_masks"
LABELS_PATH = PROJECT_DIR / "labels.csv"
BASE_FEATURES_PATH = RESULTS_DIR / "features.csv"

ENHANCED_FEATURES_PATH = RESULTS_DIR / "enhanced_features.csv"
EVALUATION_PATH = RESULTS_DIR / "enhanced_traditional_evaluation.csv"
PREDICTIONS_PATH = RESULTS_DIR / "enhanced_traditional_predictions.csv"
CONFUSION_PATH = RESULTS_DIR / "enhanced_traditional_confusion.png"

BASE_COLUMNS = [
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

GEOMETRY_COLUMNS = [
    "defect_count",
    "mean_defect_depth",
    "max_defect_depth",
    "palm_radius_ratio",
    "skeleton_length_ratio",
    "skeleton_endpoints",
    "skeleton_branches",
    "top_width_ratio",
    "middle_width_ratio",
    "bottom_width_ratio",
]


def normalize_mask(mask, hand, size=128):
    binary = np.where(mask > 0, 255, 0).astype(np.uint8)
    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        raise ValueError("No hand contour")

    points = np.vstack(contours)
    x, y, width, height = cv2.boundingRect(points)
    crop = binary[y : y + height, x : x + width]

    # Mirror right hands so both hands share one canonical orientation.
    if hand == "right":
        crop = cv2.flip(crop, 1)

    margin = 8
    available = size - 2 * margin
    scale = min(available / crop.shape[1], available / crop.shape[0])
    resized = cv2.resize(
        crop,
        (
            max(1, round(crop.shape[1] * scale)),
            max(1, round(crop.shape[0] * scale)),
        ),
        interpolation=cv2.INTER_NEAREST,
    )
    canvas = np.zeros((size, size), dtype=np.uint8)
    y0 = (size - resized.shape[0]) // 2
    x0 = (size - resized.shape[1]) // 2
    canvas[y0 : y0 + resized.shape[0], x0 : x0 + resized.shape[1]] = resized
    return canvas


def morphological_skeleton(binary):
    image = np.where(binary > 0, 255, 0).astype(np.uint8)
    skeleton = np.zeros_like(image)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    while cv2.countNonZero(image) > 0:
        eroded = cv2.erode(image, element)
        opened = cv2.dilate(eroded, element)
        skeleton = cv2.bitwise_or(skeleton, cv2.subtract(image, opened))
        image = eroded
    return skeleton


def neighbor_features(skeleton):
    pixels = (skeleton > 0).astype(np.uint8)
    neighbors = cv2.filter2D(
        pixels,
        cv2.CV_16S,
        np.ones((3, 3), dtype=np.uint8),
        borderType=cv2.BORDER_CONSTANT,
    ) - pixels
    endpoints = int(np.sum((pixels == 1) & (neighbors == 1)))
    branches = int(np.sum((pixels == 1) & (neighbors >= 3)))
    return endpoints, branches


def width_ratio(binary, row_fraction):
    row = binary[min(binary.shape[0] - 1, round(row_fraction * (binary.shape[0] - 1)))]
    return float(np.count_nonzero(row)) / binary.shape[1]


def compute_hog(binary, cell_size=16, bins=9):
    image = binary.astype(np.float32) / 255.0
    gradient_x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
    magnitude, angle = cv2.cartToPolar(
        gradient_x, gradient_y, angleInDegrees=True
    )
    angle %= 180.0
    bin_width = 180.0 / bins
    features = []
    for y in range(0, binary.shape[0], cell_size):
        for x in range(0, binary.shape[1], cell_size):
            cell_magnitude = magnitude[y : y + cell_size, x : x + cell_size]
            cell_angle = angle[y : y + cell_size, x : x + cell_size]
            histogram = np.zeros(bins, dtype=np.float32)
            indices = np.floor(cell_angle / bin_width).astype(int) % bins
            for bin_index in range(bins):
                histogram[bin_index] = cell_magnitude[indices == bin_index].sum()
            norm = np.linalg.norm(histogram)
            if norm > 0:
                histogram /= norm
            features.extend(histogram)
    return np.asarray(features, dtype=np.float32)


def extract_enhanced_features(mask, hand):
    normalized = normalize_mask(mask, hand)
    contours, _ = cv2.findContours(
        normalized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    contour = max(contours, key=cv2.contourArea)
    hull_indices = cv2.convexHull(contour, returnPoints=False)
    defects = (
        cv2.convexityDefects(contour, hull_indices)
        if hull_indices is not None and len(hull_indices) >= 4
        else None
    )

    depths = []
    if defects is not None:
        for _, _, _, depth in defects.reshape(-1, 4):
            normalized_depth = (depth / 256.0) / normalized.shape[0]
            if normalized_depth >= 0.015:
                depths.append(normalized_depth)

    distance = cv2.distanceTransform(normalized, cv2.DIST_L2, 5)
    palm_radius = float(distance.max()) / normalized.shape[0]

    skeleton = morphological_skeleton(normalized)
    endpoints, branches = neighbor_features(skeleton)
    skeleton_ratio = float(np.count_nonzero(skeleton)) / np.count_nonzero(normalized)

    hog = compute_hog(normalized)

    geometry = {
        "defect_count": len(depths),
        "mean_defect_depth": float(np.mean(depths)) if depths else 0.0,
        "max_defect_depth": float(np.max(depths)) if depths else 0.0,
        "palm_radius_ratio": palm_radius,
        "skeleton_length_ratio": skeleton_ratio,
        "skeleton_endpoints": endpoints,
        "skeleton_branches": branches,
        "top_width_ratio": width_ratio(normalized, 0.25),
        "middle_width_ratio": width_ratio(normalized, 0.50),
        "bottom_width_ratio": width_ratio(normalized, 0.75),
    }
    return geometry, hog


def build_feature_table():
    labels = pd.read_csv(LABELS_PATH)
    base = pd.read_csv(BASE_FEATURES_PATH)
    base = base.set_index("filename")
    records = []

    for _, row in labels.iterrows():
        stem = Path(row["filename"]).stem
        mask = cv2.imread(
            str(MASK_DIR / f"{stem}_selected_mask.png"),
            cv2.IMREAD_GRAYSCALE,
        )
        if mask is None:
            raise FileNotFoundError(row["filename"])

        geometry, hog = extract_enhanced_features(mask, row["hand"])
        record = {
            "filename": row["filename"],
            "hand": row["hand"],
            "pose": row["pose"],
            "view": row["view"],
            "occlusion": row["occlusion"],
            "subject": int(row["index"]),
        }
        record.update(base.loc[row["filename"], BASE_COLUMNS].to_dict())
        record.update(geometry)
        for index, value in enumerate(hog):
            record[f"hog_{index:04d}"] = float(value)
        records.append(record)

    table = pd.DataFrame(records)
    table.to_csv(ENHANCED_FEATURES_PATH, index=False)
    return table


def candidate_spec(name, hog_columns):
    if name == "baseline_fixed_svm":
        pipeline = Pipeline(
            [
                ("scale", StandardScaler()),
                ("svm", SVC(kernel="rbf", C=1.0, gamma="scale")),
            ]
        )
        return BASE_COLUMNS, pipeline, None

    if name == "enhanced_geometry_svm":
        pipeline = Pipeline(
            [
                ("scale", StandardScaler()),
                ("svm", SVC(kernel="rbf")),
            ]
        )
        grid = {
            "svm__C": [0.1, 1.0, 10.0, 100.0],
            "svm__gamma": ["scale", 0.01, 0.1, 1.0],
        }
        return BASE_COLUMNS + GEOMETRY_COLUMNS, pipeline, grid

    if name == "enhanced_geometry_random_forest":
        model = RandomForestClassifier(
            n_estimators=500,
            max_features="sqrt",
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        return BASE_COLUMNS + GEOMETRY_COLUMNS, model, None

    if name == "enhanced_geometry_gradient_boosting":
        model = HistGradientBoostingClassifier(
            max_iter=200,
            l2_regularization=1.0,
            random_state=42,
        )
        return BASE_COLUMNS + GEOMETRY_COLUMNS, model, None

    pipeline = Pipeline(
        [
            ("scale", StandardScaler()),
            ("pca", PCA(n_components=0.95, svd_solver="full")),
            ("svm", SVC(kernel="rbf")),
        ]
    )
    grid = {
        "svm__C": [0.1, 1.0, 10.0, 100.0],
        "svm__gamma": ["scale", 0.01, 0.1],
    }
    return BASE_COLUMNS + GEOMETRY_COLUMNS + hog_columns, pipeline, grid


def fit_model(x, y, pipeline, grid):
    if grid is None:
        return pipeline.fit(x, y)
    inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=17)
    search = GridSearchCV(
        pipeline,
        grid,
        cv=inner,
        scoring="balanced_accuracy",
        n_jobs=-1,
    )
    return search.fit(x, y)


def evaluate_candidate(data, name, columns, pipeline, grid):
    predictions = []
    protocols = []

    outer = StratifiedKFold(n_splits=6, shuffle=True, random_state=42)
    for fold, (train_idx, test_idx) in enumerate(
        outer.split(data[columns], data["pose"]), start=1
    ):
        fitted = fit_model(
            data.iloc[train_idx][columns],
            data.iloc[train_idx]["pose"],
            pipeline,
            grid,
        )
        predicted = fitted.predict(data.iloc[test_idx][columns])
        for index, value in zip(test_idx, predicted):
            predictions.append(("stratified_6fold", fold, index, value))

    for subject in sorted(data["subject"].unique()):
        test_idx = np.flatnonzero(data["subject"].to_numpy() == subject)
        train_idx = np.flatnonzero(data["subject"].to_numpy() != subject)
        fitted = fit_model(
            data.iloc[train_idx][columns],
            data.iloc[train_idx]["pose"],
            pipeline,
            grid,
        )
        predicted = fitted.predict(data.iloc[test_idx][columns])
        protocol = f"train_other_test_subject_{subject}"
        for index, value in zip(test_idx, predicted):
            predictions.append((protocol, subject, index, value))

    # Refit for the controlled occlusion and view holdouts.
    holdouts = [
        ("train_none_test_paper", data["occlusion"] == "none", data["occlusion"] == "paper"),
        ("train_none_test_hand", data["occlusion"] == "none", data["occlusion"] == "hand"),
        ("holdout_front", data["view"] != "front", data["view"] == "front"),
        ("holdout_45deg", data["view"] != "45deg", data["view"] == "45deg"),
        ("holdout_90deg", data["view"] != "90deg", data["view"] == "90deg"),
    ]
    for protocol, train_rows, test_rows in holdouts:
        fitted = fit_model(
            data.loc[train_rows, columns],
            data.loc[train_rows, "pose"],
            pipeline,
            grid,
        )
        predicted = fitted.predict(data.loc[test_rows, columns])
        for index, value in zip(data.index[test_rows], predicted):
            predictions.append((protocol, 0, index, value))

    prediction_records = []
    for protocol, fold, index, predicted in predictions:
        row = data.loc[index]
        prediction_records.append(
            {
                "method": name,
                "protocol": protocol,
                "fold": fold,
                "filename": row["filename"],
                "subject": row["subject"],
                "true_pose": row["pose"],
                "predicted_pose": predicted,
                "correct": row["pose"] == predicted,
            }
        )
    prediction_table = pd.DataFrame(prediction_records)

    for protocol, group in prediction_table.groupby("protocol", sort=False):
        protocols.append(
            {
                "method": name,
                "protocol": protocol,
                "sample_count": len(group),
                "accuracy": group["correct"].mean(),
            }
        )
    return pd.DataFrame(protocols), prediction_table


def main():
    data = build_feature_table()
    hog_columns = [column for column in data.columns if column.startswith("hog_")]

    evaluations = []
    predictions = []
    for method in [
        "baseline_fixed_svm",
        "enhanced_geometry_svm",
        "enhanced_geometry_random_forest",
        "enhanced_geometry_gradient_boosting",
        "enhanced_geometry_hog_svm",
    ]:
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

    cv_rows = prediction_table[
        prediction_table["protocol"] == "stratified_6fold"
    ]
    methods = list(evaluation_table["method"].unique())
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
        accuracy = accuracy_score(rows["true_pose"], rows["predicted_pose"])
        axis.set_title(f"{method}\naccuracy={accuracy:.3f}")
    for axis in axes[len(methods) :]:
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(CONFUSION_PATH, dpi=180, bbox_inches="tight")
    plt.close(figure)

    print(evaluation_table.to_string(index=False))
    print(f"Features: {ENHANCED_FEATURES_PATH}")
    print(f"Evaluation: {EVALUATION_PATH}")
    print(f"Predictions: {PREDICTIONS_PATH}")


if __name__ == "__main__":
    main()
