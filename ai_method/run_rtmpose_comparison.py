from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rtmlib import Hand
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


PROJECT_DIR = Path(__file__).resolve().parents[1]
LABELS_PATH = PROJECT_DIR / "labels.csv"
OUTPUT_DIR = PROJECT_DIR / "ai_method/results"
OVERLAY_DIR = OUTPUT_DIR / "keypoint_overlays"
FEATURES_PATH = OUTPUT_DIR / "rtmpose_features.csv"
EVALUATION_PATH = OUTPUT_DIR / "rtmpose_evaluation.csv"
PREDICTIONS_PATH = OUTPUT_DIR / "rtmpose_predictions.csv"
CONFUSION_PATH = OUTPUT_DIR / "rtmpose_confusion.png"

CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]


def angle_at(a, b, c):
    first = a - b
    second = c - b
    denominator = np.linalg.norm(first) * np.linalg.norm(second)
    if denominator <= 1e-8:
        return np.nan
    cosine = np.clip(np.dot(first, second) / denominator, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def select_target(keypoints, scores):
    if len(keypoints) == 0:
        return None, None, -1
    best_index = 0
    best_value = -np.inf
    for index, (points, point_scores) in enumerate(zip(keypoints, scores)):
        width = float(points[:, 0].max() - points[:, 0].min())
        height = float(points[:, 1].max() - points[:, 1].min())
        # The recorded target hand is upright; an assisting/occluding hand is
        # usually horizontal. Prefer the instance whose wrist-to-middle-finger
        # direction is most vertical, while retaining area and confidence.
        direction = points[12] - points[0]
        verticality = abs(float(direction[1])) / max(
            float(np.linalg.norm(direction)), 1e-6
        )
        value = (
            width
            * height
            * max(float(point_scores.mean()), 1e-6)
            * (0.25 + verticality) ** 2
        )
        if value > best_value:
            best_value = value
            best_index = index
    return keypoints[best_index], scores[best_index], best_index


def normalize_keypoints(points, hand):
    wrist = points[0]
    centered = points - wrist
    palm_vector = centered[9]
    rotation = -np.pi / 2.0 - np.arctan2(palm_vector[1], palm_vector[0])
    matrix = np.array(
        [
            [np.cos(rotation), -np.sin(rotation)],
            [np.sin(rotation), np.cos(rotation)],
        ]
    )
    aligned = centered @ matrix.T
    scale_candidates = [
        np.linalg.norm(centered[index]) for index in [5, 9, 13, 17]
    ]
    scale = float(np.median(scale_candidates))
    if scale <= 1e-8:
        scale = 1.0
    aligned /= scale
    if hand == "right":
        aligned[:, 0] *= -1.0
    return aligned, scale


def keypoint_features(points, scores, hand):
    normalized, _ = normalize_keypoints(points, hand)
    features = {}
    for index, (x, y) in enumerate(normalized):
        features[f"kp_{index:02d}_x"] = float(x)
        features[f"kp_{index:02d}_y"] = float(y)
        features[f"score_{index:02d}"] = float(scores[index])

    features.update(
        {
            "mean_score": float(scores.mean()),
            "min_score": float(scores.min()),
            "low_score_count": int(np.sum(scores < 0.3)),
            "thumb_mcp_angle": angle_at(normalized[0], normalized[1], normalized[2]),
            "thumb_ip_angle": angle_at(normalized[2], normalized[3], normalized[4]),
            "thumb_index_spread": angle_at(normalized[4], normalized[0], normalized[8]),
            "thumb_index_tip_distance": float(np.linalg.norm(normalized[4] - normalized[8])),
            "thumb_middle_tip_distance": float(np.linalg.norm(normalized[4] - normalized[12])),
            "thumb_ring_tip_distance": float(np.linalg.norm(normalized[4] - normalized[16])),
            "thumb_pinky_tip_distance": float(np.linalg.norm(normalized[4] - normalized[20])),
            "thumb_index_mcp_distance": float(np.linalg.norm(normalized[4] - normalized[5])),
            "thumb_middle_mcp_distance": float(np.linalg.norm(normalized[4] - normalized[9])),
            "thumb_tip_radius": float(np.linalg.norm(normalized[4])),
            "index_flexion": angle_at(normalized[5], normalized[6], normalized[8]),
        }
    )
    return features


def draw_overlay(image, points, scores, selected_index, instance_count):
    result = image.copy()
    if points is None:
        cv2.putText(
            result,
            "RTMPose: no hand detected",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2,
        )
        return result
    for start, end in CONNECTIONS:
        if scores[start] >= 0.15 and scores[end] >= 0.15:
            cv2.line(
                result,
                tuple(np.round(points[start]).astype(int)),
                tuple(np.round(points[end]).astype(int)),
                (0, 255, 0),
                5,
            )
    for index, (point, score) in enumerate(zip(points, scores)):
        if score >= 0.15:
            color = (0, 0, 255) if index in [0, 4, 8] else (255, 0, 0)
            cv2.circle(result, tuple(np.round(point).astype(int)), 8, color, -1)
    cv2.putText(
        result,
        f"selected={selected_index + 1}/{instance_count} mean={scores.mean():.2f}",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 0),
        3,
    )
    return result


def extract_features():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OVERLAY_DIR.mkdir(parents=True, exist_ok=True)
    labels = pd.read_csv(LABELS_PATH)
    model = Hand(backend="onnxruntime", device="cpu")
    records = []

    for row_number, (_, row) in enumerate(labels.iterrows(), start=1):
        image = cv2.imread(str(PROJECT_DIR / row["path"]))
        if image is None:
            raise FileNotFoundError(row["path"])
        keypoints, scores = model(image)
        points, point_scores, selected_index = select_target(keypoints, scores)
        record = {
            "filename": row["filename"],
            "hand": row["hand"],
            "pose": row["pose"],
            "view": row["view"],
            "occlusion": row["occlusion"],
            "subject": int(row["index"]),
            "detected": points is not None,
            "instance_count": len(keypoints),
        }
        if points is not None:
            record.update(keypoint_features(points, point_scores, row["hand"]))
        records.append(record)

        overlay = draw_overlay(
            image, points, point_scores, selected_index, len(keypoints)
        )
        cv2.imwrite(
            str(OVERLAY_DIR / f"{Path(row['filename']).stem}_rtmpose.jpg"),
            overlay,
        )
        print(
            f"[{row_number:03d}/108] {row['filename']} "
            f"instances={len(keypoints)} "
            f"mean_score={point_scores.mean() if points is not None else 0:.3f}"
        )

    table = pd.DataFrame(records)
    table.to_csv(FEATURES_PATH, index=False)
    return table


def model_spec(name):
    if name == "rtmpose_keypoints_svm":
        pipeline = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("model", SVC(kernel="rbf")),
            ]
        )
        grid = {
            "model__C": [0.1, 1.0, 10.0, 100.0],
            "model__gamma": ["scale", 0.01, 0.1, 1.0],
        }
        return pipeline, grid
    if name == "rtmpose_keypoints_random_forest":
        pipeline = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=500,
                        max_features="sqrt",
                        min_samples_leaf=2,
                        class_weight="balanced",
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
        return pipeline, None
    if name == "rtmpose_keypoints_gradient_boosting":
        pipeline = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        max_iter=200,
                        l2_regularization=1.0,
                        random_state=42,
                    ),
                ),
            ]
        )
        return pipeline, None
    pipeline = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            (
                "model",
                DecisionTreeClassifier(
                    max_depth=4,
                    min_samples_leaf=5,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )
    return pipeline, None


def fit_model(x, y, pipeline, grid):
    if grid is None:
        return pipeline.fit(x, y)
    inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=17)
    return GridSearchCV(
        pipeline,
        grid,
        cv=inner,
        scoring="balanced_accuracy",
        n_jobs=-1,
    ).fit(x, y)


def evaluate_method(data, feature_columns, method):
    pipeline, grid = model_spec(method)
    records = []

    outer = StratifiedKFold(n_splits=6, shuffle=True, random_state=42)
    for fold, (train_idx, test_idx) in enumerate(
        outer.split(data[feature_columns], data["pose"]), start=1
    ):
        fitted = fit_model(
            data.iloc[train_idx][feature_columns],
            data.iloc[train_idx]["pose"],
            pipeline,
            grid,
        )
        predicted = fitted.predict(data.iloc[test_idx][feature_columns])
        for index, value in zip(test_idx, predicted):
            records.append(("stratified_6fold", fold, index, value))

    for subject in sorted(data["subject"].unique()):
        test_idx = np.flatnonzero(data["subject"].to_numpy() == subject)
        train_idx = np.flatnonzero(data["subject"].to_numpy() != subject)
        fitted = fit_model(
            data.iloc[train_idx][feature_columns],
            data.iloc[train_idx]["pose"],
            pipeline,
            grid,
        )
        predicted = fitted.predict(data.iloc[test_idx][feature_columns])
        for index, value in zip(test_idx, predicted):
            records.append(
                (f"train_other_test_subject_{subject}", subject, index, value)
            )

    holdouts = [
        ("train_none_test_paper", data["occlusion"] == "none", data["occlusion"] == "paper"),
        ("train_none_test_hand", data["occlusion"] == "none", data["occlusion"] == "hand"),
        ("holdout_front", data["view"] != "front", data["view"] == "front"),
        ("holdout_45deg", data["view"] != "45deg", data["view"] == "45deg"),
        ("holdout_90deg", data["view"] != "90deg", data["view"] == "90deg"),
    ]
    for protocol, train_rows, test_rows in holdouts:
        fitted = fit_model(
            data.loc[train_rows, feature_columns],
            data.loc[train_rows, "pose"],
            pipeline,
            grid,
        )
        predicted = fitted.predict(data.loc[test_rows, feature_columns])
        for index, value in zip(data.index[test_rows], predicted):
            records.append((protocol, 0, index, value))

    output = []
    for protocol, fold, index, predicted in records:
        row = data.loc[index]
        output.append(
            {
                "method": method,
                "protocol": protocol,
                "fold": fold,
                "filename": row["filename"],
                "subject": row["subject"],
                "true_pose": row["pose"],
                "predicted_pose": predicted,
                "correct": row["pose"] == predicted,
            }
        )
    return pd.DataFrame(output)


def main():
    data = extract_features()
    metadata = {
        "filename", "hand", "pose", "view", "occlusion", "subject",
        "detected", "instance_count",
    }
    feature_columns = [column for column in data.columns if column not in metadata]
    predictions = []
    methods = [
        "rtmpose_keypoints_svm",
        "rtmpose_keypoints_random_forest",
        "rtmpose_keypoints_gradient_boosting",
        "rtmpose_keypoints_decision_tree",
    ]
    for method in methods:
        predictions.append(evaluate_method(data, feature_columns, method))
    prediction_table = pd.concat(predictions, ignore_index=True)
    prediction_table.to_csv(PREDICTIONS_PATH, index=False)

    evaluations = []
    for (method, protocol), group in prediction_table.groupby(
        ["method", "protocol"], sort=False
    ):
        evaluations.append(
            {
                "method": method,
                "protocol": protocol,
                "sample_count": len(group),
                "accuracy": group["correct"].mean(),
            }
        )
    evaluation_table = pd.DataFrame(evaluations)
    evaluation_table.to_csv(EVALUATION_PATH, index=False)

    cv_rows = prediction_table[
        prediction_table["protocol"] == "stratified_6fold"
    ]
    figure, axes = plt.subplots(2, 2, figsize=(11, 10))
    for axis, method in zip(axes.flatten(), methods):
        rows = cv_rows[cv_rows["method"] == method]
        ConfusionMatrixDisplay.from_predictions(
            rows["true_pose"],
            rows["predicted_pose"],
            labels=["abduction", "adduction", "opposition"],
            display_labels=["ABD", "ADD", "OPP"],
            cmap="Purples",
            colorbar=False,
            ax=axis,
        )
        accuracy = accuracy_score(rows["true_pose"], rows["predicted_pose"])
        axis.set_title(f"{method}\naccuracy={accuracy:.3f}")
    figure.tight_layout()
    figure.savefig(CONFUSION_PATH, dpi=180, bbox_inches="tight")
    plt.close(figure)

    detected = int(data["detected"].sum())
    print(f"Detected hands: {detected}/{len(data)}")
    print(evaluation_table.to_string(index=False))
    print(f"Features: {FEATURES_PATH}")
    print(f"Evaluation: {EVALUATION_PATH}")


if __name__ == "__main__":
    main()
