"""Tune the fixed segmentation parameters on the original 54-image dataset.

The experiment uses downstream repeated stratified six-fold accuracy as a proxy
because no manually annotated ground-truth masks are available.  It performs
three coordinate-search passes: vary one parameter, retain the best value, then
move to the next parameter.  Results are exploratory and selection-biased; they
must not be reported as an unbiased generalization estimate.
"""

from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from extract_features import extract_features


PROJECT_DIR = Path(__file__).resolve().parents[1]
LABELS_PATH = PROJECT_DIR / "labels.csv"
OUTPUT_DIR = PROJECT_DIR / "traditional_method/results_parameter_tuning"
TRIALS_PATH = OUTPUT_DIR / "segmentation_parameter_trials.csv"
ROUNDS_PATH = OUTPUT_DIR / "segmentation_parameter_rounds.csv"
BEST_PATH = OUTPUT_DIR / "segmentation_parameter_best.csv"

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

DEFAULTS = OrderedDict(
    [
        ("target_height", 1280),
        ("y_low", 0),
        ("y_high", 255),
        ("cr_low", 133),
        ("cr_high", 173),
        ("cb_low", 77),
        ("cb_high", 127),
        ("s_min", 50),
        ("open_size", 5),
        ("close_size", 11),
        ("min_area", 1000),
        ("max_gap", 50),
    ]
)

GRIDS = OrderedDict(
    [
        ("target_height", [640, 960, 1280, 1440, 1600, 1800]),
        ("y_low", [0, 20, 40, 60]),
        ("y_high", [180, 190, 200, 210, 220, 240, 255]),
        ("cr_low", [125, 129, 133, 137, 139, 141, 143, 145, 147]),
        ("cr_high", [165, 169, 173, 177, 181]),
        ("cb_low", [65, 71, 77, 83, 89]),
        ("cb_high", [119, 123, 127, 131, 135]),
        ("s_min", [20, 35, 50, 65, 80]),
        ("open_size", [1, 3, 5, 7, 9]),
        ("close_size", [1, 5, 9, 11, 15, 17, 19, 21, 23, 25, 31]),
        ("min_area", [250, 500, 1000, 2000, 4000]),
        ("max_gap", [0, 5, 10, 25, 50, 75, 100]),
    ]
)


def model():
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("svm", SVC(kernel="rbf", C=1.0, gamma="scale")),
        ]
    )


def bounding_box_gap(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    horizontal = max(x1 - (x2 + w2), x2 - (x1 + w1), 0)
    vertical = max(y1 - (y2 + h2), y2 - (y1 + h1), 0)
    return float(np.hypot(horizontal, vertical))


def select_components(mask, min_area, max_gap):
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    valid = [contour for contour in contours if cv2.contourArea(contour) >= min_area]
    valid.sort(key=cv2.contourArea, reverse=True)
    selected = np.zeros_like(mask)
    if not valid:
        return selected, 0

    primary = valid[0]
    primary_box = cv2.boundingRect(primary)
    retained = [primary]
    for contour in valid[1:]:
        if bounding_box_gap(primary_box, cv2.boundingRect(contour)) <= max_gap:
            retained.append(contour)
    cv2.drawContours(selected, retained, -1, 255, thickness=cv2.FILLED)
    return selected, len(retained)


class ImageCache:
    """Keep one resized colour-space dataset in memory at a time."""

    def __init__(self, labels):
        self.labels = labels
        self.height = None
        self.items = None

    def get(self, target_height):
        if self.height == target_height and self.items is not None:
            return self.items

        print(f"  loading colour channels at height={target_height}", flush=True)
        items = []
        for _, row in self.labels.iterrows():
            image = cv2.imread(str(PROJECT_DIR / row["path"]))
            if image is None:
                raise FileNotFoundError(row["path"])
            height, width = image.shape[:2]
            if height != target_height:
                scale = target_height / height
                image = cv2.resize(
                    image,
                    (round(width * scale), target_height),
                    interpolation=(
                        cv2.INTER_AREA if target_height < height else cv2.INTER_LINEAR
                    ),
                )
            ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            items.append(
                {
                    "filename": row["filename"],
                    "pose": row["pose"],
                    "view": row["view"],
                    "occlusion": row["occlusion"],
                    "y": ycrcb[:, :, 0],
                    "cr": ycrcb[:, :, 1],
                    "cb": ycrcb[:, :, 2],
                    "s": hsv[:, :, 1],
                }
            )
        self.height = target_height
        self.items = items
        return items


def segment(item, config):
    mask = (
        (item["y"] >= config["y_low"])
        & (item["y"] <= config["y_high"])
        & (item["cr"] >= config["cr_low"])
        & (item["cr"] <= config["cr_high"])
        & (item["cb"] >= config["cb_low"])
        & (item["cb"] <= config["cb_high"])
        & (item["s"] >= config["s_min"])
    ).astype(np.uint8) * 255

    if config["open_size"] > 1:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (config["open_size"], config["open_size"]),
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    if config["close_size"] > 1:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (config["close_size"], config["close_size"]),
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return select_components(mask, config["min_area"], config["max_gap"])


def configuration_key(config):
    return tuple((name, int(config[name])) for name in DEFAULTS)


def evaluate_configuration(config, cache):
    rows = []
    areas = []
    component_counts = []
    invalid = 0

    for item in cache.get(config["target_height"]):
        mask, component_count = segment(item, config)
        component_counts.append(component_count)
        areas.append(float(np.count_nonzero(mask)) / mask.size)
        if component_count == 0:
            invalid += 1
            continue
        features = extract_features(mask)
        rows.append(
            {
                "filename": item["filename"],
                "pose": item["pose"],
                "view": item["view"],
                "occlusion": item["occlusion"],
                **{column: features[column] for column in FEATURE_COLUMNS},
            }
        )

    if invalid:
        return {
            "cv_mean": np.nan,
            "cv_std": np.nan,
            "invalid_masks": invalid,
            "median_area_fraction": float(np.median(areas)),
            "median_components": float(np.median(component_counts)),
        }

    table = pd.DataFrame(rows)
    cv = RepeatedStratifiedKFold(
        n_splits=6,
        n_repeats=5,
        random_state=42,
    )
    scores = cross_val_score(
        model(),
        table[FEATURE_COLUMNS],
        table["pose"],
        cv=cv,
        scoring="accuracy",
        n_jobs=-1,
    )
    return {
        "cv_mean": float(scores.mean()),
        "cv_std": float(scores.std(ddof=1)),
        "invalid_masks": 0,
        "median_area_fraction": float(np.median(areas)),
        "median_components": float(np.median(component_counts)),
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    labels = pd.read_csv(LABELS_PATH)
    if len(labels) != 54:
        raise ValueError("This tuning experiment requires exactly 54 images")

    cache = ImageCache(labels)
    result_cache = {}
    trials = []
    rounds = []
    current = DEFAULTS.copy()

    def score(config):
        key = configuration_key(config)
        if key not in result_cache:
            result_cache[key] = evaluate_configuration(config, cache)
        return result_cache[key]

    baseline = score(current)
    print(
        f"Baseline mean={baseline['cv_mean']:.4f} std={baseline['cv_std']:.4f}",
        flush=True,
    )

    for pass_number in [1, 2, 3]:
        print(f"\n=== coordinate pass {pass_number} ===", flush=True)
        for parameter, values in GRIDS.items():
            candidates = []
            ordered_values = [current[parameter]] + [
                value for value in values if value != current[parameter]
            ]
            for value in ordered_values:
                candidate = current.copy()
                candidate[parameter] = value
                result = score(candidate)
                record = {
                    "pass": pass_number,
                    "parameter": parameter,
                    "candidate_value": value,
                    **candidate,
                    **result,
                }
                trials.append(record)
                candidates.append((candidate, result))
                print(
                    f"pass={pass_number} {parameter}={value} "
                    f"mean={result['cv_mean']:.4f} std={result['cv_std']:.4f} "
                    f"invalid={result['invalid_masks']}",
                    flush=True,
                )

            viable = [
                pair for pair in candidates if np.isfinite(pair[1]["cv_mean"])
            ]
            best_config, best_result = max(
                viable,
                # Ignore meaningless floating-point differences such as
                # 0.3962962962962962 versus 0.3962962962962963.  Candidate
                # order puts the current value first, so exact ties retain it.
                key=lambda pair: (
                    round(pair[1]["cv_mean"], 12),
                    -round(pair[1]["cv_std"], 12),
                ),
            )
            current = best_config.copy()
            rounds.append(
                {
                    "pass": pass_number,
                    "parameter": parameter,
                    "selected_value": current[parameter],
                    **current,
                    **best_result,
                }
            )
            print(
                f"  selected {parameter}={current[parameter]} "
                f"mean={best_result['cv_mean']:.4f}",
                flush=True,
            )

        pass_result = score(current)
        print(
            f"pass {pass_number} result mean={pass_result['cv_mean']:.4f} "
            f"std={pass_result['cv_std']:.4f}",
            flush=True,
        )

    final_result = score(current)
    pd.DataFrame(trials).to_csv(TRIALS_PATH, index=False)
    pd.DataFrame(rounds).to_csv(ROUNDS_PATH, index=False)
    best = pd.DataFrame(
        [
            {"configuration": "original", **DEFAULTS, **baseline},
            {"configuration": "coordinate_search_best", **current, **final_result},
        ]
    )
    best.to_csv(BEST_PATH, index=False)

    print("\n=== final comparison ===")
    print(best.to_string(index=False))
    print(f"Trials: {TRIALS_PATH}")
    print(f"Rounds: {ROUNDS_PATH}")
    print(f"Best: {BEST_PATH}")


if __name__ == "__main__":
    main()
