from pathlib import Path

import cv2
import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
LABELS_PATH = PROJECT_DIR / "labels.csv"
MASK_DIR = PROJECT_DIR / "traditional_method/results/selected_masks"
OUTPUT_PATH = (
    PROJECT_DIR
    / "traditional_method/results/rule_based_evaluation.csv"
)

POSE_MAP = {
    "abduction": "ABD",
    "adduction": "ADD",
    "opposition": "OPP",
}


def analyze_image(filename):
    stem = Path(filename).stem
    mask_path = MASK_DIR / f"{stem}_selected_mask.png"

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    if mask is None:
        raise FileNotFoundError(mask_path)

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    contour = max(contours, key=cv2.contourArea)

    # 手掌中心和半径
    distance_map = cv2.distanceTransform(
        mask,
        cv2.DIST_L2,
        5,
    )

    _, palm_radius, _, palm_center = cv2.minMaxLoc(
        distance_map
    )

    # 凸缺陷
    hull_indices = cv2.convexHull(
        contour,
        returnPoints=False,
    )

    defects = cv2.convexityDefects(
        contour,
        hull_indices,
    )

    valid_defects = []
    wrist_limit = palm_center[1] + 0.3 * palm_radius

    if defects is not None:
        for start_i, end_i, far_i, depth in defects.reshape(-1, 4):
            start = tuple(contour[start_i][0])
            end = tuple(contour[end_i][0])
            far = tuple(contour[far_i][0])

            depth = depth / 256.0

            if depth >= 20 and far[1] < wrist_limit:
                valid_defects.append((start, end, far))

    # 如果没有找到指缝，暂时判断为OPP
    if not valid_defects:
        return {
            "valid_defect_count": 0,
            "palm_radius": palm_radius,
            "thumb_detected": False,
            "normalized_thumb_distance": np.nan,
            "thumb_index_angle": np.nan,
            "predicted_pose": "OPP",
            "status": "no_valid_defects",
        }

    hand_side = filename.split("_")[0]

    if hand_side == "L":
        thumb_defect = min(
            valid_defects,
            key=lambda item: item[2][0],
        )

        start, end, _ = thumb_defect
        thumb_tip = min((start, end), key=lambda p: p[0])
        index_tip = max((start, end), key=lambda p: p[0])

    else:
        thumb_defect = max(
            valid_defects,
            key=lambda item: item[2][0],
        )

        start, end, _ = thumb_defect
        thumb_tip = max((start, end), key=lambda p: p[0])
        index_tip = min((start, end), key=lambda p: p[0])

    thumb_vertical_distance = (
        palm_center[1] - thumb_tip[1]
    )

    thumb_detected = (
        thumb_vertical_distance <= 1.5 * palm_radius
    )

    if not thumb_detected:
        return {
            "valid_defect_count": len(valid_defects),
            "palm_radius": palm_radius,
            "thumb_detected": False,
            "normalized_thumb_distance": np.nan,
            "thumb_index_angle": np.nan,
            "predicted_pose": "OPP",
            "status": "thumb_rejected",
        }

    thumb_vector = (
        np.array(thumb_tip, dtype=float)
        - np.array(palm_center, dtype=float)
    )

    index_vector = (
        np.array(index_tip, dtype=float)
        - np.array(palm_center, dtype=float)
    )

    thumb_distance = np.linalg.norm(thumb_vector)
    normalized_distance = thumb_distance / palm_radius

    cos_angle = np.dot(thumb_vector, index_vector) / (
        np.linalg.norm(thumb_vector)
        * np.linalg.norm(index_vector)
    )

    angle = np.degrees(
        np.arccos(np.clip(cos_angle, -1.0, 1.0))
    )

    if normalized_distance > 1.6:
        predicted_pose = "ABD"
    else:
        predicted_pose = "ADD"

    return {
        "valid_defect_count": len(valid_defects),
        "palm_radius": palm_radius,
        "thumb_detected": True,
        "normalized_thumb_distance": normalized_distance,
        "thumb_index_angle": angle,
        "predicted_pose": predicted_pose,
        "status": "ok",
    }


def main():
    labels = pd.read_csv(LABELS_PATH)
    records = []

    for _, row in labels.iterrows():
        result = analyze_image(row["filename"])

        true_pose = POSE_MAP[row["pose"]]
        predicted_pose = result["predicted_pose"]

        records.append({
            "filename": row["filename"],
            "hand": row["hand"],
            "view": row["view"],
            "occlusion": row["occlusion"],
            "true_pose": true_pose,
            **result,
            "correct": true_pose == predicted_pose,
        })

        print(
            f"{row['filename']}: "
            f"{true_pose} -> {predicted_pose}"
        )

    evaluation = pd.DataFrame(records)
    evaluation.to_csv(OUTPUT_PATH, index=False)

    accuracy = evaluation["correct"].mean()

    print()
    print(f"Overall accuracy: {accuracy:.3f}")
    print()
    print(pd.crosstab(
        evaluation["true_pose"],
        evaluation["predicted_pose"],
    ))
    print()
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()