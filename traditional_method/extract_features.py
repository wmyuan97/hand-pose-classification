from pathlib import Path

import cv2
import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]

LABELS_PATH = PROJECT_DIR / "labels.csv"

SELECTED_MASK_DIR = (
    PROJECT_DIR
    / "traditional_method/results/selected_masks"
)

FEATURES_PATH = (
    PROJECT_DIR
    / "traditional_method/results/features.csv"
)


def extract_features(mask):
    # 确保mask只有0和255
    binary_mask = np.where(
        mask > 0,
        255,
        0,
    ).astype(np.uint8)

    contours, _ = cv2.findContours(
        binary_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if not contours:
        raise ValueError("No contour found in mask")

    # 将所有保留轮廓的坐标合并
    all_points = np.vstack(contours)

    image_height, image_width = binary_mask.shape

    # 所有轮廓的总面积和总周长
    area = sum(
        cv2.contourArea(contour)
        for contour in contours
    )

    perimeter = sum(
        cv2.arcLength(contour, True)
        for contour in contours
    )

    # 包围所有轮廓的整体矩形
    x, y, width, height = cv2.boundingRect(
        all_points
    )

    bounding_box_area = width * height

    aspect_ratio = (
        width / height
        if height > 0
        else 0.0
    )

    extent = (
        area / bounding_box_area
        if bounding_box_area > 0
        else 0.0
    )

    # 所有轮廓的整体凸包
    convex_hull = cv2.convexHull(all_points)
    hull_area = cv2.contourArea(convex_hull)

    solidity = (
        area / hull_area
        if hull_area > 0
        else 0.0
    )

    circularity = (
        4 * np.pi * area / (perimeter ** 2)
        if perimeter > 0
        else 0.0
    )

    normalized_area = (
        area / (image_width * image_height)
    )

    normalized_perimeter = (
        perimeter
        / np.sqrt(image_width * image_height)
    )

    # 使用整个二值mask计算Hu moments
    moments = cv2.moments(
        binary_mask,
        binaryImage=True,
    )

    hu_moments = cv2.HuMoments(
        moments
    ).flatten()

    # 对数转换，使Hu数值更容易比较
    hu_log = (
        -np.sign(hu_moments)
        * np.log10(np.abs(hu_moments) + 1e-12)
    )

    features = {
        "component_count": len(contours),
        "area": area,
        "normalized_area": normalized_area,
        "perimeter": perimeter,
        "normalized_perimeter": normalized_perimeter,
        "width": width,
        "height": height,
        "aspect_ratio": aspect_ratio,
        "extent": extent,
        "hull_area": hull_area,
        "solidity": solidity,
        "circularity": circularity,
    }

    for index, value in enumerate(
        hu_log,
        start=1,
    ):
        features[f"hu_{index}"] = value

    return features


def main():
    labels = pd.read_csv(LABELS_PATH)

    records = []

    for _, row in labels.iterrows():
        filename_stem = Path(
            row["filename"]
        ).stem

        mask_path = (
            SELECTED_MASK_DIR
            / f"{filename_stem}_selected_mask.png"
        )

        mask = cv2.imread(
            str(mask_path),
            cv2.IMREAD_GRAYSCALE,
        )

        if mask is None:
            raise FileNotFoundError(
                f"Cannot read mask: {mask_path}"
            )

        features = extract_features(mask)

        record = {
            "filename": row["filename"],
            "hand": row["hand"],
            "pose": row["pose"],
            "view": row["view"],
            "occlusion": row["occlusion"],
        }

        record.update(features)
        records.append(record)

        print(
            f"Extracted: {row['filename']} "
            f"| components: "
            f"{features['component_count']}"
        )

    features_table = pd.DataFrame(records)

    features_table.to_csv(
        FEATURES_PATH,
        index=False,
    )

    print(
        f"Saved {len(features_table)} records"
    )
    print(f"Features: {FEATURES_PATH}")


if __name__ == "__main__":
    main()