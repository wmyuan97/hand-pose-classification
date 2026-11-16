from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_DIR / "traditional_method/results"

PREDICTIONS_PATH = RESULTS_DIR / "svm_predictions.csv"
LABELS_PATH = PROJECT_DIR / "labels.csv"
MASK_DIR = RESULTS_DIR / "selected_masks"

OUTPUT_PATH = RESULTS_DIR / "svm_cv_errors.png"


def main():
    predictions = pd.read_csv(PREDICTIONS_PATH)
    labels = pd.read_csv(LABELS_PATH)

    # 只选择6折交叉验证中预测错误的图片
    errors = predictions[
        (predictions["protocol"] == "stratified_6fold")
        & (~predictions["correct"])
    ].copy()

    # 根据文件名查找原图路径
    path_map = labels.set_index("filename")["path"].to_dict()

    print(f"Number of errors: {len(errors)}")

    columns = 4
    rows = (len(errors) + columns - 1) // columns

    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(20, rows * 4),
    )

    axes = axes.flatten()

    for axis, (_, row) in zip(axes, errors.iterrows()):
        filename = row["filename"]
        image_path = PROJECT_DIR / path_map[filename]

        stem = Path(filename).stem
        mask_path = MASK_DIR / f"{stem}_selected_mask.png"

        image = cv2.imread(str(image_path))
        mask = cv2.imread(str(mask_path))

        if image is None or mask is None:
            raise FileNotFoundError(filename)

        # 把原图和mask缩放到相同大小
        image = cv2.resize(image, (300, 400))
        mask = cv2.resize(mask, (300, 400))

        # 左边原图，右边mask
        comparison = cv2.hconcat([image, mask])
        comparison = cv2.cvtColor(comparison, cv2.COLOR_BGR2RGB)

        axis.imshow(comparison)
        axis.set_title(
            f"{filename}\n"
            f"True: {row['true_pose']}   "
            f"Predicted: {row['predicted_pose']}",
            fontsize=9,
        )
        axis.axis("off")

    # 隐藏没有使用的格子
    for axis in axes[len(errors):]:
        axis.axis("off")

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=150)
    plt.close()

    print(f"Error figure saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()