from pathlib import Path

import cv2
import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
LABELS_PATH = PROJECT_DIR / "labels.csv"

MASK_DIR = PROJECT_DIR / "traditional_method/results/masks"
OVERLAY_DIR = PROJECT_DIR / "traditional_method/results/overlays"
CONTOUR_DIR = PROJECT_DIR / "traditional_method/results/contours"
SELECTED_MASK_DIR = (
    PROJECT_DIR
    / "traditional_method/results/selected_masks"
)

MAX_COMPONENT_GAP = 50

MIN_CONTOUR_AREA = 1000
TARGET_HEIGHT = 1280


def resize_for_processing(image):
    height, width = image.shape[:2]

    if height == TARGET_HEIGHT:
        return image

    scale = TARGET_HEIGHT / height
    target_width = round(width * scale)

    return cv2.resize(
        image,
        (target_width, TARGET_HEIGHT),
        interpolation=cv2.INTER_AREA,
    )


def segment_skin(image):
    # 1. YCrCb肤色检测
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

    lower_skin = np.array([0, 133, 77], dtype=np.uint8)
    upper_skin = np.array([255, 173, 127], dtype=np.uint8)

    ycrcb_mask = cv2.inRange(
        ycrcb,
        lower_skin,
        upper_skin,
    )

    # 2. HSV饱和度检测
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # OpenCV中H范围是0-179，S和V范围是0-255
    # 只保留饱和度S大于等于50的像素
    lower_hsv = np.array([0, 50, 0], dtype=np.uint8)
    upper_hsv = np.array([179, 255, 255], dtype=np.uint8)

    hsv_mask = cv2.inRange(
        hsv,
        lower_hsv,
        upper_hsv,
    )

    # 3. 必须同时满足YCrCb和HSV条件
    mask = cv2.bitwise_and(
        ycrcb_mask,
        hsv_mask,
    )

    # 4. 去除小噪点
    open_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (5, 5),
    )
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        open_kernel,
    )

    # 5. 填补手部区域中的小缺口
    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (11, 11),
    )
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        close_kernel,
    )

    return mask

def create_overlay(image, mask):
    colored = image.copy()

    # 将检测到的区域标为红色
    colored[mask > 0] = (0, 0, 255)

    return cv2.addWeighted(
        image, 0.7, colored, 0.3, 0
    )

def bounding_box_gap(box1, box2):
    x1, y1, width1, height1 = box1
    x2, y2, width2, height2 = box2

    right1 = x1 + width1
    bottom1 = y1 + height1

    right2 = x2 + width2
    bottom2 = y2 + height2

    horizontal_gap = max(
        x1 - right2,
        x2 - right1,
        0,
    )

    vertical_gap = max(
        y1 - bottom2,
        y2 - bottom1,
        0,
    )

    return float(
        np.hypot(horizontal_gap, vertical_gap)
    )


def extract_and_draw_contours(image, mask):
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    valid_contours = [
        contour
        for contour in contours
        if cv2.contourArea(contour)
        >= MIN_CONTOUR_AREA
    ]

    valid_contours.sort(
        key=cv2.contourArea,
        reverse=True,
    )

    result = image.copy()
    selected_mask = np.zeros_like(mask)

    if not valid_contours:
        return result, selected_mask, [], []

    # C1始终作为主要轮廓保留
    primary_contour = valid_contours[0]
    primary_box = cv2.boundingRect(primary_contour)

    selected_contours = [primary_contour]
    rejected_contours = []

    # 判断C2、C3等轮廓与C1的距离
    for contour in valid_contours[1:]:
        contour_box = cv2.boundingRect(contour)

        gap = bounding_box_gap(
            primary_box,
            contour_box,
        )

        if gap <= MAX_COMPONENT_GAP:
            selected_contours.append(contour)
        else:
            rejected_contours.append(contour)

    # 将保留轮廓填充到新的黑白mask
    cv2.drawContours(
        selected_mask,
        selected_contours,
        -1,
        255,
        thickness=cv2.FILLED,
    )

    # 绿色和蓝框表示保留区域
    for index, contour in enumerate(
        selected_contours,
        start=1,
    ):
        area = cv2.contourArea(contour)
        x, y, width, height = cv2.boundingRect(
            contour
        )

        cv2.drawContours(
            result,
            [contour],
            -1,
            (0, 255, 0),
            3,
        )

        cv2.rectangle(
            result,
            (x, y),
            (x + width, y + height),
            (255, 0, 0),
            2,
        )

        cv2.putText(
            result,
            f"S{index}: {area:.0f}",
            (x, max(y - 10, 25)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

    # 红色表示被判定为背景的区域
    for index, contour in enumerate(
        rejected_contours,
        start=1,
    ):
        area = cv2.contourArea(contour)
        x, y, width, height = cv2.boundingRect(
            contour
        )

        cv2.drawContours(
            result,
            [contour],
            -1,
            (0, 0, 255),
            3,
        )

        cv2.rectangle(
            result,
            (x, y),
            (x + width, y + height),
            (0, 0, 255),
            2,
        )

        cv2.putText(
            result,
            f"R{index}: {area:.0f}",
            (x, max(y - 10, 25)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )

    return (
        result,
        selected_mask,
        selected_contours,
        rejected_contours,
    )

def main():
    MASK_DIR.mkdir(parents=True, exist_ok=True)
    OVERLAY_DIR.mkdir(parents=True, exist_ok=True)
    CONTOUR_DIR.mkdir(parents=True, exist_ok=True)
    SELECTED_MASK_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    labels = pd.read_csv(LABELS_PATH)

    print(f"Number of records: {len(labels)}")
    assert len(labels) == 108, "labels.csv should contain 108 images"

    for _, row in labels.iterrows():
        image_path = PROJECT_DIR / row["path"]
        image = cv2.imread(str(image_path))

        if image is None:
            raise FileNotFoundError(
                f"Cannot read image: {image_path}"
            )

        image = resize_for_processing(image)
        mask = segment_skin(image)
        overlay = create_overlay(image, mask)

        (
            contour_image,
            selected_mask,
            selected_contours,
            rejected_contours,
        ) = extract_and_draw_contours(
            image,
            mask,
        )

        filename_stem = Path(row["filename"]).stem

        mask_path = MASK_DIR / f"{filename_stem}_mask.png"
        overlay_path = OVERLAY_DIR / f"{filename_stem}_overlay.jpg"
        contour_path = (
                CONTOUR_DIR / f"{filename_stem}_contours.jpg"
        )
        selected_mask_path = (
                SELECTED_MASK_DIR
                / f"{filename_stem}_selected_mask.png"
        )

        cv2.imwrite(str(mask_path), mask)
        cv2.imwrite(str(overlay_path), overlay)
        cv2.imwrite(str(contour_path), contour_image)
        cv2.imwrite(
            str(selected_mask_path),
            selected_mask,
        )

        print(
            f"Processed: {row['filename']} "
            f"| selected: {len(selected_contours)} "
            f"| rejected: {len(rejected_contours)}"
        )

    print("All segmentation results were saved successfully.")
    print(f"Masks: {MASK_DIR}")
    print(f"Overlays: {OVERLAY_DIR}")
    print(f"Selected masks: {SELECTED_MASK_DIR}")
    print(f"Contours: {CONTOUR_DIR}")


if __name__ == "__main__":
    main()
