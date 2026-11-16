from pathlib import Path

import cv2
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]

# IMAGE_NAME = "L_ABD_F_NONE_01"
# IMAGE_NAME = "L_ADD_F_NONE_01"
# IMAGE_NAME = "L_OPP_F_NONE_01"
# IMAGE_NAME = "R_ABD_F_NONE_01"
# IMAGE_NAME = "R_ADD_F_NONE_01"
IMAGE_NAME = "R_OPP_F_NONE_01"

MASK_PATH = (
    PROJECT_DIR
    / "traditional_method/results/selected_masks"
    / f"{IMAGE_NAME}_selected_mask.png"
)

OUTPUT_PATH = (
    PROJECT_DIR
    / "traditional_method/results"
    / f"{IMAGE_NAME}_geometry.jpg"
)


def main():
    mask = cv2.imread(str(MASK_PATH), cv2.IMREAD_GRAYSCALE)

    if mask is None:
        raise FileNotFoundError(MASK_PATH)

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    # 选择面积最大的轮廓
    contour = max(contours, key=cv2.contourArea)

    # 计算凸包
    hull_points = cv2.convexHull(contour)

    # 创建用于显示结果的彩色图片
    result = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    # 计算每个白色像素到黑色边界的距离
    distance_map = cv2.distanceTransform(
        mask,
        cv2.DIST_L2,
        5,
    )

    # 距离边界最远的位置，近似作为手掌中心
    _, palm_radius, _, palm_center = cv2.minMaxLoc(distance_map)

    palm_radius = int(palm_radius)

    # 紫色圆点：手掌中心
    cv2.circle(
        result,
        palm_center,
        10,
        (255, 0, 255),
        -1,
    )

    # 紫色圆圈：手掌的大致范围
    cv2.circle(
        result,
        palm_center,
        palm_radius,
        (255, 0, 255),
        3,
    )

    print(f"Palm center: {palm_center}")
    print(f"Palm radius: {palm_radius}")

    # 黄色：真实手部轮廓
    cv2.drawContours(result, [contour], -1, (0, 255, 255), 2)

    # 绿色：凸包，也就是“橡皮筋”
    cv2.drawContours(result, [hull_points], -1, (0, 255, 0), 3)

    # 计算凸缺陷需要凸包点的编号
    hull_indices = cv2.convexHull(
        contour,
        returnPoints=False,
    )

    defects = cv2.convexityDefects(contour, hull_indices)

    valid_defects = []
    if defects is not None:
        for start_index, end_index, far_index, depth in defects.reshape(-1, 4):

            start = tuple(contour[start_index][0])
            end = tuple(contour[end_index][0])
            far = tuple(contour[far_index][0])

            depth = depth / 256.0

            # 只显示比较明显的凹陷
            wrist_limit = palm_center[1] + 0.3 * palm_radius

            if depth >= 20 and far[1] < wrist_limit:
                # 蓝点：凸包两端
                cv2.circle(result, start, 7, (255, 0, 0), -1)
                cv2.circle(result, end, 7, (255, 0, 0), -1)

                # 红点：凹陷最深的位置
                cv2.circle(result, far, 8, (0, 0, 255), -1)

                valid_defects.append((start, end, far))
                print(f"Defect depth: {depth:.1f}")

    if not valid_defects:
        raise RuntimeError("No valid finger gaps were detected.")

    # # 左手：选择最靠左的指缝作为拇指和食指之间的指缝
    # thumb_defect = min(
    #     valid_defects,
    #     key=lambda item: item[2][0],
    # )
    #
    # start, end, thumb_web = thumb_defect
    #
    # # 左侧端点是拇指尖，右侧端点是食指尖
    # thumb_tip = min((start, end), key=lambda point: point[0])
    # index_tip = max((start, end), key=lambda point: point[0])

    hand_side = IMAGE_NAME.split("_")[0]

    if hand_side == "L":
        # 左手数据中，拇指位于图片左侧
        thumb_defect = min(
            valid_defects,
            key=lambda item: item[2][0],
        )

        start, end, thumb_web = thumb_defect

        thumb_tip = min((start, end), key=lambda point: point[0])
        index_tip = max((start, end), key=lambda point: point[0])

    elif hand_side == "R":
        # 右手数据中，拇指位于图片右侧
        thumb_defect = max(
            valid_defects,
            key=lambda item: item[2][0],
        )

        start, end, thumb_web = thumb_defect

        thumb_tip = max((start, end), key=lambda point: point[0])
        index_tip = min((start, end), key=lambda point: point[0])

    else:
        raise ValueError(f"Unknown hand side: {hand_side}")

    # # 计算两个方向向量
    # thumb_vector = np.array(thumb_tip) - np.array(palm_center)
    # index_vector = np.array(index_tip) - np.array(palm_center)
    #
    # cos_angle = np.dot(thumb_vector, index_vector) / (
    #         np.linalg.norm(thumb_vector)
    #         * np.linalg.norm(index_vector)
    # )
    #
    # angle = np.degrees(
    #     np.arccos(np.clip(cos_angle, -1.0, 1.0))
    # )
    #
    # # 画出手掌中心到拇指尖和食指尖的线
    # cv2.line(result, palm_center, thumb_tip, (255, 255, 0), 4)
    # cv2.line(result, palm_center, index_tip, (255, 255, 0), 4)
    #
    # cv2.circle(result, thumb_tip, 12, (0, 165, 255), -1)
    # cv2.circle(result, index_tip, 12, (0, 165, 255), -1)
    #
    # print(f"Thumb tip: {thumb_tip}")
    # print(f"Index tip: {index_tip}")
    # print(f"Thumb-index angle: {angle:.1f} degrees")

    # 检查候选拇指是否过于靠上
    thumb_vertical_distance = palm_center[1] - thumb_tip[1]

    thumb_detected = (
            thumb_vertical_distance <= 1.5 * palm_radius
    )

    print(f"Thumb vertical distance: {thumb_vertical_distance}")
    print(f"Thumb detected: {thumb_detected}")

    if thumb_detected:
        # 计算两个方向向量
        thumb_vector = (
                np.array(thumb_tip) - np.array(palm_center)
        )
        index_vector = (
                np.array(index_tip) - np.array(palm_center)
        )
        thumb_distance = np.linalg.norm(thumb_vector)
        normalized_thumb_distance = thumb_distance / palm_radius

        print(f"Normalized thumb distance: {normalized_thumb_distance:.2f}")

        if normalized_thumb_distance > 1.6:
            predicted_pose = "ABD"
        else:
            predicted_pose = "ADD"

        cos_angle = np.dot(thumb_vector, index_vector) / (
                np.linalg.norm(thumb_vector)
                * np.linalg.norm(index_vector)
        )

        angle = np.degrees(
            np.arccos(np.clip(cos_angle, -1.0, 1.0))
        )

        cv2.line(
            result,
            palm_center,
            thumb_tip,
            (255, 255, 0),
            4,
        )
        cv2.line(
            result,
            palm_center,
            index_tip,
            (255, 255, 0),
            4,
        )

        cv2.circle(result, thumb_tip, 12, (0, 165, 255), -1)
        cv2.circle(result, index_tip, 12, (0, 165, 255), -1)

        print(f"Thumb tip: {thumb_tip}")
        print(f"Index tip: {index_tip}")
        print(f"Thumb-index angle: {angle:.1f} degrees")

    else:
        # 红色叉号表示这个点不是真正的拇指
        cv2.drawMarker(
            result,
            thumb_tip,
            (0, 0, 255),
            cv2.MARKER_TILTED_CROSS,
            30,
            4,
        )
        predicted_pose = "OPP"

        print("Thumb tip was not reliably detected.")
        print("Thumb-index angle: not available")

    print(f"Rule-based prediction: {predicted_pose}")
    cv2.imwrite(str(OUTPUT_PATH), result)
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()