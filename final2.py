#!/usr/bin/env python3

from __future__ import annotations
from typing import Optional

from cv_bridge import CvBridge
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32


def detect_monitor(image):
    """
    Detect the monitor corners in the input BGR image.
    Returns: top_left, top_right, bottom_right, bottom_left
    """
    img_h, img_w = image.shape[:2]
    total_area = img_h * img_w

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 40, 120)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    edges_dilated = cv2.dilate(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(edges_dilated, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # 면적 필터링: 모니터가 화면의 5% ~ 75%를 차지한다고 가정
    area_filtered = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if (total_area * 0.05) <= area <= (total_area * 0.75):
            area_filtered.append((area, cnt))
    area_filtered.sort(key=lambda x: x[0], reverse=True)
    top_contours = area_filtered[:20]

    valid_candidates = []

    for area, cnt in top_contours:
        peri = cv2.arcLength(cnt, True)
        approx = None
        for eps in [0.02, 0.03]:
            candidate = cv2.approxPolyDP(cnt, eps * peri, True)
            if len(candidate) == 4 and cv2.isContourConvex(candidate):
                approx = candidate
                break

        if approx is None:
            continue

        # 모니터 베젤(검은 테두리) 검증
        border_pts = approx.reshape(4, 2).astype(float)
        t = np.linspace(0, 1, 15)
        all_samples = []
        for i in range(4):
            p1, p2 = border_pts[i], border_pts[(i + 1) % 4]
            xs = np.clip(np.round(p1[0] * (1 - t) + p2[0] * t).astype(int), 0, gray.shape[1] - 1)
            ys = np.clip(np.round(p1[1] * (1 - t) + p2[1] * t).astype(int), 0, gray.shape[0] - 1)
            all_samples.append(gray[ys, xs])
            
        if not all_samples or float(np.mean(np.concatenate(all_samples))) >= 100:
            continue

        valid_candidates.append((area, approx))

    # 예외 처리: 테두리 검증을 통과 못 한 경우 색상 상관없이 가장 큰 4각형 반환
    if not valid_candidates:
        for area, cnt in top_contours:
            peri = cv2.arcLength(cnt, True)
            for eps in [0.02, 0.03, 0.04, 0.05, 0.06]:
                candidate = cv2.approxPolyDP(cnt, eps * peri, True)
                if len(candidate) == 4 and cv2.isContourConvex(candidate):
                    valid_candidates.append((area, candidate))
                    break

    if not valid_candidates:
        return None, None, None, None

    # 가장 큰 사각형 영역 선택
    best_rect = max(valid_candidates, key=lambda item: item[0])[1]
    pts = best_rect.reshape(4, 2).astype("float32")
    rect = np.zeros((4, 2), dtype="float32")

    # 좌표 정렬 (좌상, 우상, 우하, 좌하)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1).flatten()
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect[0], rect[1], rect[2], rect[3]


def rectify_monitor(image, top_left, top_right, bottom_right, bottom_left):
    """
    Perspective-transform the detected monitor into a front-facing view (16:9).
    """
    if any(pt is None for pt in [top_left, top_right, bottom_right, bottom_left]):
        return None

    width_top = np.linalg.norm(top_right - top_left)
    width_bottom = np.linalg.norm(bottom_right - bottom_left)
    max_width = int(max(width_top, width_bottom))

    # 16:9 비율 강제 고정
    max_height = int(max_width * 9 / 16)

    src = np.array([top_left, top_right, bottom_right, bottom_left], dtype="float32")
    dst = np.array([[0, 0], [max_width - 1, 0], [max_width - 1, max_height - 1], [0, max_height - 1]], dtype="float32")

    transform = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(image, transform, (max_width, max_height))

    return warped


def detect_line(rectified):
    """
    LSD(Line Segment Detector) 기반 버전
    가장 긴 선분 하나를 반환
    """

    if rectified is None:
        return None

    gray = cv2.cvtColor(rectified, cv2.COLOR_BGR2GRAY)

    # 약한 노이즈 제거
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # LSD 생성
    lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)

    result = lsd.detect(gray)

    if result is None or result[0] is None:
        return None

    lines = result[0]

    max_length = 0
    best_line = None

    for line in lines:
        x1, y1, x2, y2 = line[0]

        dx = x2 - x1
        dy = y2 - y1

        length = np.hypot(dx, dy)

        # 현재 코드와 동일한 각도 체계
        angle = np.degrees(np.arctan2(dx, dy))

        if angle > 90:
            angle -= 180
        elif angle < -90:
            angle += 180

        # 수평/수직 제거
        if abs(angle) < 2.0 or abs(angle) > 88.0:
            continue

        if length > max_length:
            max_length = length
            best_line = (
                int(round(x1)),
                int(round(y1)),
                int(round(x2)),
                int(round(y2)),
            )

    return best_line


def calculate_angle(line) -> Optional[float]:
    """
    Calculate the line angle in degrees.
    """
    if line is None:
        return None

    x1, y1, x2, y2 = line
    dx, dy = x2 - x1, y2 - y1

    # 12시 방향 0도 기준, 시계방향(-), 반시계방향(+)
    angle = np.degrees(np.arctan2(dx, dy))
    if angle > 90:
        angle -= 180
    elif angle < -90:
        angle += 180

    return angle


class LineDetector(Node):
    def __init__(self) -> None:
        super().__init__("line_detector_node")

        self.declare_parameter("topic_image", "/camera/camera/color/image_raw")
        self.declare_parameter("topic_student", "/student/angle")

        topic_image = str(self.get_parameter("topic_image").value)
        topic_student = str(self.get_parameter("topic_student").value)

        self.bridge = CvBridge()

        self.image_sub = self.create_subscription(
            Image,
            topic_image,
            self.image_callback,
            10,
        )

        self.angle_pub = self.create_publisher(
            Float32,
            topic_student,
            10,
        )

        self.line_pub = self.create_publisher(
            Image,
            "/debug/line",
            10,
        )

        self.get_logger().info(
            f"Line detector started. Subscribing to {topic_image!r}, "
            f"publishing to {topic_student!r}."
        )

    def image_callback(self, msg: Image) -> None:
        try:
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:
            self.get_logger().warning(f"Failed to convert image: {exc!r}")
            return

        # 1. Detect monitor
        top_left, top_right, bottom_right, bottom_left = detect_monitor(image)
        if any(p is None for p in (top_left, top_right, bottom_right, bottom_left)):
            return

        # 2. Rectify monitor
        rectified = rectify_monitor(image, top_left, top_right, bottom_right, bottom_left)
        if rectified is None:
            return

        # 3. Detect line
        line = detect_line(rectified)
        if line is None:
            return
        self._debug_line(msg, rectified, line)

        # 4. Calculate and publish angle
        angle = calculate_angle(line)
        if angle is None:
            return

        angle_msg = Float32()
        angle_msg.data = float(angle)
        self.angle_pub.publish(angle_msg)
        
        self.get_logger().info(f"Detected Angle: {float(angle):.2f} deg")

    def _debug_line(self, msg, rectified, line) -> None:
        debug_line = rectified.copy()

        x1, y1, x2, y2 = line
        cv2.line(
            debug_line,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            (0, 0, 255),
            6,
        )

        debug_line_msg = self.bridge.cv2_to_imgmsg(debug_line, encoding="bgr8")
        debug_line_msg.header = msg.header
        self.line_pub.publish(debug_line_msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = LineDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
