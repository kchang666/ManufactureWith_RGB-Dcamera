"""V4：多點辨識、固定標記、滑鼠排序與開放折線加工。

啟動仍會移至 Initial Point。穩定辨識後鎖定標記；點選至少兩點，按 Start/S 加工。
U 復原、C 清空、R 重新偵測、P 印出 TCP、Q/Esc 離開。
位置或點數改變時停止接受舊點位，必須 R 重新偵測。每次啟動只加工一輪。
"""
#手臂到初始點位後偵測
import sys
import msvcrt
import threading
import time
import json
import textwrap
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs
from pymodbus.client import ModbusTcpClient
from pymodbus.payload import BinaryPayloadBuilder
from pymodbus.constants import Endian
from drv_modbus import request


WINDOW_NAME = "RealSense V4 - Select Machining Order"
ROBOT_HOST = "192.168.1.1"
ROBOT_PORT = 502
POSE_POLL_INTERVAL = 0.2
ARUCO_DICTIONARY = cv2.aruco.DICT_5X5_100
INITIAL_POINT = (393.085, 5.232, 730.939, -179.366, 1.545, -105.652)
INITIAL_SPEED = 20
INITIAL_MOVE_TIMEOUT = 60
ARUCO_TCP_POSES = {
    1: (367.000, -220.000, 463.084, -179.366, 1.545, -105.652),
    2: (367.000, 77.000, 463.084, -179.366, 1.545, -105.652),
    3: (614.000, -220.000, 463.084, -179.366, 1.545, -105.652),
    4: (614.000, 77.000, 463.084, -179.366, 1.545, -105.652),
}
CALIBRATION_SAMPLES = 30
MAX_CENTER_STD_PX = 1.5
CALIBRATION_PATH = Path(__file__).with_name("aruco_plane_calibration_v4.json")
DOT_TARGET_PATH = Path(__file__).with_name("selected_dot_targets_v4.json")
DOT_CONTACT_Z = 463.084
DOT_TRAVEL_Z = 500.0
DOT_MOVE_SPEED = 20
DOT_CONTACT_SPEED = 10
DOT_DWELL = 1.0
DOT_STABLE_FRAMES = 20
DOT_MAX_JITTER_PX = 2.0
# 黑色實心圓點／白紙的辨識門檻，可依實際光線及圓點大小調整。
DOT_DARK_THRESHOLD = 85
DOT_MIN_AREA_PX = 20
DOT_MAX_AREA_PX = 1500
PAPER_MIN_VALUE = 155
PAPER_MAX_SATURATION = 80
PANEL_WIDTH = 430
CLICK_RADIUS_PX = 20


def detect_paper_dots(image, calibration, detect_markers):
    """在四個標記包圍範圍內找白紙，排除 ArUco，再找黑色實心圓點。"""
    if calibration.homography is None or calibration.pixel_centers is None:
        return None, [], "Waiting for calibration"
    workspace = np.rint([calibration.pixel_centers[i] for i in (1, 2, 4, 3)]).astype(np.int32)
    workspace_mask = np.zeros(image.shape[:2], np.uint8)
    cv2.fillConvexPoly(workspace_mask, workspace, 255)
    workspace_area = cv2.countNonZero(workspace_mask)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    white = cv2.inRange(hsv, (0, 0, PAPER_MIN_VALUE), (180, PAPER_MAX_SATURATION, 255))
    white = cv2.bitwise_and(white, workspace_mask)
    white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    white = cv2.bitwise_and(white, workspace_mask)
    contours, _ = cv2.findContours(white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    papers = []
    for contour in contours:
        area = cv2.contourArea(contour)
        # 整個工作範圍都是白色時不能可靠區分紙張，不直接猜測紙張位置。
        if not 0.03 * workspace_area < area < 0.9 * workspace_area:
            continue
        outline = cv2.approxPolyDP(contour, 0.025 * cv2.arcLength(contour, True), True)
        if len(outline) == 4 and cv2.isContourConvex(outline):
            papers.append((area, outline))
    if not papers:
        return None, [], "Need a visible white sheet inside ArUco area"
    paper = max(papers, key=lambda item: item[0])[1]
    paper_mask = np.zeros(image.shape[:2], np.uint8)
    cv2.fillConvexPoly(paper_mask, paper, 255)
    paper_mask = cv2.erode(paper_mask, np.ones((11, 11), np.uint8))
    paper_mask = cv2.bitwise_and(paper_mask, workspace_mask)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    marker_corners, _, _ = detect_markers(gray)
    excluded = np.zeros_like(gray)
    for corners in marker_corners:
        cv2.fillConvexPoly(excluded, np.rint(corners.reshape(4, 2)).astype(np.int32), 255)
    excluded = cv2.dilate(excluded, np.ones((15, 15), np.uint8))
    paper_mask[excluded > 0] = 0
    _, dark = cv2.threshold(gray, DOT_DARK_THRESHOLD, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    dots = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if not DOT_MIN_AREA_PX <= area <= DOT_MAX_AREA_PX:
            continue
        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0 or 4 * np.pi * area / perimeter ** 2 < 0.65:
            continue
        hull_area = cv2.contourArea(cv2.convexHull(contour))
        if hull_area <= 0 or area / hull_area < 0.85:
            continue
        x, y, width, height = cv2.boundingRect(contour)
        if not 0.5 <= width / height <= 2:
            continue
        component_mask = np.zeros_like(gray)
        cv2.drawContours(component_mask, [contour], -1, 255, -1)
        # 整個候選點都必須位於紙內，避免紙邊及被裁斷的 ArUco 成為目標。
        if np.any((component_mask > 0) & (paper_mask == 0)):
            continue
        # 排除空心圓等只有黑色外框的形狀。
        if np.mean(gray[component_mask > 0] <= DOT_DARK_THRESHOLD) < 0.8:
            continue
        moments = cv2.moments(contour)
        if moments['m00']:
            dots.append((moments['m10'] / moments['m00'], moments['m01'] / moments['m00']))
    dots.sort()  # 初次依畫面左到右編號；後續用最小位移追蹤配對。
    return paper, dots, f"Detected: {len(dots)} black dots"


def match_points(reference, dots):
    """以距離門檻及唯一最近點配對，O(N²)，不枚舉所有排列。

不明確的配對直接拒絕，避免把近距離加工點交換身份。
"""
    points = np.asarray(dots, dtype=np.float64).reshape(-1, 2)
    if len(points) != len(reference) or not len(points) or not np.isfinite(points).all():
        return None
    distances = np.linalg.norm(np.asarray(reference)[:, None, :] - points[None, :, :], axis=2)
    candidates = distances <= DOT_MAX_JITTER_PX
    if not np.all(candidates.sum(axis=1) == 1) or not np.all(candidates.sum(axis=0) == 1):
        return None
    return points[np.argmax(candidates, axis=1)]


def point_label(index):
    """零起算索引轉 A...Z、AA...，點數不受 26 限制。"""
    result = ''
    index += 1
    while index:
        index, remainder = divmod(index - 1, 26)
        result = chr(65 + remainder) + result
    return result


class MultiDotTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.samples = []
        self.pixel_points = None
        self.valid = False
        self.invalidated = False
        self.detected_count = 0
        self.status = "Waiting for stable black dots"

    def update(self, dots, valid_pose):
        self.detected_count = len(dots)
        self.valid = False
        if self.invalidated:
            return
        if self.pixel_points is not None:
            if not valid_pose or match_points(self.pixel_points, dots) is None:
                self.invalidated = True
                self.status = "Scene changed / pose lost: press R to re-detect"
            else:
                self.valid = True
                self.status = f"LOCKED: {len(self.pixel_points)} points; click to order"
            return
        if not valid_pose or not dots:
            self.samples.clear()
            self.status = "Need visible dots and fresh Initial Point pose"
            return
        points = np.array(sorted(map(tuple, dots)), dtype=np.float64)
        if self.samples:
            matched = match_points(self.samples[0], points)
            if matched is None:
                self.samples.clear()
            else:
                points = matched
        self.samples.append(points)
        self.status = f"Collecting {len(points)} dots: {len(self.samples)}/{DOT_STABLE_FRAMES}"
        if len(self.samples) >= DOT_STABLE_FRAMES:
            averaged = np.mean(self.samples, axis=0)
            if match_points(averaged, averaged) is None:
                self.samples.clear()
                self.status = 'Dots too close to assign unique IDs'
                return
            self.pixel_points = np.array(sorted(averaged.tolist()), dtype=np.float64)
            self.valid = True
            self.status = f"LOCKED: {len(points)} points; click to order"


class PointSelection:
    def __init__(self):
        self.order = []

    def clear(self):
        self.order.clear()

    def undo(self):
        if self.order:
            self.order.pop()

    def click(self, x, y, points):
        if points is None or not len(points):
            return False
        distances = np.linalg.norm(points - np.array([x, y]), axis=1)
        index = int(np.argmin(distances))
        if distances[index] <= CLICK_RADIUS_PX and index not in self.order:
            self.order.append(index)
            return True
        return False

    def selected_points(self, points):
        if points is None:
            return None
        return points[self.order].copy()


def build_dot_targets(calibration, pixel_points):
    if calibration.homography is None or pixel_points is None or len(pixel_points) < 2:
        raise ValueError("至少選取兩點，且校正資料須就緒。")
    boundary = np.array([calibration.pixel_centers[i] for i in (1,2,4,3)], dtype=np.float32)
    reference = np.array([p[:2] for p in ARUCO_TCP_POSES.values()])
    lower, upper = reference.min(axis=0), reference.max(axis=0)
    targets = []
    for index, pixel in enumerate(pixel_points, 1):
        if cv2.pointPolygonTest(boundary, tuple(map(float, pixel)), False) < 0:
            raise ValueError("目標位於四點校正範圍外。")
        pose = pixel_to_robot_pose(*pixel, calibration.homography)
        target = (pose[0], pose[1], DOT_CONTACT_Z, *pose[3:])
        if (not np.isfinite(target).all() or np.any(np.array(target[:2]) < lower)
                or np.any(np.array(target[:2]) > upper)):
            raise ValueError("換算點位超出校正範圍。")
        targets.append((index, target))
    return tuple(targets)


def build_machining_path(targets):
    """依使用者選點順序產生 MovL 開放折線，不返回起點。"""
    if len(targets) < 2:
        raise ValueError('加工路徑至少需要兩個目標點。')
    poses = [tuple(target) for _, target in targets]
    if any(len(pose) != 6 or not np.isfinite(pose).all()
           or pose[2] != DOT_CONTACT_Z for pose in poses):
        raise ValueError('加工點須為有效六軸座標，且加工高度一致。')
    if DOT_TRAVEL_Z <= DOT_CONTACT_Z:
        raise ValueError('移動高度必須高於加工高度。')
    lengths = [float(np.linalg.norm(np.array(b[:2]) - np.array(a[:2])))
               for a, b in zip(poses, poses[1:])]
    if any(length <= 1e-6 for length in lengths):
        raise ValueError('相鄰加工點不得重合。')
    first, last = poses[0], poses[-1]
    steps = [
        ('travel', '白紙點 1 上方', (*first[:2], DOT_TRAVEL_Z, *first[3:]), DOT_MOVE_SPEED),
        ('plunge', '白紙點 1 落筆', first, DOT_CONTACT_SPEED),
    ]
    steps.extend(('cut', f'加工序 {i} 至 {i + 1}', pose, DOT_CONTACT_SPEED)
                 for i, pose in enumerate(poses[1:], 1))
    steps.append(('retract', '加工完成抬筆', (*last[:2], DOT_TRAVEL_Z, *last[3:]), DOT_CONTACT_SPEED))
    return {
        'point_order': list(range(1, len(poses) + 1)),
        'ordering': 'user_click_order',
        'closed': False,
        'segment_lengths_mm': lengths,
        'total_cut_length_mm': sum(lengths),
        'moves': [{'type': kind, 'label': label, 'tcp_pose': list(pose),
                   'speed': speed, 'command': 302}
                  for kind, label, pose, speed in steps],
    }


class MultiDotVisit:
    """由 Start 觸發；儲存並執行相同的使用者路徑，每次啟動只執行一輪。"""
    def __init__(self):
        self.started = False
        self.thread = None
        self.cancel = threading.Event()
        self.status = "Select at least 2 points, then Start"
        self.pixel_points = None

    def start(self, calibration, pixel_points, labels, all_points):
        if self.started:
            return
        targets = build_dot_targets(calibration, pixel_points)
        path = build_machining_path(targets)
        if len(labels) != len(targets) or len(set(labels)) != len(labels):
            raise ValueError('點位標記數量或順序無效。')
        path['marker_order'] = list(labels)
        data = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "initial_tcp_pose": list(INITIAL_POINT),
            "homography_pixel_to_tcp_xy": calibration.homography.tolist(),
            "travel_z_mm": DOT_TRAVEL_Z,
            "operation": "draw_user_selected_open_polyline",
            "detected_point_count": len(all_points),
            "selected_point_count": len(targets),
            "units": {"position": "mm", "image": "pixel", "orientation": "deg"},
            "detected_points": [{"marker": point_label(i), "pixel_uv": list(map(float, p))}
                                for i, p in enumerate(all_points)],
            "drawing_z_mm": DOT_CONTACT_Z,
            "targets": [{"point": i, "marker": label, "pixel_uv": list(map(float, pixel)), "tcp_pose": list(target)}
                        for (i,target),pixel,label in zip(targets,pixel_points,labels)],
            "machining_path": path,
        }
        temporary = DOT_TARGET_PATH.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
        temporary.replace(DOT_TARGET_PATH)
        for index, target in targets:
            print(f"白紙點 {index} 換算 TCP：{target}", flush=True)
        print(f"加工路徑 {' -> '.join(labels)}，總長 {path['total_cut_length_mm']:.3f} mm；已儲存：{DOT_TARGET_PATH}", flush=True)
        self.pixel_points = np.array(pixel_points, dtype=np.float64, copy=True)
        self.started = True
        self.thread = threading.Thread(target=self._run, args=(path,), name='selected-dot-visit')
        self.thread.start()

    def _run(self, path):
        try:
            with RobotMotionSession() as motion:
                self._execute(path, motion)
        except Exception as exc:
            self.status = 'FAILED: no further commands'
            print(f'加工失敗，取消後續動作：{exc}', file=sys.stderr, flush=True)

    def _execute(self, path, motion):
        current = None
        pen_down = False
        for index, step in enumerate(path['moves']):
            if self.cancel.is_set():
                break
            self.status = step['label']
            motion.move(step['tcp_pose'], step['label'],
                        speed=step['speed'], command=step['command'])
            current = tuple(step['tcp_pose'])
            pen_down = step['type'] in ('plunge', 'cut')
            if index == len(path['moves']) - 2:
                self.cancel.wait(DOT_DWELL)
        # 取消不會中斷已送出動作；到位後於當前端點垂直抬筆。
        if self.cancel.is_set() and pen_down:
            above_current = (*current[:2], DOT_TRAVEL_Z, *current[3:])
            motion.move(above_current, '取消加工後抬筆', speed=DOT_CONTACT_SPEED, command=302)
        if self.cancel.is_set():
            self.status = 'Cancelled; no pending path moves'
        else:
            self.status = 'DONE: selected path completed; pen raised'
            print(f'加工路徑完成，筆尖已抬至 Z={DOT_TRAVEL_Z} mm。', flush=True)

    def close(self):
        self.cancel.set()
        if self.thread is not None:
            if self.thread.is_alive():
                print('結束預覽：取消尚未開始的畫線，等待當前動作及正常抬筆結束；不是急停。', flush=True)
            self.thread.join()


def draw_machining_preview(image, pixel_points):
    """繪製所選開放路徑；移動後的疊圖僅代表拍攝時的路徑。"""
    if pixel_points is None or len(pixel_points) < 2:
        return
    points = np.rint(pixel_points).astype(np.int32)
    for start, end in zip(points, points[1:]):
        cv2.arrowedLine(image, tuple(start), tuple(end), (0, 200, 255), 2,
                        cv2.LINE_AA, tipLength=0.12)
    for index, point in enumerate(points, 1):
        cv2.circle(image, tuple(point), 10, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(image, f'P{index}', (int(point[0]) + 12, int(point[1]) + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)


class SelectionUI:
    """縮放預覽以容納操作面板，點擊換回原始相機座標；回呼只排入事件。"""
    def __init__(self):
        self.actions = []
        self.buttons = []
        self.scroll = 0
        self.message = 'Wait for point lock, then click markers.'
        self.image_size = (0, 0)
        self.preview_size = (0, 0)

    def on_mouse(self, event, x, y, flags, _param):
        if event == cv2.EVENT_MOUSEWHEEL:
            delta = (flags >> 16) & 0xffff
            if delta >= 32768:
                delta -= 65536
            self.scroll = max(0, self.scroll + (-1 if delta > 0 else 1))
        elif event == cv2.EVENT_LBUTTONDOWN:
            for action, (x1, y1, x2, y2) in self.buttons:
                if x1 <= x < x2 and y1 <= y < y2:
                    self.actions.append((action, None))
                    return
            width, height = self.image_size
            preview_width, preview_height = self.preview_size
            if 0 <= x < preview_width and 0 <= y < preview_height:
                self.actions.append(('click', (x * width / preview_width,
                                                y * height / preview_height)))

    def render(self, image, tracker, selection, visit, calibration):
        height, width = image.shape[:2]
        # 只縮放顯示用影像，偵測、校正及加工座標維持原始解析度。
        scale = min(1.0, (1280 - PANEL_WIDTH) / width, 720 / height)
        preview_width = max(1, round(width * scale))
        preview_height = max(1, round(height * scale))
        self.image_size = (width, height)
        self.preview_size = (preview_width, preview_height)
        preview = (image if self.image_size == self.preview_size else
                   cv2.resize(image, self.preview_size, interpolation=cv2.INTER_AREA))
        panel_height = 720
        canvas = np.full((panel_height, preview_width + PANEL_WIDTH, 3), (30, 30, 30), dtype=np.uint8)
        canvas[:preview_height, :preview_width] = preview
        left = preview_width + 14

        def text(value, y, color=(230, 230, 230), scale=0.5):
            cv2.putText(canvas, value, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
                        scale, color, 1, cv2.LINE_AA)

        text('V4 - Machining point selection', 28, (0, 220, 255), 0.6)
        locked_count = 0 if tracker.pixel_points is None else len(tracker.pixel_points)
        text(f'Detected: {tracker.detected_count}  Locked: {locked_count}', 56)
        text(f'Selected: {len(selection.order)} / {locked_count}', 82)
        state = visit.status if visit.started else tracker.status
        for row, line in enumerate(textwrap.wrap(state, 45)[:3]):
            text(line, 110 + row * 22, (0, 200, 255))
        for row, line in enumerate(textwrap.wrap(self.message, 45)[:2]):
            text(line, 180 + row * 20)

        editable = tracker.valid and not visit.started
        specs = [('undo', 'Undo [U]', 0, 220), ('clear', 'Clear [C]', 205, 220),
                 ('redetect', 'Re-detect [R]', 0, 270), ('start', 'Start [S]', 205, 270)]
        self.buttons = []
        for action, label, offset, top in specs:
            rectangle = (left + offset, top, left + offset + 190, top + 38)
            enabled = not visit.started if action == 'redetect' else editable
            if action == 'start':
                enabled = enabled and len(selection.order) >= 2
            color = (65, 100, 65) if enabled else (55, 55, 55)
            cv2.rectangle(canvas, rectangle[:2], rectangle[2:], color, -1)
            cv2.putText(canvas, label, (rectangle[0] + 8, top + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (245, 245, 245), 1, cv2.LINE_AA)
            self.buttons.append((action, rectangle))

        text('Open path - click order:', 342, (0, 220, 255))
        order = ' -> '.join(point_label(i) for i in selection.order) or '(no points selected)'
        lines = textwrap.wrap(order, 42)
        visible = max(1, (panel_height - 475) // 24)
        self.scroll = min(self.scroll, max(0, len(lines) - visible))
        for row, line in enumerate(lines[self.scroll:self.scroll + visible]):
            text(line, 370 + row * 24)
        if len(lines) > visible:
            text('Mouse wheel: scroll order', panel_height - 92)
        if editable and len(selection.order) >= 2:
            try:
                targets = build_dot_targets(calibration, selection.selected_points(tracker.pixel_points))
                length = build_machining_path(targets)['total_cut_length_mm']
                text(f'Cut length: {length:.2f} mm', panel_height - 66)
            except ValueError:
                text('Path invalid: re-detect / check calibration', panel_height - 66, (0, 0, 255))
        text('P: TCP    Q/Esc: quit', panel_height - 40)
        text('During machining: editing is locked.', panel_height - 16)
        return canvas


def draw_selectable_points(image, points, selection, calibration, valid=True):
    if points is None:
        return
    for index, pixel in enumerate(points):
        cx, cy = map(int, np.rint(pixel))
        selected = index in selection.order
        color = (0, 220, 0) if selected else (0, 180, 255)
        if not valid:
            color = (100, 100, 100)
        cv2.rectangle(image, (cx - 12, cy - 12), (cx + 12, cy + 12), color, 2)
        label = point_label(index)
        if selected:
            label += f' [{selection.order.index(index) + 1}]'
        if calibration.homography is not None:
            x, y, *_ = pixel_to_robot_pose(*pixel, calibration.homography)
            label += f' X:{x:.1f} Y:{y:.1f}'
        position = (max(0, cx - 20), max(16, cy - 18))
        cv2.putText(image, label, position, cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(image, label, position, cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    color, 1, cv2.LINE_AA)


def at_initial_point(pose):
    return (
        pose is not None
        and all(abs(pose[i] - INITIAL_POINT[i]) <= 1.0 for i in range(3))
        and all(abs((pose[i] - INITIAL_POINT[i] + 180) % 360 - 180) <= 1.0
                for i in range(3, 6))
    )


def marker_center(points):
    """以兩條對角線交點求標記中心，保留次像素精度。"""
    points = np.asarray(points, dtype=np.float64).reshape(4, 2)
    homogeneous = np.column_stack((points, np.ones(4)))
    intersection = np.cross(np.cross(homogeneous[0], homogeneous[2]),
                            np.cross(homogeneous[1], homogeneous[3]))
    if abs(intersection[2]) < 1e-9:
        return None
    return intersection[:2] / intersection[2]


def pixel_to_robot_pose(u, v, homography):
    """只適用校正平面與相同白板筆夾持姿態，回傳目標 TCP 六軸座標。"""
    result = homography @ np.array([u, v, 1.0], dtype=np.float64)
    if not np.isfinite(result).all() or abs(result[2]) < 1e-9:
        raise ValueError("此像素無法換算至校正平面。")
    x, y = result[:2] / result[2]
    return (float(x), float(y), *ARUCO_TCP_POSES[1][2:])


class PlaneCalibration:
    """每次啟動重新取樣；只在 Initial Point 收集並儲存四點平面對應。"""

    def __init__(self, path=CALIBRATION_PATH):
        self.path = Path(path)
        self.samples = []
        self.homography = None
        self.image_size = None
        self.pixel_centers = None
        self.status = "Waiting for Initial Point"

    def update(self, centers, image_size, pose):
        if not at_initial_point(pose) or (
            self.image_size is not None and self.image_size != image_size
        ):
            self.samples.clear()
            self.homography = None
            self.pixel_centers = None
            self.image_size = image_size
            self.status = "Invalid: return to Initial Point / same stream"
            return
        self.image_size = image_size
        if self.homography is not None:
            return  # 完成後固定 H，不隨每張畫面漂移。
        ids = sorted(ARUCO_TCP_POSES)
        if not all(marker_id in centers for marker_id in ids):
            self.samples.clear()
            self.status = "Need IDs 1, 2, 3, 4 together"
            return
        pixels = np.array([centers[marker_id] for marker_id in ids], dtype=np.float64)
        # 四點必須依 ID 1->2->4->3 組成非退化凸四邊形。
        boundary = pixels[[0, 1, 3, 2]].astype(np.float32)
        if (not np.isfinite(pixels).all() or not cv2.isContourConvex(boundary)
                or abs(cv2.contourArea(boundary)) < 100):
            self.samples.clear()
            self.status = "Invalid marker layout"
            return
        self.samples.append(pixels)
        self.status = f"Collecting: {len(self.samples)}/{CALIBRATION_SAMPLES}"
        if len(self.samples) < CALIBRATION_SAMPLES:
            return
        samples = np.array(self.samples)
        std = samples.std(axis=0)
        if std.max() > MAX_CENTER_STD_PX:
            self.samples.clear()
            self.status = "Markers unstable: collecting again"
            return
        mean_pixels = samples.mean(axis=0)
        robot_xy = np.array([ARUCO_TCP_POSES[i][:2] for i in ids], dtype=np.float64)
        homography, _ = cv2.findHomography(mean_pixels, robot_xy, method=0)
        if (homography is None or not np.isfinite(homography).all()
                or np.linalg.matrix_rank(homography) < 3):
            self.samples.clear()
            self.status = "Homography failed: collecting again"
            return
        records = []
        for index, marker_id in enumerate(ids):
            converted = pixel_to_robot_pose(*mean_pixels[index], homography)
            records.append({
                "id": marker_id,
                "pixel_center_uv": mean_pixels[index].tolist(),
                "pixel_std_uv": std[index].tolist(),
                "reference_tcp_pose": list(ARUCO_TCP_POSES[marker_id]),
                "converted_tcp_pose": list(converted),
            })
        data = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "method": "planar_homography_pixel_to_pen_tcp_xy",
            "dictionary": "DICT_5X5_100",
            "center_method": "diagonal_intersection",
            "pixel_space": "raw_color_image_without_undistortion",
            "image_size_wh": list(image_size),
            "initial_tcp_pose": list(INITIAL_POINT),
            "measured_tcp_pose_at_capture": list(pose),
            "units": {"position": "mm", "orientation": "deg", "image": "pixel"},
            "sample_count": len(samples),
            "homography_pixel_to_tcp_xy": homography.tolist(),
            "markers": records,
            "usage": "Same camera pose, stream, plane, pen grip and tool orientation only; not 3D hand-eye calibration.",
        }
        # 先完整寫入暫存檔，再替換，避免留下半份 JSON。
        temporary = self.path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(self.path)
        self.homography = homography
        self.pixel_centers = {marker_id: mean_pixels[index].copy()
                              for index, marker_id in enumerate(ids)}
        self.status = "READY: planar pixel -> pen TCP XY"
        print(f"四點平面校正已儲存：{self.path}", flush=True)
        for record in records:
            u, v = record["pixel_center_uv"]
            x, y = record["converted_tcp_pose"][:2]
            print(f"ID {record['id']}: pixel=({u:.3f}, {v:.3f}) -> TCP X={x:.3f}, Y={y:.3f} mm")


class RobotMotionSession:
    """一條加工路徑共用連線，只在速度改變時寫入速度暫存器。"""
    def __init__(self):
        self.client = None
        self.speed = None

    def __enter__(self):
        self.client = ModbusTcpClient(ROBOT_HOST, port=ROBOT_PORT, timeout=1, retries=0)
        try:
            if not self.client.connect():
                raise RuntimeError(f'無法連線至機器人 {ROBOT_HOST}:{ROBOT_PORT}，取消加工。')
        except Exception:
            self.client.close()
            raise
        return self

    def move(self, target, label, speed=INITIAL_SPEED, command=301):
        move_to_pose(target, label, speed=speed, command=command,
                     client=self.client, set_speed=self.speed != speed)
        self.speed = speed

    def __exit__(self, exc_type, exc, traceback):
        self.client.close()


def check_motion_response(result, label, operation, address):
    if result is None or result.isError():
        code = getattr(result, 'exception_code', None)
        raise RuntimeError(
            f'{label} {operation}失敗 [register=0x{address:04X}, '
            f'exception_code={code}, response={result!r}]，取消後續動作。')


def move_to_pose(target, label, speed=INITIAL_SPEED, command=301, *, client=None, set_speed=True):
    """依 DRV 協定移動並確認到位；301=MovP、302=MovL。"""
    target = tuple(target)
    if len(target) != 6 or not np.isfinite(target).all():
        raise ValueError("目標點位必須包含六個有效數值。")
    owns_client = client is None
    if owns_client:
        client = ModbusTcpClient(ROBOT_HOST, port=ROBOT_PORT, timeout=1, retries=0)
    try:
        if owns_client and not client.connect():
            raise RuntimeError(f"無法連線至機器人 {ROBOT_HOST}:{ROBOT_PORT}，取消啟動。")
        builder = BinaryPayloadBuilder(byteorder=Endian.BIG, wordorder=Endian.LITTLE)
        for value in target:
            builder.add_32bit_int(round(value * 1000))

        print(f"移動至 {label}：{target}，速度設定 {speed}", flush=True)
        # 每一步皆檢查回應；不重送動作命令。
        if set_speed:
            result = client.write_register(0x0324, speed, 2)
            check_motion_response(result, label, '速度設定', 0x0324)
        result = client.write_registers(0x0330, builder.to_registers(), 2)
        check_motion_response(result, label, '座標寫入', 0x0330)
        result = client.write_register(0x0300, command, 2)
        check_motion_response(result, label, '移動命令回應（請確認機器人實際狀態）', 0x0300)

        deadline = time.monotonic() + INITIAL_MOVE_TIMEOUT
        while time.monotonic() < deadline:
            flag = request.Get_Pose_Flag(client)
            pose = request.Get_TCP_Pose(client)
            if flag is None or pose is None:
                raise RuntimeError("移動後無法讀取機器人狀態，請確認機器人實際狀態。")
            # 同時檢查實際座標，避免把上一個動作的到位旗標誤認為完成。
            position_ok = all(abs(pose[i] - target[i]) <= 1.0 for i in range(3))
            angle_ok = all(
                abs((pose[i] - target[i] + 180) % 360 - 180) <= 1.0
                for i in range(3, 6)
            )
            if flag == 1 and position_ok and angle_ok:
                print(f"已到達 {label}。", flush=True)
                return
            time.sleep(0.1)
        raise RuntimeError(
            f"等待 {label} 到位逾時，取消後續動作；此逾時不會停止手臂，請確認機器人實際狀態。"
        )
    finally:
        if owns_client:
            client.close()


def move_to_initial_point():
    move_to_pose(INITIAL_POINT, "Initial Point")


def create_aruco_detector():
    """回傳偵測函式，相容新版與舊版 OpenCV。"""
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICTIONARY)
    if hasattr(cv2.aruco, "ArucoDetector"):
        parameters = cv2.aruco.DetectorParameters()
        return cv2.aruco.ArucoDetector(dictionary, parameters).detectMarkers

    parameters = cv2.aruco.DetectorParameters_create()

    def detect_markers(image):
        return cv2.aruco.detectMarkers(image, dictionary, parameters=parameters)

    return detect_markers


def draw_aruco_markers(image, detect_markers, calibration=None, capture_pose=None):
    """偵測原始影像後收集中心，再標記框線、中心及平面換算座標。"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detect_markers(gray)
    centers = {}
    if ids is not None:
        id_list = [int(i) for i in ids.flatten()]
        for marker_corners, marker_id in zip(corners, id_list):
            center = marker_center(marker_corners)
            # 同一 ID 出現多次時不採樣，避免建立錯誤配對。
            if center is not None and id_list.count(marker_id) == 1:
                centers[marker_id] = center
    if calibration is not None:
        height, width = image.shape[:2]
        calibration.update(centers, (width, height), capture_pose)
    if ids is None:
        return centers

    for marker_corners, marker_id in zip(corners, ids.flatten()):
        points = marker_corners.reshape(4, 2)
        outline = np.rint(points).astype(np.int32)
        center = marker_center(points)
        if center is None:
            continue
        cx, cy = np.rint(center).astype(int)
        cv2.polylines(image, [outline], True, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.circle(image, (cx, cy), 5, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.drawMarker(image, (cx, cy), (0, 255, 255), cv2.MARKER_CROSS, 16, 1)
        label = f"ID {int(marker_id)} ({cx}, {cy})"
        position = (max(0, cx - 50), max(15, cy - 15))
        cv2.putText(image, label, position, cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(image, label, position, cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 255), 1, cv2.LINE_AA)
        if calibration is not None and calibration.homography is not None:
            x, y, *_ = pixel_to_robot_pose(*center, calibration.homography)
            text = f"TCP X:{x:.2f} Y:{y:.2f} mm"
            position = (max(0, cx - 50), min(image.shape[0] - 5, cy + 25))
            cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (255, 255, 0), 1, cv2.LINE_AA)
    return centers


class RobotPoseReader:
    """在背景讀取 TCP，避免 Modbus 連線等待阻塞影像預覽。"""

    def __init__(self):
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.pose = None
        self.status = "Connecting..."
        self.updated_at = None
        self.thread = threading.Thread(target=self._run, name="robot-pose-reader")

    def snapshot(self):
        with self.lock:
            return self.pose, self.status, self.updated_at

    def _run(self):
        client = ModbusTcpClient(ROBOT_HOST, port=ROBOT_PORT, timeout=1, retries=0)
        try:
            while not self.stop_event.is_set():
                try:
                    if not client.connect():
                        raise ConnectionError("Robot disconnected")
                    pose = request.Get_TCP_Pose(client)
                    if pose is None or len(pose) != 6:
                        raise RuntimeError("TCP read failed")
                    with self.lock:
                        self.pose = tuple(pose)
                        self.status = "Connected"
                        self.updated_at = time.monotonic()
                except Exception as exc:
                    # 連線失敗時不顯示舊座標，下一輪自動重連。
                    with self.lock:
                        self.pose = None
                        self.updated_at = None
                        self.status = "Disconnected / read failed"
                    print(f"機器人座標讀取失敗：{exc}", file=sys.stderr)
                    client.close()
                    if self.stop_event.wait(1):
                        break
                if self.stop_event.wait(POSE_POLL_INTERVAL):
                    break
        finally:
            client.close()

    def start(self):
        self.thread.start()

    def close(self):
        self.stop_event.set()
        self.thread.join()


def draw_robot_pose(image, reader):
    pose, status, updated_at = reader.snapshot()
    # 防止讀取逾時期間把上次的資料當成即時座標。
    if updated_at is not None and time.monotonic() - updated_at > 1:
        pose, status = None, "Waiting for fresh TCP data..."
    lines = [f"Robot TCP: {status}"]
    if pose is None:
        lines += ["X: ---  Y: ---  Z: --- (mm)", "Rx: ---  Ry: ---  Rz: --- (deg)"]
    else:
        x, y, z, rx, ry, rz = pose
        lines += [f"X: {x:.3f}  Y: {y:.3f}  Z: {z:.3f} mm",
                  f"Rx: {rx:.3f}  Ry: {ry:.3f}  Rz: {rz:.3f} deg"]
    for index, text in enumerate(lines):
        position = (10, 25 + index * 25)
        cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 1, cv2.LINE_AA)


def print_robot_pose(reader):
    pose, status, updated_at = reader.snapshot()
    if pose is None or updated_at is None or time.monotonic() - updated_at > 1:
        print(f"目前沒有可用的即時機器人座標：{status}")
        return
    x, y, z, rx, ry, rz = pose
    print(f"TCP: x={x:.3f}, y={y:.3f}, z={z:.3f} mm; "
          f"rx={rx:.3f}, ry={ry:.3f}, rz={rz:.3f} deg", flush=True)


def terminal_print_requested():
    """Windows 終端機按 P 即觸發，不需要按 Enter。"""
    requested = False
    while msvcrt.kbhit():
        key = msvcrt.getwch()
        if key in ("\x00", "\xe0"):
            msvcrt.getwch()  # 跳過方向鍵等特殊按鍵的第二個字元。
        elif key.lower() == "p":
            requested = True
    return requested


def main():
    detect_markers = create_aruco_detector()
    move_to_initial_point()
    # 避免在 import 時開啟相機，並相容 Windows 設定目錄讀取失敗的環境。
    try:
        context = rs.context()
    except RuntimeError as exc:
        if "SHGetKnownFolderPath" not in str(exc):
            raise
        print("無法讀取 RealSense 設定目錄，改用 SDK 預設設定。")
        context = rs.context('{"inherit": false}')

    pipeline = rs.pipeline(context)
    config = rs.config()
    # 只指定 RGB 格式；寬、高、FPS 設為 0，交由 SDK 選擇。
    # 影像不裁切、不縮放；這不保證與 Windows 相機程式的預設模式相同。
    config.enable_stream(rs.stream.color, 0, 0, rs.format.rgb8, 0)
    started = False
    robot = RobotPoseReader()
    calibration = PlaneCalibration()
    visit = MultiDotVisit()
    dot_tracker = MultiDotTracker()
    selection = PointSelection()
    ui = SelectionUI()
    robot_started = False
    try:
        profile = pipeline.start(config)
        started = True
        color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
        print(
            f"RGB 串流：{color_profile.width()} x {color_profile.height()}，"
            f"{color_profile.fps()} FPS"
        )
        robot.start()
        robot_started = True
        print(f"讀取機器人 TCP：{ROBOT_HOST}:{ROBOT_PORT}（每 {POSE_POLL_INTERVAL} 秒）")
        print("終端機按 P 可印出最新座標（不需 Enter）；預覽視窗也支援 P。")
        print("預覽視窗按 Q、Esc 或關閉視窗結束預覽。")
        print("V4：辨識白紙內任意數量的黑色實心圓點，穩定後鎖定 A、B、C 等標記。")
        print("左鍵依序選取加工點，至少兩點；可只選部分點；重複點選不重複加入。")
        print("U 復原、C 清空、R 重新偵測、S 開始加工；也可使用右側按鈕。")
        print("開放折線不返回起點；啟動時仍會先移至 Initial Point，選點完成後才執行加工。")
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(WINDOW_NAME, ui.on_mouse)

        while True:
            frames = pipeline.wait_for_frames(5000)
            color_frame = frames.get_color_frame()
            if color_frame:
                rgb_image = np.asanyarray(color_frame.get_data())
                # OpenCV 顯示使用 BGR，僅轉換色彩通道，不改變影像範圍。
                bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
                raw_image = bgr_image.copy()
                # 在繪製 TCP 文字前偵測，避免文字遮擋原始標記。
                pose, _, updated_at = robot.snapshot()
                if updated_at is None or time.monotonic() - updated_at > 1:
                    pose = None
                # 移動開始後，路徑與選擇都固定，不再以移動中的相機更新點位。
                draw_aruco_markers(bgr_image, detect_markers,
                                   None if visit.started else calibration, pose)
                if not visit.started:
                    paper, dots, detection_status = detect_paper_dots(raw_image, calibration, detect_markers)
                    dot_tracker.update(dots, at_initial_point(pose) and calibration.homography is not None)
                    if dot_tracker.invalidated:
                        selection.clear()
                    if paper is not None:
                        cv2.polylines(bgr_image, [paper], True, (255, 0, 255), 2, cv2.LINE_AA)
                    if dot_tracker.pixel_points is None:
                        for pixel in dots:
                            cx, cy = map(int, np.rint(pixel))
                            cv2.rectangle(bgr_image, (cx - 12, cy - 12), (cx + 12, cy + 12), (150, 150, 150), 1)
                        ui.message = detection_status

                # 所有滑鼠及鍵盤事件都在取得本幀影像和最新 TCP 後才處理。
                pending, ui.actions = ui.actions, []
                for action, value in pending:
                    if visit.started:
                        ui.message = 'Machining path is locked. Restart app for a new job.'
                        continue
                    if action == 'redetect':
                        dot_tracker.reset()
                        selection.clear()
                        ui.scroll = 0
                        ui.message = 'Re-detecting; keep the sheet and camera still.'
                        continue
                    if not dot_tracker.valid:
                        ui.message = ('Selection disabled: press R to re-detect.'
                                      if dot_tracker.invalidated else
                                      'Gray boxes are candidates. Wait for LOCKED before selecting.')
                        continue
                    if action == 'click':
                        x, y = value
                        if 0 <= x < bgr_image.shape[1] and 0 <= y < bgr_image.shape[0]:
                            if selection.click(x, y, dot_tracker.pixel_points):
                                ui.message = 'Order updated. Select at least 2, then Start.'
                            else:
                                ui.message = 'Click the center of an unselected point box.'
                    elif action == 'undo':
                        selection.undo()
                        ui.message = 'Last selection removed.'
                    elif action == 'clear':
                        selection.clear()
                        ui.message = 'Order cleared; point IDs stay fixed.'
                    elif action == 'start':
                        fresh_pose, _, timestamp = robot.snapshot()
                        if len(selection.order) < 2:
                            ui.message = 'Select at least two different points.'
                        elif (timestamp is None or time.monotonic() - timestamp > 1
                              or not at_initial_point(fresh_pose)
                              or calibration.homography is None):
                            ui.message = 'Need fresh Initial Point pose and calibration.'
                        else:
                            try:
                                visit.start(calibration,
                                            selection.selected_points(dot_tracker.pixel_points),
                                            [point_label(i) for i in selection.order],
                                            dot_tracker.pixel_points.copy())
                            except (ValueError, OSError) as exc:
                                ui.message = 'Could not start; see terminal for details.'
                                print(f'無法開始加工：{exc}', file=sys.stderr, flush=True)
                            else:
                                calibration.status = 'Frozen at Initial Point'
                                ui.message = 'Path saved; machining started. Editing locked.'

                preview_points = (visit.pixel_points if visit.started else
                                  selection.selected_points(dot_tracker.pixel_points))
                draw_machining_preview(bgr_image, preview_points)
                draw_selectable_points(bgr_image, dot_tracker.pixel_points, selection, calibration,
                                       valid=dot_tracker.valid or visit.started)
                draw_robot_pose(bgr_image, robot)
                cv2.putText(bgr_image, f"Calibration: {calibration.status}", (10, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(bgr_image, f"Calibration: {calibration.status}", (10, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
                if visit.started:
                    cv2.putText(bgr_image, 'Path: Initial Point snapshot (not live registration)',
                                (10, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1, cv2.LINE_AA)
                cv2.imshow(WINDOW_NAME, ui.render(bgr_image, dot_tracker, selection, visit, calibration))

            key = cv2.waitKey(1) & 0xFF
            if terminal_print_requested() or key in (ord('p'), ord('P')):
                print_robot_pose(robot)
            if key in (ord('q'), ord('Q'), 27):
                break
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break
            key_actions = {ord('u'): 'undo', ord('c'): 'clear', ord('r'): 'redetect', ord('s'): 'start'}
            action = key_actions.get(key + 32 if ord('A') <= key <= ord('Z') else key)
            if action is not None:
                ui.actions.append((action, None))
    finally:
        try:
            visit.close()
        finally:
            try:
                if started:
                    pipeline.stop()
            finally:
                try:
                    if robot_started:
                        robot.close()
                finally:
                    cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
    except (RuntimeError, OSError, ValueError) as exc:
        print(f"程式啟動或讀取失敗：{exc}", file=sys.stderr)
        print(
            "請依錯誤確認機器人連線與狀態，或 RealSense USB 連線、相機存取權限，"
            "並關閉其他使用相機的程式。",
            file=sys.stderr,
        )
        sys.exit(1)
