"""V3：四點平面校正、白紙三黑點辨識、開放折線加工。

依畫面由左至右編號（同 X 時由上至下），路徑為 1 -> 2 -> 3，不閉合。
與 V2 相同：啟動後移至 Initial Point，校正及目標穩定後自動執行一輪。
"""
#手臂到初始點位後偵測
import sys
import msvcrt
import threading
import time
import json
from datetime import datetime, timezone
from pathlib import Path
from itertools import permutations

import cv2
import numpy as np
import pyrealsense2 as rs
from pymodbus.client import ModbusTcpClient
from pymodbus.payload import BinaryPayloadBuilder
from pymodbus.constants import Endian
from drv_modbus import request


WINDOW_NAME = "RealSense V3 - Three Dot Path"
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
CALIBRATION_PATH = Path(__file__).with_name("aruco_plane_calibration_v3.json")
DOT_TARGET_PATH = Path(__file__).with_name("three_dot_targets_v3.json")
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
    return paper, dots, f"Black dots: {len(dots)} (need exactly 3)"


class ThreeDotTracker:
    def __init__(self):
        self.samples = []
        self.pixel_points = None
        self.status = "Waiting for three black dots"

    def update(self, dots, valid_pose):
        if not valid_pose or len(dots) != 3:
            self.samples.clear()
            self.pixel_points = None
            self.status = "Need exactly 3 dots at Initial Point"
            return
        points = np.array(sorted(dots), dtype=np.float64)
        self.pixel_points = None
        if self.samples:
            previous = self.samples[-1]
            # 三點的六種配對皆需比較，避免左右排序交換造成追蹤跳號。
            points = min((points[list(order)] for order in permutations(range(3))),
                         key=lambda candidate: np.linalg.norm(candidate - previous))
            if np.max(np.linalg.norm(points - self.samples[0], axis=1)) > DOT_MAX_JITTER_PX:
                self.samples.clear()
        self.samples.append(points)
        self.status = f"Collecting dots: {len(self.samples)}/{DOT_STABLE_FRAMES}"
        if len(self.samples) >= DOT_STABLE_FRAMES:
            self.samples = self.samples[-DOT_STABLE_FRAMES:]
            averaged = np.mean(self.samples[-DOT_STABLE_FRAMES:], axis=0)
            self.pixel_points = np.array(sorted(averaged.tolist()), dtype=np.float64)
            self.status = "READY: three stable dots"


def build_dot_targets(calibration, pixel_points):
    if calibration.homography is None or pixel_points is None or len(pixel_points) != 3:
        raise ValueError("三點或校正資料尚未就緒。")
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
    """產生可儲存及執行的 MovL 路徑：接近、落筆、1→2→3、抬筆。"""
    if len(targets) != 3:
        raise ValueError('加工路徑需要三個目標點。')
    poses = [tuple(target) for _, target in targets]
    if any(len(pose) != 6 or not np.isfinite(pose).all()
           or pose[2] != DOT_CONTACT_Z for pose in poses):
        raise ValueError('三點須為有效六軸座標，且加工高度一致。')
    if DOT_TRAVEL_Z <= DOT_CONTACT_Z:
        raise ValueError('移動高度必須高於加工高度。')
    lengths = [float(np.linalg.norm(np.array(b[:2]) - np.array(a[:2])))
               for a, b in zip(poses, poses[1:])]
    if any(length <= 1e-6 for length in lengths):
        raise ValueError('相鄰加工點不得重合。')
    first, _, last = poses
    steps = [
        ('travel', '白紙點 1 上方', (*first[:2], DOT_TRAVEL_Z, *first[3:]), DOT_MOVE_SPEED),
        ('plunge', '白紙點 1 落筆', first, DOT_CONTACT_SPEED),
        ('cut', '白紙點 1 至點 2 加工', poses[1], DOT_CONTACT_SPEED),
        ('cut', '白紙點 2 至點 3 加工', last, DOT_CONTACT_SPEED),
        ('retract', '加工完成抬筆', (*last[:2], DOT_TRAVEL_Z, *last[3:]), DOT_CONTACT_SPEED),
    ]
    return {
        'point_order': [1, 2, 3],
        'ordering': 'image_left_to_right_then_top_to_bottom',
        'closed': False,
        'segment_lengths_mm': lengths,
        'total_cut_length_mm': sum(lengths),
        'moves': [{'type': kind, 'label': label, 'tcp_pose': list(pose),
                   'speed': speed, 'command': 302}
                  for kind, label, pose, speed in steps],
    }


class ThreeDotVisit:
    """儲存並執行同一份三點加工路徑，每次啟動只執行一輪。"""
    def __init__(self):
        self.started = False
        self.thread = None
        self.cancel = threading.Event()
        self.status = "Waiting for three dots"
        self.pixel_points = None

    def start(self, calibration, pixel_points):
        if self.started:
            return
        targets = build_dot_targets(calibration, pixel_points)
        path = build_machining_path(targets)
        data = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "initial_tcp_pose": list(INITIAL_POINT),
            "homography_pixel_to_tcp_xy": calibration.homography.tolist(),
            "travel_z_mm": DOT_TRAVEL_Z,
            "operation": "draw_polyline_point_1_to_2_to_3",
            "drawing_z_mm": DOT_CONTACT_Z,
            "targets": [{"point": i, "pixel_uv": list(map(float, pixel)), "tcp_pose": list(target)}
                        for (i,target),pixel in zip(targets,pixel_points)],
            "machining_path": path,
        }
        temporary = DOT_TARGET_PATH.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
        temporary.replace(DOT_TARGET_PATH)
        for index, target in targets:
            print(f"白紙點 {index} 換算 TCP：{target}", flush=True)
        print(f"加工路徑 1 -> 2 -> 3，總長 {path['total_cut_length_mm']:.3f} mm；已儲存：{DOT_TARGET_PATH}", flush=True)
        self.pixel_points = np.array(pixel_points, dtype=np.float64, copy=True)
        self.started = True
        self.thread = threading.Thread(target=self._run, args=(path,), name='three-dot-visit')
        self.thread.start()

    def _run(self, path):
        try:
            current = None
            pen_down = False
            for index, step in enumerate(path['moves']):
                if self.cancel.is_set():
                    break
                self.status = step['label']
                move_to_pose(step['tcp_pose'], step['label'],
                             speed=step['speed'], command=step['command'])
                current = tuple(step['tcp_pose'])
                pen_down = step['type'] in ('plunge', 'cut')
                if index == len(path['moves']) - 2:
                    self.cancel.wait(DOT_DWELL)
            # 取消不會中斷已送出動作；到位後於當前端點垂直抬筆。
            if self.cancel.is_set() and pen_down:
                above_current = (*current[:2], DOT_TRAVEL_Z, *current[3:])
                move_to_pose(above_current, '取消加工後抬筆', speed=DOT_CONTACT_SPEED, command=302)
            if self.cancel.is_set():
                self.status = 'Cancelled; no pending path moves'
            else:
                self.status = 'DONE: path 1 -> 2 -> 3; pen raised'
                print('三點加工路徑完成，筆尖已抬至 Z=500 mm。', flush=True)
        except Exception as exc:
            self.status = 'FAILED: no further commands'
            print(f'三點畫線失敗，取消後續動作：{exc}', file=sys.stderr, flush=True)

    def close(self):
        self.cancel.set()
        if self.thread is not None:
            if self.thread.is_alive():
                print('結束預覽：取消尚未開始的畫線，等待當前動作及正常抬筆結束；不是急停。', flush=True)
            self.thread.join()


def draw_machining_preview(image, pixel_points):
    """繪製 1→2→3 開放路徑；移動後的疊圖僅代表拍攝時的路徑。"""
    if pixel_points is None or len(pixel_points) != 3:
        return
    points = np.rint(pixel_points).astype(np.int32)
    for start, end in zip(points, points[1:]):
        cv2.arrowedLine(image, tuple(start), tuple(end), (0, 200, 255), 2,
                        cv2.LINE_AA, tipLength=0.12)
    for index, point in enumerate(points, 1):
        cv2.circle(image, tuple(point), 10, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(image, f'P{index}', (int(point[0]) + 12, int(point[1]) + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)


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


def move_to_pose(target, label, speed=INITIAL_SPEED, command=301):
    """依 DRV 協定移動並確認到位；301=MovP、302=MovL。"""
    target = tuple(target)
    if len(target) != 6 or not np.isfinite(target).all():
        raise ValueError("目標點位必須包含六個有效數值。")
    client = ModbusTcpClient(ROBOT_HOST, port=ROBOT_PORT, timeout=1, retries=0)
    try:
        if not client.connect():
            raise RuntimeError(f"無法連線至機器人 {ROBOT_HOST}:{ROBOT_PORT}，取消啟動。")
        builder = BinaryPayloadBuilder(byteorder=Endian.BIG, wordorder=Endian.LITTLE)
        for value in target:
            builder.add_32bit_int(round(value * 1000))

        print(f"移動至 {label}：{target}，速度設定 {speed}", flush=True)
        # 每一步皆檢查回應；不重送動作命令。
        result = client.write_register(0x0324, speed, 2)
        if result is None or result.isError():
            raise RuntimeError(f"{label} 速度設定失敗，取消後續動作。")
        result = client.write_registers(0x0330, builder.to_registers(), 2)
        if result is None or result.isError():
            raise RuntimeError(f"{label} 座標寫入失敗，取消後續動作。")
        result = client.write_register(0x0300, command, 2)
        if result is None or result.isError():
            raise RuntimeError(f"{label} 移動命令回應異常，請確認機器人實際狀態。")

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
    visit = ThreeDotVisit()
    dot_tracker = ThreeDotTracker()
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
        print("V3：白紙請平放在四個 ArUco 包圍範圍內，畫三個清楚分離的黑色實心圓點。")
        print("白紙邊界需能與背景區分；穩定辨識三點後自動加工一輪。")
        print("依畫面左至右編號：1 -> 2 -> 3，為開放折線，不連回第一點。")
        print(f"第一點上方 Z={DOT_TRAVEL_Z}；落筆後固定 Z={DOT_CONTACT_Z} mm 經第二點畫至第三點，再抬筆。")
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)

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
                # 移動前已複製三點加工路徑；手臂離開拍攝點後不重算 H。
                draw_aruco_markers(bgr_image, detect_markers,
                                   None if visit.started else calibration, pose)
                if not visit.started:
                    paper, dots, detection_status = detect_paper_dots(raw_image, calibration, detect_markers)
                    dot_tracker.update(dots, at_initial_point(pose) and calibration.homography is not None)
                    visit.status = dot_tracker.status if len(dots) == 3 else detection_status
                    if paper is not None:
                        cv2.polylines(bgr_image, [paper], True, (255, 0, 255), 2, cv2.LINE_AA)
                    for index, pixel in enumerate(dots, 1):
                        cx, cy = np.rint(pixel).astype(int)
                        cv2.circle(bgr_image, (cx, cy), 10, (0, 0, 255), 2, cv2.LINE_AA)
                        label = f'Dot {index} ({cx}, {cy})'
                        if calibration.homography is not None:
                            x, y, *_ = pixel_to_robot_pose(*pixel, calibration.homography)
                            label += f' X:{x:.2f} Y:{y:.2f}'
                        position = (max(0,cx-30), max(15,cy-15))
                        cv2.putText(bgr_image, label, position, cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                                    (0,0,0), 3, cv2.LINE_AA)
                        cv2.putText(bgr_image, label, position, cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                                    (0,0,255), 1, cv2.LINE_AA)
                preview_points = visit.pixel_points if visit.started else dot_tracker.pixel_points
                draw_machining_preview(bgr_image, preview_points)
                if visit.started:
                    cv2.putText(bgr_image, "Path: Initial Point snapshot (not live registration)",
                                (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1, cv2.LINE_AA)
                draw_robot_pose(bgr_image, robot)
                cv2.putText(bgr_image, f"Calibration: {calibration.status}",
                            (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(bgr_image, f"Calibration: {calibration.status}",
                            (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(bgr_image, f"Visit: {visit.status}", (10, 125),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(bgr_image, f"Visit: {visit.status}", (10, 125),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
                cv2.imshow(WINDOW_NAME, bgr_image)

            key = cv2.waitKey(1) & 0xFF
            if terminal_print_requested() or key in (ord("p"), ord("P")):
                print_robot_pose(robot)
            if key in (ord("q"), ord("Q"), 27):
                break
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break
            if (not visit.started and calibration.homography is not None
                    and dot_tracker.pixel_points is not None):
                pose, _, updated_at = robot.snapshot()
                if (updated_at is not None and time.monotonic() - updated_at <= 1
                        and at_initial_point(pose)):
                    visit.start(calibration, dot_tracker.pixel_points.copy())
                    calibration.status = "Frozen at Initial Point; no live conversion while moving"
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
