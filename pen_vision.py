"""
方法二共用模組：RealSense 彩色+深度對齊、ArUco 位姿、筆蓋色彩分割、座標轉換
(不 import 專案內的 realsense/realsense.py，因為它一 import 就開相機且沒有深度對齊)
"""
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as Rot

# ---------------- 手臂姿態 <-> 4x4 矩陣 ----------------
# 台達手臂 (x,y,z,rx,ry,rz) 的尤拉角順序不確定，校正時兩種都試，自動選誤差小的
#   "RzRyRx": R = Rz·Ry·Rx (固定軸 XYZ，多數手臂用這個)
#   "RxRyRz": R = Rx·Ry·Rz (動軸 XYZ)
EULER_CANDIDATES = ["RzRyRx", "RxRyRz"]


def pose_to_T(pose, conv="RzRyRx"):
    x, y, z, rx, ry, rz = pose
    seq = "xyz" if conv == "RzRyRx" else "XYZ"
    T = np.eye(4)
    T[:3, :3] = Rot.from_euler(seq, [rx, ry, rz], degrees=True).as_matrix()
    T[:3, 3] = [x, y, z]
    return T


def rt_to_T(rvec, tvec):
    T = np.eye(4)
    T[:3, :3] = cv2.Rodrigues(np.asarray(rvec, float).reshape(3, 1))[0]
    T[:3, 3] = np.asarray(tvec, float).reshape(3)
    return T


# ---------------- RealSense ----------------
class Camera:
    def __init__(self, width=640, height=480, fps=30):
        import pyrealsense2 as rs
        self.rs = rs
        self.pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        prof = self.pipe.start(cfg)
        self.depth_scale = prof.get_device().first_depth_sensor().get_depth_scale()
        self.align = rs.align(rs.stream.color)          # 深度對齊到彩色影像
        intr = prof.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.K = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]])
        self.D = np.array(intr.coeffs, float)
        for _ in range(30):                              # 等自動曝光穩定
            self.pipe.wait_for_frames()

    def read(self):
        """回傳 (BGR 影像, 深度圖 mm float32，0 = 無效)"""
        fs = self.align.process(self.pipe.wait_for_frames())
        color = np.asanyarray(fs.get_color_frame().get_data()).copy()
        depth = np.asanyarray(fs.get_depth_frame().get_data()).astype(np.float32)
        return color, depth * self.depth_scale * 1000.0

    def read_avg(self, n=10):
        """多張平均：彩色取最後一張，深度取每個像素的中位數 (忽略 0)"""
        ds = []
        for _ in range(n):
            color, d = self.read()
            ds.append(np.where(d > 0, d, np.nan))
        with np.errstate(all="ignore"):
            depth = np.nanmedian(np.stack(ds), axis=0)
        return color, np.nan_to_num(depth, nan=0.0)

    def stop(self):
        self.pipe.stop()


def deproject(K, u, v, z):
    """像素 (u,v) + 深度 z(mm) -> 相機座標 (mm)"""
    return np.array([(u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z, z])


# ---------------- ArUco 位姿 ----------------
_detector = cv2.aruco.ArucoDetector(
    cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100),
    cv2.aruco.DetectorParameters())


def marker_pose(img, K, D, marker_id, length_mm):
    """回傳 (rvec, tvec[mm], corners) 或 None；T_cam<-marker"""
    corners, ids, _ = _detector.detectMarkers(img)
    if ids is None:
        return None
    for c, i in zip(corners, ids.flatten()):
        if int(i) != marker_id:
            continue
        h = length_mm / 2
        obj = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], np.float32)
        ok, rvec, tvec = cv2.solvePnP(obj, c[0].astype(np.float32), K, D,
                                      flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if ok:
            return rvec.ravel(), tvec.ravel(), c[0]
    return None


# ---------------- 筆蓋色彩分割 ----------------
# 依現場照片調整：筆蓋側面陰影很暗 (V 只有 15~100)，所以 V 下限放低；
# 桌上紫藍色墨水痕跡飽和度只有 ~40，靠 S >= 100 濾掉
COLORS = {
    "red":  [((0, 100, 30), (10, 255, 255)), ((170, 100, 30), (180, 255, 255))],
    "blue": [((100, 80, 20), (130, 255, 255))],   # 藍色筆蓋頂部很暗，V 下限要更低
}
DRAW = {"red": (0, 0, 255), "blue": (255, 0, 0)}
MIN_AREA, MAX_AREA = 300, 30000
MIN_CIRCULARITY = 0.55   # 直立筆蓋從上往下看是圓形；桌上墨水痕跡通常是長條，會被濾掉
_kernel = np.ones((5, 5), np.uint8)


def color_mask(hsv, ranges):
    mask = np.zeros(hsv.shape[:2], np.uint8)
    for lo, hi in ranges:
        mask |= cv2.inRange(hsv, np.array(lo), np.array(hi))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _kernel)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _kernel)


def workspace_roi(img, ids_used=(1, 2, 3, 4)):
    """四角 ArUco 圍出的灰色板子範圍 (凸包)。
    只看到 3 個時 (例如被夾爪擋住)，用矩形性質補出第 4 個：
    缺的角 = a + b - c，c 是直角那個頂點 (對著最長邊)"""
    corners, ids, _ = _detector.detectMarkers(img)
    if ids is None:
        return None
    found = {int(i): c[0] for c, i in zip(corners, ids.flatten()) if int(i) in ids_used}
    if len(found) < 3:
        return None
    pts = [p for c in found.values() for p in c]
    if len(found) == 3:
        ctr = [c.mean(axis=0) for c in found.values()]
        sides = [(np.linalg.norm(ctr[(k + 1) % 3] - ctr[(k + 2) % 3]), k) for k in range(3)]
        k = max(sides)[1]                               # 最長邊的對角頂點 = 直角頂點
        pts.append(ctr[(k + 1) % 3] + ctr[(k + 2) % 3] - ctr[k])
    return cv2.convexHull(np.array(pts, np.float32)).astype(np.int32)


def marker_mask(img, grow=15):
    """ArUco 黑色區域 (加大 grow px) 的遮罩；暗像素色相不穩，會被誤判成藍色"""
    m = np.zeros(img.shape[:2], np.uint8)
    corners, ids, _ = _detector.detectMarkers(img)
    if ids is not None:
        for c in corners:
            cv2.fillPoly(m, [c[0].astype(np.int32)], 255)
        m = cv2.dilate(m, np.ones((2 * grow + 1, 2 * grow + 1), np.uint8))
    return m


def segment(img, colors=None, roi=None):
    colors = COLORS if colors is None else colors
    """回傳 [{color, center(u,v), axis(兩端點), contour, mask, elong}]"""
    hsv = cv2.cvtColor(cv2.GaussianBlur(img, (5, 5), 0), cv2.COLOR_BGR2HSV)
    objs = []
    roi_mask = np.full(img.shape[:2], 255, np.uint8)
    if roi is not None:
        roi_mask[:] = 0
        cv2.fillPoly(roi_mask, [roi], 255)
    roi_mask &= ~marker_mask(img)
    for name, ranges in colors.items():
        m = color_mask(hsv, ranges)
        m &= roi_mask
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in cnts:
            area = cv2.contourArea(cnt)
            if not (MIN_AREA < area < MAX_AREA):
                continue
            hull = cv2.convexHull(cnt)                   # 用凸包算圓度，避免筆蓋頂部白色 logo 造成凹洞
            per = cv2.arcLength(hull, True)
            circ = 4 * np.pi * cv2.contourArea(hull) / (per * per) if per > 0 else 0
            if circ < MIN_CIRCULARITY:
                continue
            cnt = hull
            M = cv2.moments(cnt)
            cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]
            _, (w, h), ang = cv2.minAreaRect(cnt)
            th = np.deg2rad(ang if w >= h else ang + 90)
            d = np.array([np.cos(th), np.sin(th)]) * max(w, h) / 2
            om = np.zeros(img.shape[:2], np.uint8)
            cv2.drawContours(om, [cnt], -1, 255, -1)
            om = cv2.erode(om, _kernel)                 # 縮一圈避開邊緣的深度誤差
            objs.append(dict(color=name, center=np.array([cx, cy]),
                             axis=(np.array([cx, cy]) - d, np.array([cx, cy]) + d),
                             contour=cnt, mask=om, area=area, circ=circ,
                             elong=max(w, h) / max(min(w, h), 1)))
    return objs


def object_top(depth, mask):
    """筆蓋頂面：區域內最靠近相機的 50% 像素 → (深度中位數 mm, 頂面中心 u, v)；
    用頂面中心比用整塊輪廓中心準 (輪廓會包含筆蓋側面)"""
    vs, us = np.nonzero((mask > 0) & (depth > 0))
    if vs.size < 20:
        return None
    d = depth[vs, us]
    sel = d <= np.median(d)
    return float(np.median(d[sel])), float(us[sel].mean()), float(vs[sel].mean())


def object_depth(depth, mask):
    """物件區域內有效深度的中位數 (mm)，取最近的 50% (筆蓋頂面)"""
    v = depth[(mask > 0) & (depth > 0)]
    if v.size < 20:
        return None
    v = np.sort(v)[: max(10, v.size // 2)]
    return float(np.median(v))
