"""
筆蓋色彩辨識與夾取放置 (Eye-to-Hand 平面手眼校正 + HSV 物件分割)

流程：
  1. 手眼校正 (--calib)：把夾爪尖端依序移到 ArUco ID1~ID4 的中心，
     程式讀取手臂 TCP 座標存成 calib.json。
     執行時偵測四個 ArUco 的像素中心，與其手臂座標建立 Homography
     → 任意像素 (u, v) 可換算成手臂平面座標 (x, y)。
  2. 物件分割：HSV 色彩閾值 + 形態學 + 輪廓，只保留 ArUco 四角圍成區域內的物件。
  3. 夾取放置：每種顏色放到各自的放置點，每夾一個就回 Home 重新拍照辨識。

用法：
  python pen_cap_pick_place.py --tune     # 調整 HSV 閾值 (不動手臂)
  python pen_cap_pick_place.py --calib    # 手眼校正，紀錄 4 個 ArUco 的手臂座標
  python pen_cap_pick_place.py --dry      # 只辨識 + 印出手臂座標，不動手臂
  python pen_cap_pick_place.py            # 正式執行 (畫面中按 g 開始, q 離開)
"""
import argparse
import json
import os
import time

import cv2
import numpy as np

from realsense import realsense

# ======================= 參數設定 (依現場修改) =======================
ROBOT_IP = "192.168.1.1"

# 拍照 / 待命姿態 (相機必須每次都在同一個位置拍照，校正才有效)
HOME = [366.165, 72.886, 448.785, 177.263, -0.626, -125.997]   # 原點 (拍照點)
RX, RY, RZ = HOME[3], HOME[4], HOME[5]          # 夾取時的工具姿態

SAFE_Z = 448.785      # 移動時的安全高度 (mm)，與原點同高
PICK_Z = 409.667      # 夾取高度 (mm)，請用實際筆蓋高度微調
PLACE_Z = 420.0       # 放置高度 (mm)
SPEED = 50

# 各顏色的放置點 (x, y)；同色多個時，沿 y 方向每個偏移 DROP_STEP
DROP = {
    "red":  (263.180, -145.766),
    "blue": (503.075,  144.214),
}
DROP_STEP = 30.0

# 是否依筆蓋長軸方向旋轉夾爪 (筆蓋躺著時建議開啟，需先實測角度方向)
ROTATE_GRIPPER = False
ANGLE_SIGN = 1.0      # 若夾爪轉反方向，改成 -1.0

# HSV 色彩範圍 (H: 0~180, S: 0~255, V: 0~255)，可用 --tune 調整後貼回來
COLORS = {
    "red":  [((0, 120, 70), (10, 255, 255)), ((170, 120, 70), (180, 255, 255))],
    "blue": [((100, 120, 70), (130, 255, 255))],
}
DRAW = {"red": (0, 0, 255), "blue": (255, 0, 0)}
MIN_AREA, MAX_AREA = 300, 20000                 # 輪廓面積過濾 (pixel)

ARUCO_IDS = [1, 2, 3, 4]                        # 1 左上, 2 右上, 3 左下, 4 右下
CALIB_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calib.json")

# ======================= ArUco / 手眼校正 =======================
_detector = cv2.aruco.ArucoDetector(
    cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100),
    cv2.aruco.DetectorParameters())


def get_frame():
    return cv2.cvtColor(realsense.Get_RGB_Frame(), cv2.COLOR_RGB2BGR)


def detect_aruco_centers(frame):
    """回傳 {id: (u, v)} 每個 ArUco 的像素中心"""
    corners, ids, _ = _detector.detectMarkers(frame)
    result = {}
    if ids is not None:
        for c, i in zip(corners, ids.flatten()):
            result[int(i)] = c[0].mean(axis=0)
    return result


def load_calib():
    if not os.path.exists(CALIB_FILE):
        raise SystemExit(f"找不到 {CALIB_FILE}，請先執行 --calib 做手眼校正")
    with open(CALIB_FILE, "r") as f:
        data = json.load(f)
    return {int(k): np.array(v, dtype=np.float32) for k, v in data.items()}


def compute_homography(frame, robot_pts):
    """用四個 ArUco 中心 (像素) 與其手臂座標 (mm) 求 Homography"""
    centers = detect_aruco_centers(frame)
    if not all(i in centers for i in ARUCO_IDS):
        return None, None
    src = np.float32([centers[i] for i in ARUCO_IDS])
    dst = np.float32([robot_pts[i] for i in ARUCO_IDS])
    H = cv2.getPerspectiveTransform(src, dst)
    roi = np.int32([centers[1], centers[2], centers[4], centers[3]])  # 依順時針圍成工作區
    return H, roi


def pixel_to_robot(H, pts):
    pts = np.float32(pts).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, H).reshape(-1, 2)


# ======================= 色彩分割 =======================
_kernel = np.ones((5, 5), np.uint8)


def color_mask(hsv, ranges):
    mask = np.zeros(hsv.shape[:2], np.uint8)
    for lo, hi in ranges:
        mask |= cv2.inRange(hsv, np.array(lo), np.array(hi))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _kernel)
    return mask


def segment(frame, roi=None):
    """回傳 list of dict: color, center(u,v), axis_pts(長軸兩端), contour"""
    hsv = cv2.cvtColor(cv2.GaussianBlur(frame, (5, 5), 0), cv2.COLOR_BGR2HSV)
    roi_mask = None
    if roi is not None:
        roi_mask = np.zeros(frame.shape[:2], np.uint8)
        cv2.fillPoly(roi_mask, [roi], 255)

    objs, masks = [], {}
    for name, ranges in COLORS.items():
        mask = color_mask(hsv, ranges)
        if roi_mask is not None:
            mask &= roi_mask
        masks[name] = mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if not (MIN_AREA < area < MAX_AREA):
                continue
            M = cv2.moments(cnt)
            cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]
            (rx, ry), (w, h), ang = cv2.minAreaRect(cnt)
            # 長軸方向的單位向量
            theta = np.deg2rad(ang if w >= h else ang + 90)
            L = max(w, h) / 2
            d = np.array([np.cos(theta), np.sin(theta)]) * L
            objs.append(dict(color=name, center=(cx, cy), area=area,
                             axis=(np.array([cx, cy]) - d, np.array([cx, cy]) + d),
                             contour=cnt, elong=max(w, h) / max(min(w, h), 1)))
    return objs, masks


def locate(frame, robot_pts):
    """辨識 + 轉換成手臂座標，回傳 (objs, 畫好的影像)"""
    out = frame.copy()
    H, roi = compute_homography(frame, robot_pts)
    if H is None:
        cv2.putText(out, "ArUco 1~4 not all found", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return [], out
    cv2.polylines(out, [roi], True, (0, 255, 255), 2)
    objs, _ = segment(frame, roi)
    for o in objs:
        xy = pixel_to_robot(H, [o["center"]])[0]
        a = pixel_to_robot(H, list(o["axis"]))
        o["xy"] = xy
        o["angle"] = np.degrees(np.arctan2(a[1][1] - a[0][1], a[1][0] - a[0][0]))
        u, v = map(int, o["center"])
        cv2.drawContours(out, [o["contour"]], -1, DRAW[o["color"]], 2)
        cv2.circle(out, (u, v), 4, (0, 255, 0), -1)
        cv2.putText(out, f"{o['color']} ({xy[0]:.1f},{xy[1]:.1f})", (u - 60, v - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, DRAW[o["color"]], 2)
    return objs, out


def stable_locate(robot_pts, n=5):
    """連續拍 n 張取座標中位數，降低雜訊"""
    samples, out = [], None
    for _ in range(n):
        objs, out = locate(get_frame(), robot_pts)
        samples.append(objs)
    base = max(samples, key=len)
    for o in base:
        near = [p["xy"] for s in samples for p in s
                if p["color"] == o["color"] and np.linalg.norm(p["xy"] - o["xy"]) < 10]
        o["xy"] = np.median(near, axis=0)
    return base, out


# ======================= 手臂動作 =======================
def connect_robot():
    from pymodbus.client import ModbusTcpClient
    c = ModbusTcpClient(host=ROBOT_IP, port=502, unit_id=2)
    if not c.connect():
        raise SystemExit("無法連線到手臂 " + ROBOT_IP)
    return c


def go(c, x, y, z, rz=RZ):
    from drv_modbus import send
    send.Go_Position(c, x, y, z, RX, RY, rz, SPEED)


def pick_and_place(c, obj, drop_xy):
    from drv_modbus import send
    x, y = map(float, obj["xy"])
    rz = RZ
    if ROTATE_GRIPPER and obj["elong"] > 1.3:          # 有明顯長軸才旋轉
        rz = RZ + ANGLE_SIGN * obj["angle"] + 90        # 夾爪垂直於長軸夾取
        rz = (rz + 180) % 360 - 180
    print(f"[PICK] {obj['color']} at ({x:.1f}, {y:.1f}) rz={rz:.1f}")
    send.gripper_OFF(c); time.sleep(0.5)               # 先張開
    go(c, x, y, SAFE_Z, rz)
    go(c, x, y, PICK_Z, rz)
    send.gripper_ON(c); time.sleep(1.5)                # 夾緊
    go(c, x, y, SAFE_Z, rz)
    print(f"[PLACE] -> ({drop_xy[0]:.1f}, {drop_xy[1]:.1f})")
    go(c, drop_xy[0], drop_xy[1], SAFE_Z)
    go(c, drop_xy[0], drop_xy[1], PLACE_Z)
    send.gripper_OFF(c); time.sleep(1.0)               # 放開
    go(c, drop_xy[0], drop_xy[1], SAFE_Z)


# ======================= 各模式 =======================
def mode_tune():
    """HSV 閾值即時調整"""
    name = input("要調整哪個顏色? (red/blue): ").strip() or "blue"
    lo, hi = COLORS[name][0]
    cv2.namedWindow("tune")
    for i, ch in enumerate("HSV"):
        mx = 180 if ch == "H" else 255
        cv2.createTrackbar(f"{ch} low", "tune", lo[i], mx, lambda v: None)
        cv2.createTrackbar(f"{ch} high", "tune", hi[i], mx, lambda v: None)
    print("調整滑桿，按 p 印出數值，q 離開。(紅色有兩段 H 範圍，第二段請另外調)")
    while True:
        frame = get_frame()
        hsv = cv2.cvtColor(cv2.GaussianBlur(frame, (5, 5), 0), cv2.COLOR_BGR2HSV)
        lo = tuple(cv2.getTrackbarPos(f"{ch} low", "tune") for ch in "HSV")
        hi = tuple(cv2.getTrackbarPos(f"{ch} high", "tune") for ch in "HSV")
        mask = color_mask(hsv, [(lo, hi)])
        cv2.imshow("tune", np.hstack([frame, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)]))
        k = cv2.waitKey(1) & 0xFF
        if k == ord("p"):
            print(f'"{name}": [({lo}, {hi})],')
        elif k == ord("q"):
            break
    cv2.destroyAllWindows()


def mode_calib():
    """依序把夾爪尖端對準 ArUco 中心，紀錄 TCP 座標"""
    from drv_modbus import request
    c = connect_robot()
    pts = {}
    for i in ARUCO_IDS:
        input(f"請把夾爪尖端移到 ArUco ID {i} 的中心 (用教導器或 keyboard_jog_example)，完成後按 Enter...")
        x, y, z, *_ = request.Get_TCP_Pose(c)
        pts[i] = [x, y]
        print(f"  ID {i}: x={x:.3f}, y={y:.3f}, z={z:.3f}  (此 z 可作為 PICK_Z 參考)")
    with open(CALIB_FILE, "w") as f:
        json.dump(pts, f, indent=2)
    print("已儲存", CALIB_FILE)
    print("提醒：手臂回 HOME 後再執行，四個 ArUco 都要在畫面中。")
    c.close()


def mode_run(dry=False):
    robot_pts = load_calib()
    c = None
    if not dry:
        c = connect_robot()
        go(c, *HOME[:3])
    for _ in range(30):                                # 丟掉前幾張讓自動曝光穩定
        realsense.Get_RGB_Frame()

    print("畫面中按 g 開始夾取, q 離開")
    started, count = False, {k: 0 for k in DROP}
    while True:
        objs, out = locate(get_frame(), robot_pts)
        cv2.putText(out, f"red:{sum(o['color']=='red' for o in objs)} "
                         f"blue:{sum(o['color']=='blue' for o in objs)}  [g]start [q]quit",
                    (10, out.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow("pen caps", out)
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        if k == ord("g"):
            started = True
        if not started:
            continue

        objs, out = stable_locate(robot_pts)
        cv2.imshow("pen caps", out); cv2.waitKey(1)
        if not objs:
            print("所有筆蓋都已完成！")
            started = False
            continue
        for o in objs:
            print(f"  {o['color']:5s} robot xy = ({o['xy'][0]:.2f}, {o['xy'][1]:.2f}) mm")
        if dry:
            started = False
            continue
        target = objs[0]
        dx, dy = DROP[target["color"]]
        drop_xy = (dx, dy + count[target["color"]] * DROP_STEP)
        pick_and_place(c, target, drop_xy)
        count[target["color"]] += 1
        go(c, *HOME[:3])                               # 回拍照點，重新辨識
        time.sleep(0.5)
        for _ in range(5):
            realsense.Get_RGB_Frame()

    cv2.destroyAllWindows()
    if c:
        go(c, *HOME[:3])
        c.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tune", action="store_true", help="調整 HSV 閾值")
    ap.add_argument("--calib", action="store_true", help="手眼校正")
    ap.add_argument("--dry", action="store_true", help="只辨識不動手臂")
    args = ap.parse_args()
    if args.tune:
        mode_tune()
    elif args.calib:
        mode_calib()
    else:
        mode_run(dry=args.dry)
