"""
方法二：Eye-in-Hand 手眼校正 + 深度 → 直接算出筆蓋的手臂座標並夾取放置
(針對：RealSense 裝在夾爪旁朝下、AirTAC 平行氣壓夾爪、筆蓋直立放在灰色板上)

  T_base<-cam = T_base<-gripper(拍照當下 TCP) · T_gripper<-cam(手眼校正)
  P_base      = T_base<-cam · deproject(u, v, depth)

深度無效或不合理時 (相機離太近、反光)，改用「像素射線 ∩ 筆蓋頂面平面」：
  筆蓋頂面高度 = 桌面高度 (手眼校正時 ArUco 的 z) + CAP_HEIGHT

用法：
  python pen_cap_handeye.py --dry     # 只辨識、印出座標，不夾取
  python pen_cap_handeye.py           # 正式執行；畫面中按 g 開始，q 離開
需先完成 handeye_calib.py 產生 handeye.json
"""
import argparse
import json
import os
import time

import cv2
import numpy as np

import pen_vision as pv

# ======================= 參數設定 =======================
ROBOT_IP = "192.168.1.1"
HOME = [366.165, 72.886, 448.785, 177.263, -0.626, -125.997]   # 原點 (拍照點)
RX, RY, RZ = HOME[3], HOME[4], HOME[5]
SPEED_FAST = 50       # 空中移動
SPEED_SLOW = 20       # 下降 / 上升 (避免撞倒筆蓋)

SAFE_Z = 448.785      # 移動高度 (mm)

# ---- 筆蓋與桌面 (請用尺量) ----
CAP_HEIGHT = 30.0     # 筆蓋直立高度 (mm)，已實測 3 cm
GRASP_DEPTH = 15.0    # 夾爪尖端從筆蓋頂面往下多少 mm 夾 (約筆蓋高度一半)
TIP_CLEARANCE = 3.0   # 夾爪尖端至少離桌面多少 mm
TABLE_Z = None        # 桌面高度；None = 自動用手眼校正時 ArUco 的 z
TCP_Z_OFFSET = 0.0    # 若控制器的 TCP 設在法蘭而不是夾爪尖端，填夾爪長度 (mm)
DEPTH_TOL = 12.0      # 深度量到的頂面高度與預期差超過這個值，就改用平面法

# ---- 放置點 (請改成助教指定位置) ----
DROP = {"red": (263.180, -145.766), "blue": (503.075, 144.214)}
DROP_STEP = 30.0      # 同色第 2、3 個往 y 方向偏移
PLACE_RELEASE_H = 5.0 # 放置時筆蓋底部離桌面多少 mm 鬆開

# ---- 避開相鄰筆蓋：旋轉夾爪讓手指閉合方向垂直於「兩筆蓋連線」 ----
AVOID_NEIGHBOR = False   # 確認 FINGER_AXIS 後再開啟
FINGER_AXIS = "x"        # 夾爪閉合方向是工具座標的 x 還是 y 軸
NEIGHBOR_DIST = 50.0     # 鄰居距離小於這個值 (mm) 才需要旋轉

HERE = os.path.dirname(os.path.abspath(__file__))


# ======================= 座標計算 =======================
def load_handeye():
    f = os.path.join(HERE, "handeye.json")
    if not os.path.exists(f):
        raise SystemExit("找不到 handeye.json，請先執行 handeye_calib.py")
    d = json.load(open(f))
    if d["mode"] != "eye_in_hand":
        raise SystemExit("此程式針對 eye_in_hand，請用 --mode eye_in_hand 重新校正")
    table_z = TABLE_Z if TABLE_Z is not None else d["marker"][2]
    print(f"手眼校正: 尤拉角 {d['euler']}, 誤差 {d['err_mean']:.2f} mm, 桌面 z = {table_z:.1f}")
    return d["euler"], np.array(d["T"]), table_z


def ray_plane(T_bc, K, u, v, z_plane):
    """像素射線與 z = z_plane 水平面的交點"""
    d = T_bc[:3, :3] @ np.array([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], 1.0])
    o = T_bc[:3, 3]
    if abs(d[2]) < 1e-6:
        return None
    s = (z_plane - o[2]) / d[2]
    return o + s * d if s > 0 else None


def locate(cam, T_bc, table_z):
    img, depth = cam.read_avg(10)
    out = img.copy()
    roi = pv.workspace_roi(img)
    if roi is not None:
        cv2.polylines(out, [roi], True, (0, 255, 255), 1)
    top_z = table_z + CAP_HEIGHT
    result = []
    for o in pv.segment(img, roi=roi):
        u, v = o["center"]
        P, src = None, "plane"
        top = pv.object_top(depth, o["mask"])
        if top is not None:
            z, ut, vt = top
            Pd = (T_bc @ np.append(pv.deproject(cam.K, ut, vt, z), 1))[:3]
            h = Pd[2] - table_z
            if h < CAP_HEIGHT * 0.4:               # 幾乎貼平桌面 → 墨水痕跡等，不是筆蓋
                continue
            if abs(Pd[2] - top_z) <= DEPTH_TOL:
                P, src = Pd, "depth"
        if P is None:                              # 深度無效或不合理 → 平面法
            P = ray_plane(T_bc, cam.K, u, v, top_z)
            if P is None:
                continue
        o["P"], o["src"] = P, src
        result.append(o)
        cv2.drawContours(out, [o["contour"]], -1, pv.DRAW[o["color"]], 2)
        cv2.circle(out, (int(u), int(v)), 4, (0, 255, 0), -1)
        cv2.putText(out, f"{o['color']} ({P[0]:.0f},{P[1]:.0f},{P[2]:.0f}) {src}",
                    (int(u) - 90, int(v) - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    pv.DRAW[o["color"]], 2)
    # 相鄰資訊：最近鄰距離與方向；最孤立的先夾
    for o in result:
        others = [p for p in result if p is not o]
        if others:
            n = min(others, key=lambda p: np.linalg.norm(p["P"][:2] - o["P"][:2]))
            vec = n["P"][:2] - o["P"][:2]
            o["nn_dist"] = float(np.linalg.norm(vec))
            o["nn_angle"] = float(np.degrees(np.arctan2(vec[1], vec[0])))
        else:
            o["nn_dist"], o["nn_angle"] = 1e9, 0.0
    result.sort(key=lambda o: -o["nn_dist"])
    return result, out


def finger_angle(conv, rz):
    """工具姿態 (RX,RY,rz) 下，夾爪閉合方向在基座 xy 平面的角度 (deg, 0~180)"""
    R = pv.pose_to_T([0, 0, 0, RX, RY, rz], conv)[:3, :3]
    f = R[:, 0] if FINGER_AXIS == "x" else R[:, 1]
    return np.degrees(np.arctan2(f[1], f[0])) % 180


def choose_rz(conv, o):
    """讓手指閉合方向垂直於鄰居方向，rz 在 RZ±90° 內找最接近 RZ 的解"""
    if not AVOID_NEIGHBOR or o["nn_dist"] > NEIGHBOR_DIST:
        return RZ
    want = (o["nn_angle"] + 90) % 180
    best, best_cost = RZ, 1e9
    for drz in np.arange(-90, 90.5, 0.5):
        diff = abs((finger_angle(conv, RZ + drz) - want + 90) % 180 - 90)
        cost = diff * 10 + abs(drz) * 0.01
        if cost < best_cost:
            best, best_cost = RZ + drz, cost
    return (best + 180) % 360 - 180


# ======================= 手臂 =======================
def connect():
    from pymodbus.client import ModbusTcpClient
    c = ModbusTcpClient(host=ROBOT_IP, port=502, unit_id=2)
    if not c.connect():
        raise SystemExit("無法連線到手臂")
    return c


def go(c, x, y, z, rz=RZ, speed=SPEED_FAST):
    from drv_modbus import send
    send.Go_Position(c, x, y, z, RX, RY, rz, speed)


def pick_place(c, o, drop_xy, table_z, conv):
    from drv_modbus import send
    x, y, ztop = o["P"]
    z_tip = max(ztop - GRASP_DEPTH, table_z + TIP_CLEARANCE)   # 夾爪尖端高度
    rz = choose_rz(conv, o)
    print(f"[PICK] {o['color']} x={x:.1f} y={y:.1f} 頂面z={ztop:.1f}({o['src']}) "
          f"夾取z={z_tip:.1f} rz={rz:.1f}")
    send.gripper_OFF(c); time.sleep(0.5)                 # 張開
    go(c, x, y, SAFE_Z, rz)
    go(c, x, y, z_tip + TCP_Z_OFFSET, rz, SPEED_SLOW)
    send.gripper_ON(c); time.sleep(1.0)                  # 夾緊
    go(c, x, y, SAFE_Z, rz, SPEED_SLOW)
    # 放置：夾著時尖端比筆蓋底部高 (z_tip - table_z)，讓筆蓋底部停在桌面上方 PLACE_RELEASE_H
    zp = z_tip + PLACE_RELEASE_H + TCP_Z_OFFSET
    print(f"[PLACE] -> ({drop_xy[0]:.1f}, {drop_xy[1]:.1f}) z={zp:.1f}")
    go(c, drop_xy[0], drop_xy[1], SAFE_Z, rz)
    go(c, drop_xy[0], drop_xy[1], zp, rz, SPEED_SLOW)
    send.gripper_OFF(c); time.sleep(0.8)                 # 放開
    go(c, drop_xy[0], drop_xy[1], SAFE_Z, rz, SPEED_SLOW)


def main(dry):
    from drv_modbus import request
    conv, X, table_z = load_handeye()
    c = connect()
    cam = pv.Camera()
    go(c, *HOME[:3])
    count = {k: 0 for k in DROP}
    started = False
    print("畫面中按 g 開始 (dry 模式只印座標)，q 離開")
    while True:
        time.sleep(0.3)                                   # 手臂停穩再拍
        tcp = request.Get_TCP_Pose(c)                     # 用實際 TCP，不用指令值
        T_bc = pv.pose_to_T(tcp, conv) @ X
        objs, out = locate(cam, T_bc, table_z)
        cv2.putText(out, f"caps: {len(objs)}  [g]start [q]quit", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow("pen caps (hand-eye)", out)
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        if k == ord("g"):
            started = True
            for o in objs:
                print(f"  {o['color']:5s} x={o['P'][0]:.2f} y={o['P'][1]:.2f} "
                      f"z={o['P'][2]:.2f} ({o['src']})  最近鄰 {o['nn_dist']:.0f} mm")
        if not started:
            continue
        if dry:
            started = False
            continue
        if not objs:
            print("全部完成！"); started = False
            continue
        o = objs[0]
        dx, dy = DROP[o["color"]]
        pick_place(c, o, (dx, dy + count[o["color"]] * DROP_STEP), table_z, conv)
        count[o["color"]] += 1
        go(c, *HOME[:3])
    cv2.destroyAllWindows()
    go(c, *HOME[:3])
    cam.stop()
    c.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    main(ap.parse_args().dry)
