from drv_modbus import send
from drv_modbus import request
from pymodbus.client import ModbusTcpClient
import pyrealsense2 as rs
import argparse
import numpy as np
import cv2
import time
import threading

# =====================================================================
# ★ 1. 實測座標與高度設定區 ★
# =====================================================================

# 相機初始位置 (已更新為 midterm.py 的 Home 點位): [X, Y, Z, Rx, Ry, Rz]
HOME_POS = [377.626, -20.001, 555.342, -179.366, 1.545, -106.344]

# 放置區位置 (Place 點位)
RED_PLACE_POS  = [577.925,  66.76,  413.992, 177.836, -0.754, -140.161]
BLUE_PLACE_POS = [577.925, -204.08, 413.992, 177.836, -0.754, -140.161]

# 高度與夾取姿態設定
Z_SAFE  = 486.099  # 移動時的安全高度 (防撞機)
Z_PICK  = 413.992  # 下降夾取筆蓋的高度
Z_PLACE = 413.992  # 下降放置筆蓋的高度
GRIP_ORIENTATION = [177.836, -0.754, -140.161]  # 夾取與放置時的固定姿態 [Rx, Ry, Rz]

# ★ 夾取位置微調補償 (單位: mm) ★
# 如果夾爪下去的位置「有點歪」，請直接修改這裡的數值 (例如偏後就改 OFFSET_X = 5.0 或 -5.0)
OFFSET_X = 0.0
OFFSET_Y = -6.0

# =====================================================================
# ★ 2. ArUco 四角定位設定 (ID 1~4 機械手臂座標) ★
# =====================================================================

ARUCO_ROBOT_POS = {
    1: [350.782, -213.945],  # 左上 ID 1
    2: [350.782,   74.102],  # 右上 ID 2
    3: [596.691, -213.945],  # 左下 ID 3
    4: [596.691,   74.102]   # 右下 ID 4
}

LATEST_H_MATRIX = None

def pixel_to_robot(u, v, H):
    """透過 ArUco 算出的 H 矩陣，將像素 (u, v) 轉換為機械手臂座標 (X, Y) 並加上微調補償"""
    vec = np.array([u, v, 1.0], dtype=np.float32)
    res = np.dot(H, vec)
    X = float(res[0] / res[2]) + OFFSET_X
    Y = float(res[1] / res[2]) + OFFSET_Y
    return X, Y

# =====================================================================
# ★ 3. 機械手臂移動與夾爪控制函式 (修正夾爪開合與到位等待時間) ★
# =====================================================================

def pick_and_place_single(c, pick_x, pick_y, place_pos, color_name):
    """單顆筆蓋的夾取與放置流程 (依照 midterm.py: ON=張開, OFF=夾緊)"""
    rx, ry, rz = GRIP_ORIENTATION
    place_x, place_y = place_pos[0], place_pos[1]
    prx, pry, prz = place_pos[3], place_pos[4], place_pos[5]

    print(f"\n▶ 開始處理【{color_name}筆蓋】 -> 目標夾取座標: X={pick_x:.3f}, Y={pick_y:.3f}")

    # 1. 確保夾爪先張開 (gripper_ON)
    send.gripper_ON(c)
    time.sleep(2)

    # 2. 移至筆蓋正上方安全高度 (Z_SAFE = 486.099) -> 垂直下降至夾取高度 (Z_PICK = 413.992)
    send.Go_Position(c, pick_x, pick_y, Z_SAFE, rx, ry, rz, 70)
    time.sleep(2.5)  # 等待手臂移動到正上方
    send.Go_Position(c, pick_x, pick_y, Z_PICK, rx, ry, rz, 50)
    time.sleep(2.0)  # 【關鍵修正】等待手臂完全下降到位後，才執行夾爪動作！

    # 3. 夾爪閉合夾取 (gripper_OFF)
    send.gripper_OFF(c)
    time.sleep(2)

    # 4. 垂直抬升回安全高度 (防撞機)
    send.Go_Position(c, pick_x, pick_y, Z_SAFE, rx, ry, rz, 50)
    time.sleep(2.0)

    # 5. 移至放置區正上方安全高度 (Z_SAFE) -> 垂直下降至放置高度 (Z_PLACE = 413.992)
    send.Go_Position(c, place_x, place_y, Z_SAFE, prx, pry, prz, 70)
    time.sleep(2.5)  # 等待手臂移動到放置區上方
    send.Go_Position(c, place_x, place_y, Z_PLACE, prx, pry, prz, 50)
    time.sleep(2.0)  # 【關鍵修正】等待手臂下降到位後，才張開夾爪！

    # 6. 夾爪張開放置 (gripper_ON)
    send.gripper_ON(c)
    time.sleep(2)

    # 7. 垂直抬升回安全高度
    send.Go_Position(c, place_x, place_y, Z_SAFE, prx, pry, prz, 50)
    time.sleep(2.0)

def execute_task_thread(c, red_list, blue_list, H_mat, finish_callback):
    """背景執行緒：依序透過 H_mat 換算座標並夾取所有紅色與藍色筆蓋"""
    print("\n================ 開始執行 ArUco 動態夾取任務 ================")
    
    # 依序處理畫面中所有紅色筆蓋
    for (u, v) in red_list:
        px, py = pixel_to_robot(u, v, H_mat)
        print(f"🎯 紅色筆蓋: 像素({u}, {v}) -> 轉換手臂座標({px:.3f}, {py:.3f})")
        pick_and_place_single(c, px, py, RED_PLACE_POS, "紅色")

    # 依序處理畫面中所有藍色筆蓋
    for (u, v) in blue_list:
        px, py = pixel_to_robot(u, v, H_mat)
        print(f"🎯 藍色筆蓋: 像素({u}, {v}) -> 轉換手臂座標({px:.3f}, {py:.3f})")
        pick_and_place_single(c, px, py, BLUE_PLACE_POS, "藍色")

    # 任務完成，回到相機初始位置 HOME_POS
    print("\n✅ 所有筆蓋夾取完畢，返回相機初始位置 (Home)...")
    send.Go_Position(c, HOME_POS[0], HOME_POS[1], HOME_POS[2], HOME_POS[3], HOME_POS[4], HOME_POS[5], 50)
    time.sleep(2.5)
    finish_callback()

# =====================================================================
# ★ 4. 視覺辨識 (ArUco + 紅藍筆蓋) 與主程式 ★
# =====================================================================

def get_centroids(mask, output_img, draw_color, label, H_mat):
    coords = []
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) > 500:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                coords.append((cx, cy))

                cv2.drawContours(output_img, [contour], -1, draw_color, 3)
                cv2.circle(output_img, (cx, cy), 5, (0, 255, 0), -1)
                
                if H_mat is not None:
                    rx, ry = pixel_to_robot(cx, cy, H_mat)
                    text = f"{label} X:{rx:.1f} Y:{ry:.1f}"
                else:
                    text = f"{label} ({cx},{cy})"
                    
                cv2.putText(output_img, text, (cx - 50, cy - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, draw_color, 2)
    return coords

def start_camera():
    """延後開啟相機；Windows 設定目錄無法讀取時，改用 SDK 預設設定。"""
    try:
        context = rs.context()
    except RuntimeError as exc:
        if "SHGetKnownFolderPath" not in str(exc):
            raise
        print("RealSense 無法讀取 Windows 設定目錄，改用不繼承設定的 context。")
        context = rs.context('{"inherit": false}')

    pipeline = rs.pipeline(context)
    config = rs.config()
    # 明確指定 RGB，與後面的 COLOR_RGB2BGR 對應。
    #config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
    try:
        pipeline.start(config)
    except RuntimeError as exc:
        if "0x80070005" in str(exc):
            raise RuntimeError(
                "RealSense 相機存取遭拒 (0x80070005)。請確認 Windows 設定 > "
                "隱私權與安全性 > 相機，已開啟相機存取及桌面應用程式存取，"
                "並關閉其他使用相機的程式；若在受限環境執行，請改在本機終端機測試。"
            ) from exc
        if "No device connected" in str(exc):
            raise RuntimeError(
                "RealSense 未找到可開啟的相機。請確認 USB 連線、相機權限，"
                "並關閉其他使用相機的程式，再從本機終端機執行。"
            ) from exc
        raise
    return pipeline


def main(camera_only=False):
    pipeline = None
    c = None
    workers = []
    try:
        # 先確認相機能啟動，再連線與移動手臂。
        pipeline = start_camera()
        if not camera_only:
            c = ModbusTcpClient(host="192.168.1.1", port=502, timeout=3)
            if not c.connect():
                raise RuntimeError("無法連線至手臂 192.168.1.1:502，請檢查電源與網路。")
        run_vision(c, pipeline, workers)
    finally:
        # 避免背景動作尚未結束就關閉 Modbus 連線。
        for worker in workers:
            worker.join()
        if pipeline is not None:
            pipeline.stop()
        cv2.destroyAllWindows()
        if c is not None:
            c.close()


def run_vision(c, pipeline, workers):
    global LATEST_H_MATRIX
    LATEST_H_MATRIX = None

    # 初始化 ArUco 偵測器 (DICT_5X5_100)
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
    if hasattr(cv2.aruco, "ArucoDetector"):
        parameters = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
        detect_markers = detector.detectMarkers
    else:
        parameters = cv2.aruco.DetectorParameters_create()

        def detect_markers(image):
            return cv2.aruco.detectMarkers(image, aruco_dict, parameters=parameters)

    if c is not None:
        print("手臂初始化：張開夾爪並移動至相機初始位置 (Home)...")
        send.gripper_ON(c)
        time.sleep(1.0)
        worker = threading.Thread(target=send.Go_Position, args=(c, *HOME_POS, 50))
        workers.append(worker)
        worker.start()
    else:
        print("相機測試模式：僅顯示影像與座標，不連線或控制手臂。")

    # 定義紅、藍 HSV 範圍
    lower_red1, upper_red1 = np.array([0, 120, 70]), np.array([10, 255, 255])
    lower_red2, upper_red2 = np.array([170, 120, 70]), np.array([180, 255, 255])
    # lower_blue, upper_blue = np.array([100, 120, 70]), np.array([130, 255, 255])
    lower_blue, upper_blue = np.array([85, 80, 50]), np.array([130, 255, 255])
    kernel = np.ones((5, 5), np.uint8)

    is_busy = False

    def on_task_finished():
        nonlocal is_busy
        is_busy = False
        print("🔄 任務完成已解鎖！可按 'g' 進行下一輪夾取，按 'p' 讀取座標，或按 'q' 離開。")

    print("==========================================")
    print("ArUco 自動定位夾取系統啟動！")
    print(" 👉 確認畫面左上角顯示【ArUco: 4/4 (Calibrated OK)】")
    print(" 👉 按 【g 鍵】：鎖定筆蓋動態座標並開始夾取與放置")
    print(" 👉 按 【p 鍵】：印出當前手臂 TCP 座標與筆蓋轉換後座標")
    print(" 👉 按 【q 鍵】：結束程式")
    print("==========================================")

    while True:
        frames = pipeline.wait_for_frames(5000)
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue
        frame = np.asanyarray(color_frame.get_data())

        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        output = frame_bgr.copy()

        # ---------------- 1. 偵測 4 張 ArUco 並即時更新轉換矩陣 ----------------
        corners, ids, _ = detect_markers(frame_bgr)
        detected_aruco = {}

        if ids is not None:
            cv2.aruco.drawDetectedMarkers(output, corners, ids)
            for i, marker_id in enumerate(ids.flatten()):
                m_id = int(marker_id)
                if m_id in ARUCO_ROBOT_POS:
                    pts = corners[i][0]
                    cx = float(np.mean(pts[:, 0]))
                    cy = float(np.mean(pts[:, 1]))
                    detected_aruco[m_id] = [cx, cy]
                    cv2.circle(output, (int(cx), int(cy)), 4, (0, 255, 255), -1)

        if len(detected_aruco) == 4 and not is_busy:
            pts_pixel = np.float32([detected_aruco[1], detected_aruco[2], detected_aruco[3], detected_aruco[4]])
            pts_robot = np.float32([ARUCO_ROBOT_POS[1], ARUCO_ROBOT_POS[2], ARUCO_ROBOT_POS[3], ARUCO_ROBOT_POS[4]])
            LATEST_H_MATRIX, _ = cv2.findHomography(pts_pixel, pts_robot)

        # ---------------- 2. 偵測紅、藍筆蓋並即時顯示機械座標 ----------------
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

        mask_red = cv2.bitwise_or(cv2.inRange(hsv, lower_red1, upper_red1),
                                  cv2.inRange(hsv, lower_red2, upper_red2))
        mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel)

        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
        mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_OPEN, kernel)

        red_coords = get_centroids(mask_red, output, (0, 0, 255), "Red", LATEST_H_MATRIX)
        blue_coords = get_centroids(mask_blue, output, (255, 0, 0), "Blue", LATEST_H_MATRIX)

        # ---------------- 3. 畫面狀態顯示 ----------------
        status_text = "Status: BUSY (Moving...)" if is_busy else "Status: READY ('G':Pick, 'P':Pose)"
        status_color = (0, 0, 255) if is_busy else (0, 255, 0)
        cv2.putText(output, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

        aruco_ok = (LATEST_H_MATRIX is not None)
        aruco_text = f"ArUco: {len(detected_aruco)}/4 ({'Calibrated OK' if aruco_ok else 'Need 4 Markers!'})"
        aruco_color = (0, 255, 0) if len(detected_aruco) == 4 else (0, 165, 255)
        cv2.putText(output, aruco_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.65, aruco_color, 2)

        cv2.imshow("RealSense Cap Pick & Place (ArUco)", output)

        # ---------------- 4. 鍵盤控制 ----------------
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("結束程式。")
            break
        elif key == ord('p'):
            if c is None:
                print("相機測試模式無法讀取手臂 TCP；筆蓋座標可查看影像標示。")
                continue
            x, y, z, rx, ry, rz = request.Get_TCP_Pose(c)
            print(f"📍 當前手臂座標: [{x:.3f}, {y:.3f}, {z:.3f}, {rx:.3f}, {ry:.3f}, {rz:.3f}]")
            if LATEST_H_MATRIX is not None:
                for (u, v) in red_coords:
                    rx_p, ry_p = pixel_to_robot(u, v, LATEST_H_MATRIX)
                    print(f"   🔴 紅色筆蓋像素({u},{v}) -> 預測手臂座標: X={rx_p:.3f}, Y={ry_p:.3f}")
                for (u, v) in blue_coords:
                    bx_p, by_p = pixel_to_robot(u, v, LATEST_H_MATRIX)
                    print(f"   🔵 藍色筆蓋像素({u},{v}) -> 預測手臂座標: X={bx_p:.3f}, Y={by_p:.3f}")
        elif key == ord('g'):
            if c is None:
                print("相機測試模式不執行夾取。")
                continue
            if not is_busy:
                if LATEST_H_MATRIX is None:
                    print("⚠️ 尚未完成 ArUco 定位！請確保畫面中能同時看到 ID 1, 2, 3, 4 四張標籤。")
                elif len(red_coords) == 0 and len(blue_coords) == 0:
                    print("⚠️ 畫面中未偵測到任何紅、藍筆蓋！")
                else:
                    is_busy = True
                    r_targets = list(red_coords)
                    b_targets = list(blue_coords)
                    h_copy = LATEST_H_MATRIX.copy()
                    worker = threading.Thread(
                        target=execute_task_thread,
                        args=(c, r_targets, b_targets, h_copy, on_task_finished)
                    )
                    workers.append(worker)
                    worker.start()
            else:
                print("⏳ 手臂正在移動中，請稍候...")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RealSense ArUco 筆蓋夾取測試")
    parser.add_argument("--camera-only", action="store_true", help="只測試影像，不控制手臂")
    args = parser.parse_args()
    main(camera_only=args.camera_only)
