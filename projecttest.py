from drv_modbus import send
from drv_modbus import request
from pymodbus.client import ModbusTcpClient
from realsense import realsense
import numpy as np
import cv2
import time
import threading

# =====================================================================
# ★ 1. 實測座標與高度設定區 ★
# =====================================================================

# 相機初始位置 (Home 點位): [X, Y, Z, Rx, Ry, Rz]
HOME_POS = [382.004, -20.001, 555.342, -179.366, 1.545, -106.344]

# 放置區位置 (Place 點位)
RED_PLACE_POS  = [577.925,  66.76,  413.992, 177.836, -0.754, -140.161]
BLUE_PLACE_POS = [577.925, -204.08, 413.992, 177.836, -0.754, -140.161]

# 高度與夾取姿態設定
Z_SAFE  = 486.099  # 移動時的安全高度 (防撞機)
Z_PICK  = 413.992  # 下降夾取筆蓋的高度
Z_PLACE = 413.992  # 下降放置筆蓋的高度
GRIP_ORIENTATION = [177.836, -0.754, -140.161]  # 夾取與放置時的固定姿態 [Rx, Ry, Rz]

# ★ 夾取位置偏移量微調 (單位: mm) <-- 在這裡修改！ ★
# 如果夾爪下去偏後/偏前，請加減 OFFSET_X；如果偏左/偏右，請加減 OFFSET_Y
OFFSET_X = 0.0
OFFSET_Y = -5.0

# =====================================================================
# ★ 2. ArUco 四角定位設定 (已填入你實測的 ID 1~4 機械手臂座標) ★
# =====================================================================

ARUCO_ROBOT_POS = {
    1: [350.782, -213.945],  # 左上 ID 1
    2: [350.782,   74.102],  # 右上 ID 2
    3: [596.691, -213.945],  # 左下 ID 3
    4: [596.691,   74.102]   # 右下 ID 4
}

# 儲存最新計算出的單應性轉換矩陣 H (當成功看到 4 張 ArUco 時自動更新)
LATEST_H_MATRIX = None

def pixel_to_robot(u, v, H):
    """透過 ArUco 算出的 H 矩陣，將像素 (u, v) 轉換為機械手臂座標 (X, Y) 並加上偏移量"""
    vec = np.array([u, v, 1.0], dtype=np.float32)
    res = np.dot(H, vec)
    X = float(res[0] / res[2]) + OFFSET_X  # <-- 自動加上 X 偏移量
    Y = float(res[1] / res[2]) + OFFSET_Y  # <-- 自動加上 Y 偏移量
    return X, Y

# =====================================================================
# ★ 3. 機械手臂移動與夾爪控制函式 (ON=張開, OFF=夾緊) ★
# =====================================================================

def pick_and_place_single(c, pick_x, pick_y, place_pos, color_name):
    """單顆筆蓋的夾取與放置流程 (三段式垂直升降防撞機)"""
    rx, ry, rz = GRIP_ORIENTATION
    place_x, place_y = place_pos[0], place_pos[1]
    prx, pry, prz = place_pos[3], place_pos[4], place_pos[5]

    print(f"\n▶ 開始處理【{color_name}筆蓋】 -> 目標夾取座標: X={pick_x:.3f}, Y={pick_y:.3f}")

    # 1. 確保夾爪先張開 (gripper_ON)
    send.gripper_ON(c)
    time.sleep(2)

    # 2. 移至筆蓋正上方安全高度 (Z_SAFE) -> 垂直下降至夾取高度 (Z_PICK)
    send.Go_Position(c, pick_x, pick_y, Z_SAFE, rx, ry, rz, 70)
    time.sleep(2.5)
    send.Go_Position(c, pick_x, pick_y, Z_PICK, rx, ry, rz, 50)
    time.sleep(2.0)

    # 3. 夾爪閉合夾取 (gripper_OFF)
    send.gripper_OFF(c)
    time.sleep(2)

    # 4. 垂直抬升回安全高度 (防撞機)
    send.Go_Position(c, pick_x, pick_y, Z_SAFE, rx, ry, rz, 50)
    time.sleep(2.0)

    # 5. 移至放置區正上方安全高度 (Z_SAFE) -> 垂直下降至放置高度 (Z_PLACE)
    send.Go_Position(c, place_x, place_y, Z_SAFE, prx, pry, prz, 70)
    time.sleep(2.5)
    send.Go_Position(c, place_x, place_y, Z_PLACE, prx, pry, prz, 50)
    time.sleep(2.0)

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
                
                # 若已完成 ArUco 定位，直接在畫面上即時預覽換算後的機械手臂 (X, Y) 座標！
                if H_mat is not None:
                    rx, ry = pixel_to_robot(cx, cy, H_mat)
                    text = f"{label} X:{rx:.1f} Y:{ry:.1f}"
                else:
                    text = f"{label} ({cx},{cy})"
                    
                cv2.putText(output_img, text, (cx - 50, cy - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, draw_color, 2)
    return coords

def main():
    global LATEST_H_MATRIX

    print("連線至機械手臂...")
    c = ModbusTcpClient(host="192.168.1.1", port=502, unit_id=2)
    c.connect()

    print("手臂初始化：移動至相機初始位置 (Home)...")
    threading.Thread(
        target=send.Go_Position,
        args=(c, HOME_POS[0], HOME_POS[1], HOME_POS[2], HOME_POS[3], HOME_POS[4], HOME_POS[5], 50)
    ).start()

    # 初始化 ArUco 偵測器 (DICT_5X5_100)
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    # 定義紅、藍 HSV 範圍
    lower_red1, upper_red1 = np.array([0, 120, 70]), np.array([10, 255, 255])
    lower_red2, upper_red2 = np.array([170, 120, 70]), np.array([180, 255, 255])
    lower_blue, upper_blue = np.array([85, 130, 50]), np.array([130, 255, 255])
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
        frame = realsense.Get_RGB_Frame()
        if frame is None:
            continue

        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        output = frame_bgr.copy()

        # ---------------- 1. 偵測 4 張 ArUco 並即時更新轉換矩陣 ----------------
        corners, ids, _ = detector.detectMarkers(frame_bgr)
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

        # 當 4 個角落 (ID 1, 2, 3, 4) 同時在畫面中且手臂未移動時，自動更新 H_MATRIX
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
                    threading.Thread(
                        target=execute_task_thread,
                        args=(c, r_targets, b_targets, h_copy, on_task_finished)
                    ).start()
            else:
                print("⏳ 手臂正在移動中，請稍候...")

    cv2.destroyAllWindows()
    c.close()

if __name__ == "__main__":
    main()