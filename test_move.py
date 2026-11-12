from drv_modbus import send
from pymodbus.client import ModbusTcpClient
from realsense import realsense
import numpy as np
import cv2
import time
import threading  # 匯入多執行緒模組

# 建立一個專門用來控制手臂移動的背景函式
def move_robot_thread(client, target_point):
    send.Go_Position(client, target_point[0], target_point[1], target_point[2], target_point[3], target_point[4], target_point[5], 50)

def main():
    # ================= 1. 初始化機械手臂連線 =================
    print("連線至機械手臂...")
    c = ModbusTcpClient(host="192.168.1.1", port=502, unit_id=2)
    c.connect()
    
    # 設定好預備點(Home)與目標點
    home_pos = [353.208, -48.09, 680.0, 179.987, -0.004, -106.121]
    target_pos = [479.442, 385.366, 521.66, 179.987, -0.004, -106.121] 
    
    print("手臂初始化：移動至 Home 點位...")
    # 初始移動也放進背景執行，讓畫面能提早彈出
    threading.Thread(target=move_robot_thread, args=(c, home_pos)).start()

    # ================= 2. 初始化 ArUco 視覺偵測 =================
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    print("開始偵測！畫面上會顯示即時影像。")
    print("請將包含【ID 1】的 ArUco 標籤放入畫面中即可觸發手臂。")
    
    # 控制開關：防止一秒鐘內連續發送幾十次指令給手臂
    has_moved = False 

    while True:
        # 抓取並轉換影像，作為顯示與辨識用途
        frame = realsense.Get_RGB_Frame()
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # 辨識 ArUco
        corners, ids, rejected = detector.detectMarkers(frame_bgr)

        # ================= 3. 觸發手臂動作邏輯 =================
        if ids is not None:
            # 畫出綠色邊框與數字
            cv2.aruco.drawDetectedMarkers(frame_bgr, corners, ids)
            
            # 將陣列攤平變成一維
            flat_ids = ids.flatten()
            
            # 條件：如果畫面中包含 ID 1，「並且」手臂還沒有執行過移動
            if 1 in flat_ids and not has_moved:
                print("✅ 成功偵測到指定標籤 (ID: 1)！發送移動指令...")
                has_moved = True 
                
                # 【關鍵修改】：使用 Thread 背景執行移動指令，這樣畫面就不會卡住
                threading.Thread(target=move_robot_thread, args=(c, target_pos)).start()
                
                print("指令發送完畢。若要再次測試，請在影像視窗按 'r' 鍵重置，或按 'q' 結束。")

        # 為了讓您在畫面上直接看到目前狀態，加上文字提示
        status_text = "Status: Locked (Press 'R' to reset)" if has_moved else "Status: Waiting for ID 1"
        color = (0, 0, 255) if has_moved else (0, 255, 0)
        cv2.putText(frame_bgr, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # 顯示即時影像 (此時畫面會非常流暢)
        cv2.imshow("RealSense + ArUco", frame_bgr)

        # 讀取鍵盤輸入
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("結束程式。")
            break
        elif key == ord('r'):
            if has_moved: # 只有在鎖定狀態下才允許重置
                print("🔄 狀態重置：手臂回到 Home...")
                has_moved = False
                
                # 同樣將回到 Home 的動作丟到背景執行
                threading.Thread(target=move_robot_thread, args=(c, home_pos)).start()

    # 關閉視窗與連線
    cv2.destroyAllWindows()
    c.close()

if __name__ == "__main__":
    main()