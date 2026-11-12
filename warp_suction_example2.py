from realsense import realsense
import numpy as np
import cv2

def main():
    # 使用 OpenCV 4.7.0 以上的新版 ArUco 寫法
    # 設定字典為 DICT_5X5_100 (與您原本的設定相同)
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    print("開始辨識 ArUco，請將攝影機對準標籤。按 'q' 鍵可結束程式。")

    while True:
        # 從 RealSense 取得 RGB 影像
        frame = realsense.Get_RGB_Frame()
        
        # 將 RGB 轉換為 OpenCV 使用的 BGR 格式
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # 進行 ArUco 標籤偵測
        # corners: 標籤四個角的像素座標, ids: 標籤的 ID 號碼
        corners, ids, rejected = detector.detectMarkers(frame_bgr)

        # 如果畫面中有偵測到任何 ArUco 標籤
        if ids is not None:
            # 直接在畫面上畫出綠色邊框與 ID 數字
            cv2.aruco.drawDetectedMarkers(frame_bgr, corners, ids)

            # (選用) 在終端機印出偵測到的 ID，方便您確認
            # print(f"偵測到標籤 ID: {ids.flatten()}")

        # 顯示即時影像視窗
        cv2.imshow("ArUco Detection", frame_bgr)

        # 偵測鍵盤輸入，按 'q' 鍵離開迴圈
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 關閉所有 OpenCV 視窗
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()