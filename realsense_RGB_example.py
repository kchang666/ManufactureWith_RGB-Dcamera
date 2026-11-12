from realsense import realsense
import cv2

while True:
    frame = realsense.Get_RGB_Frame()
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.imshow("realsense", frame)
    cv2.waitKey(1)
# img = cv2.imread(r"C:\Users\ricky\Desktop\code\python\course\DRV-Modbus-main\123.png")
# cv2.imshow("realsense", img)
# cv2.waitKey(10000)


