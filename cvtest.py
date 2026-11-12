import cv2
import numpy as np

image = cv2.imread("C:\\Users\\ricky\\Desktop\\code\\python\\course\\DRV-Modbus-main\\123.png")

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

lower_hsv1 = np.array([0, 120, 70]) #色相，飽和度，亮度
upper_hsv1 = np.array([10, 255, 255])
lower_hsv2 = np.array([170, 120, 70])
upper_hsv2 = np.array([180, 255, 255])

mask1 = cv2.inRange(hsv, lower_hsv1, upper_hsv1)
mask2 = cv2.inRange(hsv, lower_hsv2, upper_hsv2)
mask = mask1 + mask2 #篩選紅色部分遮罩

contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)#輪廓檢測方法及輪廓逼近模式

output = image.copy()
for contour in contours:
    if cv2.contourArea(contour) > 500:#過濾小輪廓
        cv2.drawContours(output, [contour], -1, (0, 255, 0), 3)

scale = 0.5
image_resized = cv2.resize(image, (0, 0), fx=scale, fy=scale)
mask_resized = cv2.resize(mask, (0, 0), fx=scale, fy=scale)
output_resized = cv2.resize(output, (0, 0), fx=scale, fy=scale)

cv2.imshow("Original Image", image_resized)
cv2.imshow("Mask", mask_resized)
cv2.imshow("Apple Contours", output_resized)

cv2.waitKey(0)
cv2.destroyAllWindows()