import cv2
import numpy as np

# 1. 读取你切好的图标
img = cv2.imread('assets/icons/icon_main.png')
# 2. 转为灰度
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 3. 自动生成黑白 Mask (二值化)
_, mask = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
# 4. 保存
cv2.imwrite('assets/icons/icon_main_mask.png', mask)
print("Mask 生成成功！")
