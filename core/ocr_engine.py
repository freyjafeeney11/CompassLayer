import cv2
import pytesseract
import numpy as np

class OCREngine:
    """
    识别层：使用 tesseract 进行区域距离数值的提取 (Dynamic Distance Parsing)
    完全依赖传入的相对比例坐标进行反向计算
    """
    def __init__(self):
        # 限制白名单为数字及字母 'm'
        self.config: str = '--psm 7 -c tessedit_char_whitelist=0123456789m'

    def extract_distance(self, frame_bgr: np.ndarray, x_rel: float, y_rel: float, w_rel: float, h_rel: float, screen_width: int, screen_height: int) -> str:
        """
        根据图标在屏幕上的比例位置，反向计算出其上方的一小块像素区域并提取数字。
        """
        # 反向计算中心点像素坐标
        cx = int(x_rel * screen_width)
        cy = int(y_rel * screen_height)
        
        # 反向计算宽高像素
        icon_w = int(w_rel * screen_width)
        icon_h = int(h_rel * screen_height)
        
        icon_top_y = cy - icon_h // 2

        # 动态 ROI 区域：设在图标正上方，高度根据屏幕比例计算（例如屏幕高度的 3%）
        roi_h = int(0.03 * screen_height)
        roi_w = max(icon_w * 2, int(0.05 * screen_width))  # 至少保证一定的宽度
        
        ocr_y2 = max(0, icon_top_y)
        ocr_y1 = max(0, ocr_y2 - roi_h)
        
        ocr_x1 = max(0, cx - roi_w // 2)
        ocr_x2 = min(screen_width, cx + roi_w // 2)
        
        # 越界检查
        if ocr_y2 <= ocr_y1 or ocr_x2 <= ocr_x1:
            return "N/A"
            
        ocr_roi = frame_bgr[ocr_y1:ocr_y2, ocr_x1:ocr_x2]
        
        # 转为灰度，进行二值化反转增强文字对比度
        ocr_gray = cv2.cvtColor(ocr_roi, cv2.COLOR_BGR2GRAY)
        _, ocr_thresh = cv2.threshold(ocr_gray, 150, 255, cv2.THRESH_BINARY_INV)
        
        text = pytesseract.image_to_string(ocr_thresh, config=self.config).strip()
        
        return text if text else "N/A"
