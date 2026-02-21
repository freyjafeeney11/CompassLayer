import cv2
import sys
import os
import numpy as np
from typing import List, Dict, Any

# 确保能导入 config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import COLORS

class Visualizer:
    """
    可视化工具：自适应分辨率绘图 (Adaptive UI)
    """
    def __init__(self, window_name: str = "Multi-Icon Navigator"):
        self.window_name = window_name

    def draw_detections(self, frame_bgr: np.ndarray, detections: List[Dict[str, Any]], screen_width: int, screen_height: int) -> np.ndarray:
        """
        绘制所有识别出的检测框及其附带文本，字体和线条粗细根据当前分辨率动态调整
        """
        # 自适应字体大小和线条粗细
        thickness = max(1, int(screen_width / 1000))
        font_scale = max(0.4, screen_width / 3000.0)

        for det in detections:
            # 将相对比例转换回像素坐标进行绘制
            x_rel = det["x_rel"]
            y_rel = det["y_rel"]
            w_rel = det["w_rel"]
            h_rel = det["h_rel"]
            
            cx = int(x_rel * screen_width)
            cy = int(y_rel * screen_height)
            w = int(w_rel * screen_width)
            h = int(h_rel * screen_height)
            
            x1 = cx - w // 2
            y1 = cy - h // 2
            x2 = cx + w // 2
            y2 = cy + h // 2
            
            color = COLORS.get(det["label"], (0, 255, 0)) # 若无指定则默认绿色
            
            # 画实体框
            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, thickness)
            
            # 格式化文本
            distance_str = det.get('distance', 'N/A')
            rel_offset = det.get('rel_offset', 0.0)
            text = f"{det['label']} | {distance_str} | {rel_offset:+.2f}"
            
            text_y = max(y1 - 10, int(20 * font_scale))  # 防止文字出界
            
            cv2.putText(frame_bgr, text, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)
            
        return frame_bgr

    def show(self, frame_bgr: np.ndarray) -> int:
        """
        调出 OpenCV 窗口展示
        返回键盘按键，用于捕捉是否退出
        """
        cv2.imshow(self.window_name, frame_bgr)
        return cv2.waitKey(1) & 0xFF
