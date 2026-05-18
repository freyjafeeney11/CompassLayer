import cv2
import sys
import os
import numpy as np
from typing import List, Dict, Any
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import COLORS

class Visualizer:

    def __init__(self, window_name: str='Multi-Icon Navigator'):
        self.window_name = window_name

    def draw_detections(self, frame_bgr: np.ndarray, detections: List[Dict[str, Any]], screen_width: int, screen_height: int) -> np.ndarray:
        thickness = max(1, int(screen_width / 1000))
        font_scale = max(0.4, screen_width / 3000.0)
        for det in detections:
            x_rel = det['x_rel']
            y_rel = det['y_rel']
            w_rel = det['w_rel']
            h_rel = det['h_rel']
            cx = int(x_rel * screen_width)
            cy = int(y_rel * screen_height)
            w = int(w_rel * screen_width)
            h = int(h_rel * screen_height)
            x1 = cx - w // 2
            y1 = cy - h // 2
            x2 = cx + w // 2
            y2 = cy + h // 2
            color = COLORS.get(det['label'], (0, 255, 0))
            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, thickness)
            distance_str = det.get('distance', 'N/A')
            rel_offset = det.get('rel_offset', 0.0)
            text = f"{det['label']} | {distance_str} | {rel_offset:+.2f}"
            text_y = max(y1 - 10, int(20 * font_scale))
            cv2.putText(frame_bgr, text, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)
        return frame_bgr

    def show(self, frame_bgr: np.ndarray) -> int:
        cv2.imshow(self.window_name, frame_bgr)
        return cv2.waitKey(1) & 255