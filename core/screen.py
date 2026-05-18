import mss
import numpy as np
import cv2
from typing import Tuple, Dict

class ScreenCapturer:

    def __init__(self, roi_height_ratio: float=0.15, monitor_idx: int=1):
        self.sct = mss.mss()
        if monitor_idx < len(self.sct.monitors):
            self.monitor = self.sct.monitors[monitor_idx]
        else:
            self.monitor = self.sct.monitors[0]
        self.screen_width: int = self.monitor['width']
        self.screen_height: int = self.monitor['height']
        self.capture_height: int = int(self.screen_height * roi_height_ratio)
        self.roi_monitor: Dict[str, int] = {'top': self.monitor['top'], 'left': self.monitor['left'], 'width': self.screen_width, 'height': self.capture_height}

    def get_frame(self) -> np.ndarray:
        sct_img = self.sct.grab(self.roi_monitor)
        frame_bgra = np.array(sct_img)
        frame_bgr = cv2.cvtColor(frame_bgra, cv2.COLOR_BGRA2BGR)
        return frame_bgr

    def get_screen_info(self) -> Dict[str, int]:
        return {'width': self.screen_width, 'height': self.screen_height, 'capture_height': self.capture_height}

    def normalize_coord(self, px_x: float, px_y: float) -> Tuple[float, float]:
        rel_x = px_x / self.screen_width
        rel_y = px_y / self.screen_height
        return (rel_x, rel_y)