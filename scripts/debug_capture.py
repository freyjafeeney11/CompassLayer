import cv2
import os
import sys
import numpy as np

# Ensure we can import from core
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.screen import ScreenCapturer
from config import ROI_HEIGHT_RATIO

def debug_capture():
    print("Initializing ScreenCapturer...")
    capturer = ScreenCapturer(roi_height_ratio=ROI_HEIGHT_RATIO, monitor_idx=1)
    
    # Create debug directory
    debug_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "debug")
    os.makedirs(debug_dir, exist_ok=True)
    
    # 添加倒计时，给用户切换窗口的时间
    import time
    countdown = 10
    print(f"\nPreparation: You have {countdown} seconds to switch to your video player and make it full screen/top-aligned...")
    for i in range(countdown, 0, -1):
        print(f"Capturing in {i}...", end="\r", flush=True)
        time.sleep(1)
    print("\nCapturing NOW!")
    
    print(f"Capturing ROI (Ratio: {ROI_HEIGHT_RATIO})...")
    frame = capturer.get_frame()
    
    output_path = os.path.join(debug_dir, "raw_capture_debug.png")
    cv2.imwrite(output_path, frame)
    
    info = capturer.get_screen_info()
    print(f"\nCapture saved to: {output_path}")
    print(f"Screen Info: {info['width']}x{info['height']}")
    print(f"Capture Height: {info['capture_height']} pixels")
    print("\nPLEASE CHECK THIS IMAGE TO SEE IF THE GAME'S NAVIGATION BAR IS ACTUALLY IN THE PICTURE.")

if __name__ == "__main__":
    debug_capture()
