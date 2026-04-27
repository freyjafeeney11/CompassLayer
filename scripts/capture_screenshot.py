import cv2
import os
import sys
import time

# Ensure we can import from core
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.screen import ScreenCapturer
from config import ROI_HEIGHT_RATIO

def capture_main():
    # 获取文件名
    filename = input("请输入要保存的文件名 (例如: ScreenShot_new1.png): ").strip()
    if not filename:
        filename = f"ScreenShot_{int(time.time())}.png"
    if not filename.endswith(".png"):
        filename += ".png"

    # 初始化捕获器
    # 注意：我们使用当前配置的 ROI 比例
    print(f"初始化屏幕捕获器 (当前 ROI 比例: {ROI_HEIGHT_RATIO})...")
    capturer = ScreenCapturer(roi_height_ratio=ROI_HEIGHT_RATIO, monitor_idx=1)
    
    # 路径准备
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    target_dir = os.path.join(base_dir, "assets", "ScreenShots")
    os.makedirs(target_dir, exist_ok=True)
    target_path = os.path.join(target_dir, filename)

    # 10秒倒计时
    countdown = 10
    print(f"\n准备时间：您有 {countdown} 秒钟切换到游戏/视频画面...")
    for i in range(countdown, 0, -1):
        print(f"将在 {i} 秒后截取图标区域...", end="\r", flush=True)
        time.sleep(1)
    
    print("\n正在截取...")
    frame = capturer.get_frame()
    cv2.imwrite(target_path, frame)
    
    print(f"\n截图已保存至: {target_path}")
    print("您可以使用 test_screenshot.py 直接测试这张图片。")

if __name__ == "__main__":
    capture_main()
