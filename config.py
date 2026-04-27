import os
from typing import Dict, Tuple

# 获取当前文件所在的主目录路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 目标图标清单字典
# 目标图标配置：支持路径与可选的验证参数
TARGET_ICONS: Dict[str, Dict] = {
    "main_quest": {
        "path": os.path.join(BASE_DIR, "assets", "icons", "icon_main_centered.png"),
        # "hsv_range": ([15, 40, 40], [120, 255, 255]), 
    },
    # "side_quest": {
    #     "path": os.path.join(BASE_DIR, "assets", "icons", "icon_blue.png"),
    #     "hsv_range": ([100, 50, 50], [130, 255, 255]),
    # },
    # "waypoint": {
    #     "path": os.path.join(BASE_DIR, "assets", "icons", "icon_white.png"),
    #     # 纯白图标不设 hsv_range，后续将通过“实心度”校验
    # }
}

# 预设不同类别的颜色用于可视化 (BGR格式)
COLORS: Dict[str, Tuple[int, int, int]] = {
    "main_quest": (0, 0, 255),    # 红色
    # "side_quest": (255, 0, 0),    # 蓝色
    # "waypoint": (255, 255, 255)   # 白色
}

# 全局配置参数
ROI_HEIGHT_RATIO: float = 0.17              
MATCH_THRESHOLD: float = 0.88               # 暂时关闭颜色校验时，保持较高阈值以抑制噪声
NMS_IOU_THRESHOLD: float = 0.3              # 非极大值抑制的 IoU 阈值
STRAIGHT_AHEAD_THRESHOLD: float = 0.04      # 屏幕宽度 2% 内的偏移视为正前方
COMPASS_WIDTH_RATIO: float = 0.397          # 导航条占据屏幕宽度的比例 (1200/3024)
DESIGN_WIDTH: int = 3024                    # 开发/截图时的基准宽度（用于自动缩放模板）
BLUR_KSIZE: tuple = (5, 5)                  # 高斯模糊核大小 (5, 5)，设为 None 则关闭
