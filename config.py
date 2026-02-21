import os
from typing import Dict, Tuple

# 获取当前文件所在的主目录路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 目标图标清单字典
TARGET_ICONS: Dict[str, str] = {
    "main_quest": os.path.join(BASE_DIR, "assets", "icons", "icon_main_centered.png"),
    "side_quest": os.path.join(BASE_DIR, "assets", "icons", "icon_blue.png"),
    "waypoint": os.path.join(BASE_DIR, "assets", "icons", "icon_white.png")
}

# 预设不同类别的颜色用于可视化 (BGR格式)
COLORS: Dict[str, Tuple[int, int, int]] = {
    "main_quest": (0, 0, 255),    # 红色
    "side_quest": (255, 0, 0),    # 蓝色
    "waypoint": (255, 255, 255)   # 白色
}

# 全局配置参数
ROI_HEIGHT_RATIO: float = 0.25              # 屏幕顶部 25% 区域用于捕获和检测
MATCH_THRESHOLD: float = 0.8                # 模板匹配阈值（针对带背景图标）
NMS_IOU_THRESHOLD: float = 0.3              # 非极大值抑制的 IoU 阈值
STRAIGHT_AHEAD_THRESHOLD: float = 0.02      # 屏幕宽度 2% 内的偏移视为正前方
