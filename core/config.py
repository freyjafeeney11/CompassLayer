import os
import sys
from typing import Dict, Tuple

def resource_path(relative_path: str) -> str:
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath('.')
    return os.path.join(base_path, relative_path)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TARGET_ICONS: Dict[str, Dict] = {'main_quest': {'path': resource_path(os.path.join('assets', 'icons', 'icon_main_centered.png'))}, 'treasure': {'path': resource_path(os.path.join('assets', 'icons', 'icon_treasure.png'))}, 'stockpile': {'path': resource_path(os.path.join('assets', 'icons', 'icon_stockpile.png'))}}
COLORS: Dict[str, Tuple[int, int, int]] = {'main_quest': (0, 0, 255)}
ROI_HEIGHT_RATIO: float = 0.17
MATCH_THRESHOLD: float = 0.90
NMS_IOU_THRESHOLD: float = 0.3
STRAIGHT_AHEAD_THRESHOLD: float = 0.04
COMPASS_WIDTH_RATIO: float = 0.397
DESIGN_WIDTH: int = 3024
BLUR_KSIZE: tuple = (5, 5)