import cv2
import numpy as np
from typing import Dict, List, Any, Callable

class IconDetector:
    """
    检测层：支持带 Alpha 掩码的模板匹配 (Masked Template Matching)
    基于比例的坐标转换与 cv2.dnn.NMSBoxes 过滤。
    """
    def __init__(self, target_icons: Dict[str, str], match_threshold: float = 0.8, nms_iou_threshold: float = 0.3):
        self.match_threshold = match_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.templates = self._load_templates(target_icons)

    def _load_templates(self, template_paths: Dict[str, str]) -> Dict[str, Dict[str, Any]]:
        templates = {}
        for label, path in template_paths.items():
            # 使用 cv2.IMREAD_UNCHANGED 读取所有通道，包括 Alpha
            template = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if template is not None:
                has_alpha = template.shape[2] == 4 if len(template.shape) == 3 else False
                
                if has_alpha:
                    # 分离 BGR 和 Alpha 掩码
                    bgr = template[:, :, :3]
                    # cv2.matchTemplate 的 mask 需要保持相同通道数，或明确单通道
                    # 虽然标准说法是 1 还是 3 都行，但为了安全我们只提取 1 个通道。
                    alpha = template[:, :, 3]
                else:
                    bgr = template
                    alpha = None
                    
                templates[label] = {
                    "image": bgr,
                    "mask": alpha,
                    "h": template.shape[0],
                    "w": template.shape[1]
                }
            else:
                print(f"Warning: 指定的模板图片未找到 [{label}]: {path}")
        return templates

    def _apply_laplacian(self, img: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_8U)

    def detect(self, frame_bgr: np.ndarray, screen_width: int, screen_height: int, normalize_fn: Callable[[float, float], tuple], use_laplacian: bool = False) -> List[Dict[str, Any]]:
        all_detections = []
        global_id = 0

        search_frame = self._apply_laplacian(frame_bgr) if use_laplacian else frame_bgr

        for label, tmpl_data in self.templates.items():
            template_bgr = tmpl_data["image"]
            mask = tmpl_data["mask"]
            th, tw = tmpl_data["h"], tmpl_data["w"]
            
            if use_laplacian and mask is None:
                search_template = self._apply_laplacian(template_bgr)
                res = cv2.matchTemplate(search_frame, search_template, cv2.TM_CCOEFF_NORMED)
                # TM_CCOEFF_NORMED：最大值最匹配
                loc = np.where(res >= self.match_threshold)
            else:
                if mask is not None:
                    # OpenCV TM_CCORR_NORMED combined with masks often causes division by zero (NaNs)
                    # at regions where the image under the mask evaluates to 0. 
                    # We use TM_CCORR_NORMED which gives 1.0 for perfect match, and clean the NaNs.
                    res = cv2.matchTemplate(frame_bgr, template_bgr, cv2.TM_CCORR_NORMED, mask=mask)
                    res = np.nan_to_num(res, nan=-1.0)
                    loc = np.where(res >= self.match_threshold)
                else:
                    res = cv2.matchTemplate(frame_bgr, template_bgr, cv2.TM_CCOEFF_NORMED)
                    loc = np.where(res >= self.match_threshold)
            
            boxes = []
            scores = []
            for pt in zip(*loc[::-1]):
                boxes.append([int(pt[0]), int(pt[1]), int(tw), int(th)])
                scores.append(float(res[pt[1], pt[0]]))
                
            if len(boxes) > 0:
                indices = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=self.match_threshold, nms_threshold=self.nms_iou_threshold)
                
                if len(indices) > 0:
                    for i in indices.flatten():
                        x, y, w, h = boxes[i]
                        
                        # 这个中心坐标在 Mask 处理后是真正的原图标视觉中心
                        cx = x + w / 2.0
                        cy = y + h / 2.0
                        
                        rel_x, rel_y = normalize_fn(cx, cy)
                        rel_w = w / screen_width
                        rel_h = h / screen_height
                            
                        detection = {
                            "id": global_id,
                            "label": label,
                            "x_rel": rel_x,
                            "y_rel": rel_y,
                            "w_rel": rel_w,
                            "h_rel": rel_h
                        }
                        all_detections.append(detection)
                        global_id += 1
                    
        return all_detections
