import cv2
import numpy as np
from typing import Dict, List, Any, Callable

class IconDetector:
    """
    检测层：支持带 Alpha 掩码的模板匹配 (Masked Template Matching)
    基于比例的坐标转换与 cv2.dnn.NMSBoxes 过滤。
    """
    def __init__(self, target_icons: Dict[str, Dict], match_threshold: float = 0.8, nms_iou_threshold: float = 0.3, manual_scale: float = None):
        self.match_threshold = match_threshold
        self.nms_iou_threshold = nms_iou_threshold
        
        if manual_scale is not None:
            self.scale = manual_scale
        else:
            # 比例修复：基于 Retina 屏幕的 3024 像素基准计算当前缩放比例
            import mss
            sct = mss.mss()
            current_width = sct.monitors[1]["width"]
            self.scale = current_width / 3024.0
            if self.scale != 1.0:
                print(f"[Detector] 检测到逻辑分辨率差异，正在按比例 {self.scale:.2f} 缩放模板...")
            
        self.templates = self._load_templates(target_icons)

    def _load_templates(self, icon_configs: Dict[str, Dict]) -> Dict[str, Dict[str, Any]]:
        templates = {}
        for label, cfg in icon_configs.items():
            path = cfg["path"]
            # 使用 cv2.IMREAD_UNCHANGED 读取所有通道，包括 Alpha
            template = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if template is not None:
                # 执行缩放
                if abs(self.scale - 1.0) > 0.01:
                    new_size = (int(template.shape[1] * self.scale), int(template.shape[0] * self.scale))
                    if new_size[0] > 0 and new_size[1] > 0:
                        template = cv2.resize(template, new_size, interpolation=cv2.INTER_AREA)

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
                    "w": template.shape[1],
                    "hsv_range": cfg.get("hsv_range") # 获取可选的颜色范围
                }
            else:
                print(f"Warning: 指定的模板图片未找到 [{label}]: {path}")
        return templates

    def _apply_laplacian(self, img: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_8U)

    def detect(self, frame_bgr: np.ndarray, screen_width: int, screen_height: int, normalize_fn: Callable[[float, float], tuple], 
               use_laplacian: bool = False, blur_ksize: tuple = None) -> List[Dict[str, Any]]:
        all_detections = []
        global_id = 0

        # 预处理：高斯模糊 (减少背景细节干扰)
        if blur_ksize:
            search_frame = cv2.GaussianBlur(frame_bgr, blur_ksize, 0)
        else:
            search_frame = frame_bgr

        if use_laplacian:
            search_frame = self._apply_laplacian(search_frame)

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
                    # template_bgr 是我们预设的任务图标 RGB，mask 是图标抠好的 Alpha 通道(黑白图)
                    # mask 让 cv2 只关心有图标的像素
                    res = cv2.matchTemplate(frame_bgr, template_bgr, cv2.TM_CCORR_NORMED, mask=mask)
                    res = np.nan_to_num(res, nan=-1.0)
                    loc = np.where(res >= self.match_threshold)
                else:
                    res = cv2.matchTemplate(frame_bgr, template_bgr, cv2.TM_CCOEFF_NORMED)
                    loc = np.where(res >= self.match_threshold)
            # NMS 非极大值抑制 cv2.dnn.NMSBoxes 会自动过滤掉那些互相重叠太多 (超过 nms_iou_threshold)的框
            # 只保留其中 score (匹配分数) 最大的那个。
            boxes = []
            scores = []
            for pt in zip(*loc[::-1]):
                boxes.append([int(pt[0]), int(pt[1]), int(tw), int(th)])
                scores.append(float(res[pt[1], pt[0]]))
                
            if len(boxes) > 0:
                indices = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=self.match_threshold, nms_threshold=self.nms_iou_threshold)
                
                if len(indices) > 0:
                    for i in indices.flatten():
                        # i 就是经历淘汰赛后，存活下来的、最精准的边框的索引
                        x, y, w, h = boxes[i]

                        # 颜色验证逻辑
                        if tmpl_data["hsv_range"]:
                            lower, upper = tmpl_data["hsv_range"]
                            # 提取检测到的 ROI 区域 (在原始 BGR 帧上)
                            roi = frame_bgr[y:y+h, x:x+w]
                            if roi.size == 0: continue
                            
                            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                            color_mask = cv2.inRange(hsv_roi, np.array(lower), np.array(upper))
                            
                            # 计算颜色匹配的像素比例
                            match_pixel_ratio = np.count_nonzero(color_mask) / color_mask.size
                            
                            # 如果该区域内目标颜色的占比太低（如低于 5%），则认为是形状误报（如白色代码）
                            if match_pixel_ratio < 0.05:
                                continue
                        
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
                            "h_rel": rel_h,
                            "score": scores[i]
                        }
                        all_detections.append(detection)
                        global_id += 1
                    
        return all_detections
