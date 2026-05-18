import cv2
import numpy as np
from typing import Dict, List, Any, Callable

class IconDetector:

    def __init__(self, target_icons: Dict[str, Dict], match_threshold: float=0.8, nms_iou_threshold: float=0.3, manual_scale: float=None, use_multi_scale: bool=True):
        self.match_threshold = match_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.use_multi_scale = use_multi_scale
        if manual_scale is not None:
            self.scale = manual_scale
        else:
            import mss
            sct = mss.mss()
            current_width = sct.monitors[1]['width']
            self.scale = current_width / 3024.0
            if self.scale != 1.0:
                print(f'[Detector] Resolution mismatch detected, scaling templates by {self.scale:.2f}...')
        if self.use_multi_scale:
            print('[Detector] Multi-Scale Search enabled.')
        self.templates = self._load_templates(target_icons)

    def _load_templates(self, icon_configs: Dict[str, Dict]) -> Dict[str, List[Dict[str, Any]]]:
        templates = {}
        for label, cfg in icon_configs.items():
            path = cfg['path']
            template = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if template is not None:
                templates[label] = []
                if self.use_multi_scale:
                    scale_factors = [self.scale * f for f in [0.8, 0.9, 1.0, 1.1, 1.25]]
                else:
                    scale_factors = [self.scale]
                for s in scale_factors:
                    if abs(s - 1.0) > 0.01:
                        new_size = (int(template.shape[1] * s), int(template.shape[0] * s))
                        if new_size[0] <= 0 or new_size[1] <= 0:
                            continue
                        scaled_temp = cv2.resize(template, new_size, interpolation=cv2.INTER_AREA)
                    else:
                        scaled_temp = template.copy()
                    has_alpha = scaled_temp.shape[2] == 4 if len(scaled_temp.shape) == 3 else False
                    if has_alpha:
                        bgr = scaled_temp[:, :, :3]
                        alpha = scaled_temp[:, :, 3]
                    else:
                        bgr = scaled_temp
                        alpha = None
                    templates[label].append({'image': bgr, 'mask': alpha, 'h': scaled_temp.shape[0], 'w': scaled_temp.shape[1], 'scale': s, 'hsv_range': cfg.get('hsv_range')})
            else:
                print(f'Warning: Template image not found [{label}]: {path}')
        return templates

    def _apply_laplacian(self, img: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_8U)

    def detect(self, frame_bgr: np.ndarray, screen_width: int, screen_height: int, normalize_fn: Callable[[float, float], tuple], use_laplacian: bool=False, blur_ksize: tuple=None) -> List[Dict[str, Any]]:
        all_detections = []
        global_id = 0
        if blur_ksize:
            search_frame = cv2.GaussianBlur(frame_bgr, blur_ksize, 0)
        else:
            search_frame = frame_bgr
        if use_laplacian:
            search_frame = self._apply_laplacian(search_frame)
        for label, tmpl_list in self.templates.items():
            boxes = []
            scores = []
            template_meta = []
            for tmpl_data in tmpl_list:
                template_bgr = tmpl_data['image']
                mask = tmpl_data['mask']
                th, tw = (tmpl_data['h'], tmpl_data['w'])
                if use_laplacian and mask is None:
                    search_template = self._apply_laplacian(template_bgr)
                    res = cv2.matchTemplate(search_frame, search_template, cv2.TM_CCOEFF_NORMED)
                    loc = np.where(res >= self.match_threshold)
                elif mask is not None:
                    res = cv2.matchTemplate(frame_bgr, template_bgr, cv2.TM_CCORR_NORMED, mask=mask)
                    res = np.nan_to_num(res, nan=-1.0)
                    loc = np.where(res >= self.match_threshold)
                else:
                    res = cv2.matchTemplate(frame_bgr, template_bgr, cv2.TM_CCOEFF_NORMED)
                    loc = np.where(res >= self.match_threshold)
                for pt in zip(*loc[::-1]):
                    boxes.append([int(pt[0]), int(pt[1]), int(tw), int(th)])
                    scores.append(float(res[pt[1], pt[0]]))
                    template_meta.append(tmpl_data)
            if len(boxes) > 0:
                indices = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=self.match_threshold, nms_threshold=self.nms_iou_threshold)
                if len(indices) > 0:
                    for i_idx in indices.flatten():
                        x, y, w, h = boxes[i_idx]
                        tmpl_data = template_meta[i_idx]
                        if tmpl_data['hsv_range']:
                            lower, upper = tmpl_data['hsv_range']
                            roi = frame_bgr[y:y + h, x:x + w]
                            if roi.size == 0:
                                continue
                            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                            color_mask = cv2.inRange(hsv_roi, np.array(lower), np.array(upper))
                            match_pixel_ratio = np.count_nonzero(color_mask) / color_mask.size
                            if match_pixel_ratio < 0.05:
                                continue
                        cx = x + w / 2.0
                        cy = y + h / 2.0
                        rel_x, rel_y = normalize_fn(cx, cy)
                        rel_w = w / screen_width
                        rel_h = h / screen_height
                        detection = {'id': global_id, 'label': label, 'x_rel': rel_x, 'y_rel': rel_y, 'w_rel': rel_w, 'h_rel': rel_h, 'score': scores[i_idx], 'matched_scale': tmpl_data['scale']}
                        all_detections.append(detection)
                        global_id += 1
        return all_detections