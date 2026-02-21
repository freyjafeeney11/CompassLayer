import cv2
import os
import sys

# Ensure modules can be imported
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import TARGET_ICONS, NMS_IOU_THRESHOLD, STRAIGHT_AHEAD_THRESHOLD, ROI_HEIGHT_RATIO
from core.detector import IconDetector
from core.ocr_engine import OCREngine
from utils.visualizer import Visualizer

def normalize_coord(px_x, px_y, sw, sh):
    return px_x / sw, px_y / sh

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    screenshot_dir = os.path.join(base_dir, "assets", "ScreenShots")
    output_dir = os.path.join(base_dir, "assets", "results")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load detector with mask support
    detector = IconDetector(
        target_icons=TARGET_ICONS,
        match_threshold=0.85, # Strict threshold for masked matching
        nms_iou_threshold=NMS_IOU_THRESHOLD
    )
    ocr_engine = OCREngine()
    visualizer = Visualizer()
    
    screenshots = [f for f in os.listdir(screenshot_dir) if f.endswith(".png")]
    
    for filename in screenshots:
        print(f"Processing {filename}...")
        img_path = os.path.join(screenshot_dir, filename)
        frame_bgr = cv2.imread(img_path)
        if frame_bgr is None: continue
        
        sh, sw = frame_bgr.shape[:2]
        cap_h = int(sh * ROI_HEIGHT_RATIO)
        roi_bgr = frame_bgr[0:cap_h, 0:sw]
        
        detections = detector.detect(
            roi_bgr, 
            screen_width=sw, 
            screen_height=sh, 
            normalize_fn=lambda x,y: normalize_coord(x,y,sw,sh)
        )
        
        for det in detections:
            rel_offset = det["x_rel"] - 0.5
            det["rel_offset"] = rel_offset
            
            # OCR distance if centered
            if abs(rel_offset) < 0.1:
                det["distance"] = ocr_engine.extract_distance(
                    frame_bgr, det["x_rel"], det["y_rel"], det["w_rel"], det["h_rel"], sw, sh
                )
            else:
                det["distance"] = "N/A"
        
        # Visualize
        frame_vis = visualizer.draw_detections(frame_bgr, detections, sw, sh)
        
        output_path = os.path.join(output_dir, f"result_{filename}")
        cv2.imwrite(output_path, frame_vis)
        print(f"Saved to {output_path}")

if __name__ == "__main__":
    main()
