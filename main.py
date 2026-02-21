import cv2
from typing import List, Dict, Any

from config import TARGET_ICONS, MATCH_THRESHOLD, NMS_IOU_THRESHOLD, STRAIGHT_AHEAD_THRESHOLD
from core.screen import ScreenCapturer
from core.detector import IconDetector
from core.ocr_engine import OCREngine
from utils.visualizer import Visualizer

def play_audio_feedback(rel_offset: float) -> None:
    """
    预留音频接口：根据 relative_offset (-0.5 到 0.5) 播放声音
    这可以作为声道平衡参数 (Pan) 输入给音频库，-0.5 代表声音完全在左边，0.5 在右边。
    """
    pass

def main() -> None:
    print("正在初始化纯比例运算系统组件...")
    screen_capturer = ScreenCapturer(monitor_idx=1)
    screen_info = screen_capturer.get_screen_info()
    
    detector = IconDetector(
        target_icons=TARGET_ICONS,
        match_threshold=MATCH_THRESHOLD,
        nms_iou_threshold=NMS_IOU_THRESHOLD
    )
    
    ocr_engine = OCREngine()
    visualizer = Visualizer()
    
    print(f"初始化完毕。屏幕总分辨率: {screen_info['width']}x{screen_info['height']}")
    print("（按 'q' 键退出程序）")

    try:
        while True:
            # 1. 获取屏幕局部画面
            frame_bgr = screen_capturer.get_frame()
            
            # 2. 目标检测 (比例空间)
            # 采用 Laplacian 滤镜进行干扰过滤，如果影响性能可以关闭。
            detections = detector.detect(
                frame_bgr, 
                screen_width=screen_info["width"], 
                screen_height=screen_info["height"],
                normalize_fn=screen_capturer.normalize_coord,
                use_laplacian=True
            )
            
            output_list: List[Dict[str, Any]] = []

            # 3. 数据处理与 OCR
            for det in detections:
                icon_x_rel = det["x_rel"]
                icon_y_rel = det["y_rel"]
                icon_w_rel = det["w_rel"]
                icon_h_rel = det["h_rel"]

                # 数学优势：正中心的偏移为 0.0，从 -0.5 (纯左) 到 +0.5 (纯右)
                relative_offset = icon_x_rel - 0.5
                det["rel_offset"] = relative_offset
                
                # 播放音频接口
                play_audio_feedback(relative_offset)
                
                # 判断方向
                if abs(relative_offset) < STRAIGHT_AHEAD_THRESHOLD:
                    direction = "Straight"
                elif relative_offset < 0:
                    direction = "Left"
                else:
                    direction = "Right"
                    
                det["direction"] = direction
                
                # 性能优化：只有当图标的 relative_offset 绝对值小于 0.1（即图标接近正前方中心）时，才触发 OCR 识别
                if abs(relative_offset) < 0.1:
                    dist_text = ocr_engine.extract_distance(
                        frame_bgr, icon_x_rel, icon_y_rel, icon_w_rel, icon_h_rel,
                        screen_info["width"], screen_info["height"]
                    )
                else:
                    dist_text = "N/A"
                    
                det["distance"] = dist_text
                
                # 添加到输出列表
                output_list.append({
                    "id": det["id"],
                    "label": det["label"],
                    "rel_offset": round(relative_offset, 3), # 保留三位小数，使得打印美观
                    "direction": direction,
                    "distance": dist_text
                })
                
            # 4. 打印清理后的数据
            if output_list:
                print(output_list)
                
            # 5. 可视化绘制 (Adaptive UI 根据屏幕分辨率动态缩放)
            frame_visualized = visualizer.draw_detections(
                frame_bgr, 
                detections, 
                screen_width=screen_info['width'], 
                screen_height=screen_info['height']
            )
            
            key = visualizer.show(frame_visualized)
            
            if key == ord('q'):
                print("检测到退出指令，程序即将终止。")
                break
                
    except Exception as e:
        print(f"发生异常: {e}")
    finally:
        cv2.destroyAllWindows()
        print("所有资源已释放。")

if __name__ == "__main__":
    main()
