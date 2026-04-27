import cv2
import time
import argparse
import sys
import os

# 将项目根目录加入环境，以便导入自定义模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    TARGET_ICONS,
    MATCH_THRESHOLD,
    NMS_IOU_THRESHOLD,
    STRAIGHT_AHEAD_THRESHOLD,
    COMPASS_WIDTH_RATIO,
    BLUR_KSIZE,
    ROI_HEIGHT_RATIO,
)
from core.detector import IconDetector
from core.ocr_engine import OCREngine
from utils.visualizer import Visualizer


def process_video(
    input_video_path: str,
    output_video_path: str = None,
    preview: bool = False,
) -> None:
    """
    离线处理游戏录像视频，在画面上绘制识别框并导出新的视频。
    """
    if not os.path.exists(input_video_path):
        print(f"错误: 找不到输入视频文件 {input_video_path}")
        return

    # 1. 打开视频并获取属性
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"错误: 无法打开视频 {input_video_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if output_video_path is None:
        filename, _ = os.path.splitext(input_video_path)
        output_video_path = f"{filename}_processed.mp4"

    # 2. 设置视频导出器
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    print(f"开始处理视频: {input_video_path}")
    print(f"分辨率: {width}x{height} | 帧率: {fps:.2f} | 总帧数: {total_frames}")

    # 3. 初始化核心组件
    manual_scale = width / 3024.0
    detector = IconDetector(
        target_icons=TARGET_ICONS,
        match_threshold=MATCH_THRESHOLD,
        nms_iou_threshold=NMS_IOU_THRESHOLD,
        manual_scale=manual_scale,
    )
    ocr_engine = OCREngine()
    visualizer = Visualizer("Video Processing Preview")

    def normalize_coord(px_x: float, px_y: float):
        return px_x / width, px_y / height

    capture_height = int(height * ROI_HEIGHT_RATIO)
    frame_count = 0
    start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        roi_frame = frame[0:capture_height, 0:width]

        detections = detector.detect(
            roi_frame,
            screen_width=width,
            screen_height=height,
            normalize_fn=normalize_coord,
            use_laplacian=False,
            blur_ksize=BLUR_KSIZE,
        )

        output_list = []

        # 4. 数据处理与 OCR
        for det in detections:
            icon_x_rel = det["x_rel"]
            icon_y_rel = det["y_rel"]
            icon_w_rel = det["w_rel"]
            icon_h_rel = det["h_rel"]

            relative_offset = (icon_x_rel - 0.5) / COMPASS_WIDTH_RATIO
            det["rel_offset"] = relative_offset

            if abs(relative_offset) < STRAIGHT_AHEAD_THRESHOLD:
                direction = "Straight"
            elif relative_offset < 0:
                direction = "Left"
            else:
                direction = "Right"
            det["direction"] = direction

            if abs(relative_offset) < 0.1:
                dist_text = ocr_engine.extract_distance(
                    frame, icon_x_rel, icon_y_rel, icon_w_rel, icon_h_rel, width, height
                )
            else:
                dist_text = "N/A"
            det["distance"] = dist_text

            output_list.append(
                {
                    "id": det["id"],
                    "label": det["label"],
                    "rel_offset": round(relative_offset, 3),
                    "direction": direction,
                    "distance": dist_text,
                }
            )

        # 5. 可视化绘制并写入
        frame_visualized = visualizer.draw_detections(
            frame, detections, screen_width=width, screen_height=height
        )
        out.write(frame_visualized)

        # 可选预览（无图形环境下应关闭）
        if preview:
            cv2.imshow("Video Processing Preview (Press 'q' to stop preview)", frame_visualized)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                cv2.destroyAllWindows()

        if frame_count % 10 == 0:
            sys.stdout.write(f"\r正在处理: {frame_count}/{total_frames} 帧 ({(frame_count / total_frames) * 100:.1f}%)")
            sys.stdout.flush()

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    elapsed = time.time() - start_time
    print(f"\n\n处理完成! 导出的视频已保存至: {output_video_path}")
    if elapsed > 0:
        print(f"总耗时: {elapsed:.2f}秒 (平均处理速度: {frame_count / elapsed:.2f} fps)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="处理预先录制的游戏视频，叠加识别框UI并导出")
    parser.add_argument("input", help="输入的视频文件路径")
    parser.add_argument("-o", "--output", help="导出视频路径（可选）", default=None)
    parser.add_argument("--preview", action="store_true", help="显示实时预览窗口")
    args = parser.parse_args()

    process_video(
        args.input,
        args.output,
        preview=args.preview,
    )
