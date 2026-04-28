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
from core.audiofeedback import NavigationController, from_algo_batch
from utils.visualizer import Visualizer


def process_video(
    input_video_path: str,
    output_video_path: str = None,
    preview: bool = False,
    audio_feedback: bool = False,
    realtime_audio: bool = False,
    export_audio: bool = False,
) -> None:
    """
    离线处理游戏录像视频，在画面上绘制识别框并导出新的视频。
    可选开启音频反馈（空间声+提示音）。
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
    
    if export_audio:
        temp_video_path = output_video_path.replace(".mp4", "_temp_visual.mp4")
        out = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))
    else:
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
    nav = NavigationController() if audio_feedback else None
    
    offline_nav = None
    if export_audio:
        from core.offline_audio import OfflineNavigationController
        total_duration_sec = total_frames / fps if fps > 0 else 0
        offline_nav = OfflineNavigationController(total_duration_sec)

    def normalize_coord(px_x: float, px_y: float):
        return px_x / width, px_y / height

    capture_height = int(height * ROI_HEIGHT_RATIO)
    frame_count = 0
    start_time = time.time()
    audio_start_time = time.time()

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

        # 4.1 音频反馈（可选）
        if nav is not None and output_list:
            nav_icons = from_algo_batch(output_list)
            nav.update(nav_icons)

            # 让离线处理按视频帧率节奏播放声音，避免过快连发
            if realtime_audio and fps > 0:
                target_elapsed = frame_count / fps
                actual_elapsed = time.time() - audio_start_time
                delay = target_elapsed - actual_elapsed
                if delay > 0:
                    time.sleep(delay)

        if offline_nav is not None and output_list:
            nav_icons = from_algo_batch(output_list)
            current_time = frame_count / fps if fps > 0 else 0
            offline_nav.update(nav_icons, current_time)

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
    if nav is not None:
        nav.stop()

    if export_audio:
        import subprocess
        temp_audio_path = output_video_path.replace(".mp4", "_temp_audio.wav")
        print("\n正在生成离线音频...")
        offline_nav.export(temp_audio_path)
        
        print(f"正在合成视频与音频，输出到: {output_video_path}")
        try:
            import imageio_ffmpeg
            ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
            cmd = [
                ffmpeg_path,
                "-y",
                "-i", temp_video_path,
                "-i", temp_audio_path,
                "-c:v", "copy",
                "-c:a", "aac",
                output_video_path
            ]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            try:
                os.remove(temp_audio_path)
                os.remove(temp_video_path)
            except Exception:
                pass
        except Exception as e:
            print(f"合成视频时出错: {e}")

    elapsed = time.time() - start_time
    print(f"\n\n处理完成! 导出的视频已保存至: {output_video_path}")
    if elapsed > 0:
        print(f"总耗时: {elapsed:.2f}秒 (平均处理速度: {frame_count / elapsed:.2f} fps)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="处理预先录制的游戏视频，叠加识别框UI并导出")
    parser.add_argument("input", help="输入的视频文件路径")
    parser.add_argument("-o", "--output", help="导出视频路径（可选）", default=None)
    parser.add_argument("--preview", action="store_true", help="显示实时预览窗口")
    parser.add_argument("--audio", action="store_true", help="启用音频反馈")
    parser.add_argument("--realtime-audio", action="store_true", help="按视频原始帧率节奏播放音频")
    parser.add_argument("--export-audio", action="store_true", help="将生成的空间音频写入输出的视频文件中")
    args = parser.parse_args()

    process_video(
        args.input,
        args.output,
        preview=args.preview,
        audio_feedback=args.audio,
        realtime_audio=args.realtime_audio,
        export_audio=args.export_audio,
    )
