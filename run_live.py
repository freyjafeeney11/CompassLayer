import argparse
import time
import sys
import msvcrt
import cv2
from typing import List, Dict, Any
from config import TARGET_ICONS, MATCH_THRESHOLD, NMS_IOU_THRESHOLD, STRAIGHT_AHEAD_THRESHOLD, COMPASS_WIDTH_RATIO, ROI_HEIGHT_RATIO, BLUR_KSIZE
import keyboard
COMPASS_X_START = 0.5 - COMPASS_WIDTH_RATIO / 2
COMPASS_X_END = 0.5 + COMPASS_WIDTH_RATIO / 2
from core.screen import ScreenCapturer
from core.detector import IconDetector
from core.ocr_engine import OCREngine
from utils.visualizer import Visualizer
from core.audiofeedback import NavigationController, from_algo_batch

class C:
    RESET = '\x1b[0m'
    BOLD = '\x1b[1m'
    CYAN = '\x1b[96m'
    GREEN = '\x1b[92m'
    YELLOW = '\x1b[93m'
    RED = '\x1b[91m'
    DIM = '\x1b[2m'

def _ansi(text: str, *codes: str) -> str:
    return ''.join(codes) + text + C.RESET
BANNER = f'\n{C.CYAN}{C.BOLD}╔══════════════════════════════════════════╗\n║      CompassLayer  •  Live Test Mode     ║\n╚══════════════════════════════════════════╝{C.RESET}\n'

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='CompassLayer — real-time screen-capture test runner')
    parser.add_argument('--monitor', type=int, default=1, metavar='N', help='Monitor index to capture (default: 1). Use 0 for primary.')
    parser.add_argument('--threshold', type=float, default=None, metavar='T', help=f'Override icon match threshold (default: {MATCH_THRESHOLD})')
    parser.add_argument('--verbose', '-v', action='store_true', help='Print detections every frame instead of only when icons are found.')
    parser.add_argument('--no-audio', action='store_true', help='Disable audio feedback (visual/console debug only).')
    return parser.parse_args()

def format_detection(det: Dict[str, Any]) -> str:
    offset = det['rel_offset']
    direction = det['direction']
    label = det['label']
    dist = det.get('distance', 'N/A')
    dir_color = C.GREEN if direction == 'Straight' else C.YELLOW
    bar_len = 40
    bar_pos = int((offset + 0.5) * bar_len)
    bar_pos = max(0, min(bar_len - 1, bar_pos))
    bar = [' '] * bar_len
    bar[bar_len // 2] = _ansi('|', C.DIM)
    bar[bar_pos] = _ansi('◆', C.CYAN + C.BOLD)
    bar_str = '[' + ''.join(bar) + ']'
    return f'  {_ansi(label, C.BOLD):<18} {bar_str}  {_ansi(direction, dir_color):<10} offset={offset:+.3f}  dist={dist}'

def main() -> None:
    args = parse_args()
    try:
        import os
        os.system('')
    except Exception:
        pass
    print(BANNER)
    threshold = args.threshold if args.threshold is not None else MATCH_THRESHOLD
    verbose: bool = args.verbose
    print(_ansi(f'  Monitor index : {args.monitor}', C.DIM))
    print(_ansi(f'  Match threshold : {threshold}', C.DIM))
    print(_ansi(f'  Audio enabled : {not args.no_audio}', C.DIM))
    print(_ansi(f'  Verbose mode  : {verbose}', C.DIM))
    print()
    print('Initialising screen capturer...')
    screen_capturer = ScreenCapturer(roi_height_ratio=ROI_HEIGHT_RATIO, monitor_idx=args.monitor)
    screen_info = screen_capturer.get_screen_info()
    print(f"  Capture area : {screen_info['width']}×{screen_info['capture_height']} px  (top {ROI_HEIGHT_RATIO * 100:.0f}% of {screen_info['width']}×{screen_info['height']})")
    print('Initialising icon detector...')
    detector = IconDetector(target_icons=TARGET_ICONS, match_threshold=threshold, nms_iou_threshold=NMS_IOU_THRESHOLD)
    print('Initialising OCR engine...')
    ocr_engine = OCREngine()
    print('Initialising visualiser...')
    visualizer = Visualizer()
    controller: NavigationController | None = None
    if not args.no_audio:
        print('Initialising audio controller...')
        controller = NavigationController()
        print(_ansi('  [+] TTS engine ready (win32com / SAPI5)', C.DIM))
        controller.audio.tts.speak('Lancement de Compass Layer. Appuyez sur F6 pour entendre les commandes.')
    print()
    print(_ansi('  All systems ready.  Switch to the game window now.', C.BOLD))
    print(_ansi('  Global Hotkeys:', C.CYAN))
    print(_ansi('    [F6]          Read Controls', C.CYAN))
    print(_ansi('    [Shift + F8]  Scan / Sweep', C.CYAN))
    print(_ansi('    [Shift + F9]  Exit Application', C.CYAN))
    print()
    print('  Terminal Controls: [Q] quit  [SPACE] scan  [V] toggle verbose')
    request_scan = False
    request_quit = False
    request_help = False

    def on_help():
        nonlocal request_help
        request_help = True

    def on_scan():
        nonlocal request_scan
        request_scan = True

    def on_quit():
        nonlocal request_quit
        request_quit = True
    keyboard.add_hotkey('f6', on_help)
    keyboard.add_hotkey('shift+f8', on_scan)
    keyboard.add_hotkey('shift+f9', on_quit)
    frame_count = 0
    fps_timer = time.perf_counter()
    fps_display = 0.0
    try:
        while True:
            frame_count += 1
            frame_bgr = screen_capturer.get_frame()
            detections = detector.detect(frame_bgr, screen_width=screen_info['width'], screen_height=screen_info['height'], normalize_fn=screen_capturer.normalize_coord, use_laplacian=True, blur_ksize=BLUR_KSIZE)
            output_list: List[Dict[str, Any]] = []
            for det in detections:
                icon_x_rel = det['x_rel']
                icon_y_rel = det['y_rel']
                icon_w_rel = det['w_rel']
                icon_h_rel = det['h_rel']
                relative_offset = (icon_x_rel - 0.5) / COMPASS_WIDTH_RATIO
                det['rel_offset'] = relative_offset
                if not COMPASS_X_START <= icon_x_rel <= COMPASS_X_END:
                    if verbose:
                        print(_ansi(f"  [Ignore] {det['label']} outside compass (x_rel={icon_x_rel:.3f})", C.DIM))
                    continue
                if abs(relative_offset) < STRAIGHT_AHEAD_THRESHOLD:
                    direction = 'Straight'
                elif relative_offset < 0:
                    direction = 'Left'
                else:
                    direction = 'Right'
                det['direction'] = direction
                if abs(relative_offset) < 0.1:
                    dist_text = ocr_engine.extract_distance(frame_bgr, icon_x_rel, icon_y_rel, icon_w_rel, icon_h_rel, screen_info['width'], screen_info['height'])
                else:
                    dist_text = 'N/A'
                det['distance'] = dist_text
                output_list.append({'id': det['id'], 'label': det['label'], 'rel_offset': round(relative_offset, 3), 'direction': direction, 'distance': dist_text})
            if controller:
                if output_list:
                    output_list.sort(key=lambda x: x.get('score', 0), reverse=True)
                    best_icon = [output_list[0]]
                    nav_icons = from_algo_batch(best_icon)
                else:
                    nav_icons = []
                controller.update(nav_icons)
            if verbose:
                if output_list:
                    header = _ansi(f'[Frame {frame_count:>6}]  {len(output_list)} icon(s) detected  |  {fps_display:.1f} FPS', C.CYAN)
                    print(header)
                    for item in output_list:
                        print(format_detection(item))
                else:
                    print(_ansi(f'[Frame {frame_count:>6}]  — no detections —  {fps_display:.1f} FPS', C.DIM))
            if frame_count % 30 == 0:
                elapsed = time.perf_counter() - fps_timer
                fps_display = 30 / elapsed if elapsed > 0 else 0.0
                fps_timer = time.perf_counter()
            frame_vis = visualizer.draw_detections(frame_bgr, detections, screen_width=screen_info['width'], screen_height=screen_info['height'])
            visualizer.show(frame_vis)
            if request_quit:
                break
            if request_scan:
                print(_ansi('\n  [SCAN] Global hotkey triggered...', C.CYAN))
                if controller:
                    nav_icons = from_algo_batch(output_list)
                    controller.scan(nav_icons)
                request_scan = False
            if request_help:
                print(_ansi('\n  [HELP] Global hotkey triggered...', C.CYAN))
                if controller:
                    controller.audio.tts.speak("Commandes: F 6 pour entendre ce message. Majuscule F 8 pour balayer. Majuscule F 9 pour quitter le programme.")
                request_help = False
            if msvcrt.kbhit():
                key = msvcrt.getch().lower()
                if key == b'q':
                    break
                elif key == b' ':
                    print(_ansi('\n  [SCAN] Terminal space triggered...', C.CYAN))
                    if controller:
                        nav_icons = from_algo_batch(output_list)
                        controller.scan(nav_icons)
                elif key == b'v':
                    verbose = not verbose
                    state = 'ON' if verbose else 'OFF'
                    print(_ansi(f'  [V] Verbose mode {state}', C.DIM))
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        print(_ansi(f'\nERROR: {exc}', C.RED), file=sys.stderr)
        raise
    finally:
        cv2.destroyAllWindows()
        if controller:
            controller.audio.tts.speak('Arret du programme.')
            time.sleep(1.5)
            controller.stop()
        print('\n  Shutdown complete.')
        print(_ansi(f'\nSession ended. Total frames processed: {frame_count}', C.DIM))
        print(_ansi('All resources released.', C.GREEN))
if __name__ == '__main__':
    main()