import cv2
import numpy as np
import os
import argparse

def process_icon(input_path: str, output_path: str=None) -> None:
    if not os.path.exists(input_path):
        print(f'Error: 找不到图片 {input_path}')
        return
    img_bgr = cv2.imread(input_path)
    if img_bgr is None:
        print(f'Error: 无法加载图片 {input_path}')
        return
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    edges = cv2.Canny(gray, 50, 150)
    kernel = np.ones((3, 3), np.uint8)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print('未检测到任何有效的图标轮廓。')
        return
    max_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(max_contour)
    cropped_bgr = img_bgr[y:y + h, x:x + w]
    mask = np.zeros(img_bgr.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [max_contour], -1, 255, thickness=cv2.FILLED)
    cropped_mask = mask[y:y + h, x:x + w]
    max_side = max(w, h)
    padding = 6
    canvas_size = max_side + padding * 2
    canvas_bgra = np.zeros((canvas_size, canvas_size, 4), dtype=np.uint8)
    start_y = (canvas_size - h) // 2
    start_x = (canvas_size - w) // 2
    for c in range(3):
        canvas_bgra[start_y:start_y + h, start_x:start_x + w, c] = cropped_bgr[:, :, c]
    canvas_bgra[start_y:start_y + h, start_x:start_x + w, 3] = cropped_mask
    if output_path is None:
        filename, ext = os.path.splitext(input_path)
        output_path = f'{filename}_centered.png'
    cv2.imwrite(output_path, canvas_bgra)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='图标去背与完美居中生成器')
    parser.add_argument('input', help='原始截屏图标的路径')
    parser.add_argument('-o', '--output', help='导出的目标路径（默认同级加上_centered）')
    args = parser.parse_args()
    process_icon(args.input, args.output)