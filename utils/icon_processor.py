import cv2
import numpy as np
import os
import argparse

def process_icon(input_path: str, output_path: str = None) -> None:
    """
    智能去背并居中截取图标，将其生成为支持 OpenCV Mask 的完美 4 通道 (BGRA) 模板。
    """
    if not os.path.exists(input_path):
        print(f"Error: 找不到图片 {input_path}")
        return

    # 1. 读入原始被截下来的脏图标
    img_bgr = cv2.imread(input_path)
    if img_bgr is None:
        print(f"Error: 无法加载图片 {input_path}")
        return
        
    print(f"正在处理图片: {os.path.basename(input_path)}")
    print(f"初始形状: {img_bgr.shape}")

    # 2. 转换为灰度图像并进行二值化，寻找最明显的形状（菱形）
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # 使用大津法 (Otsu) 自动选取最佳阈值进行二值化
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 偶尔前景和背景颜色可能反转（白底黑字还是黑底白字），这里做个强边缘发掘替代方案：
    # 通过 Canny 边缘检测加轮廓发现也行，但基于当前场景，通常图标较亮
    edges = cv2.Canny(gray, 50, 150)
    
    # 形态学闭操作：把菱形的边界连接起来形成完整的纯色块
    kernel = np.ones((3, 3), np.uint8)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    
    # 3. 寻找所有外部轮廓
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print("未检测到任何有效的图标轮廓。")
        return
        
    # 假设最大的可见轮廓就是我们的图标 (菱形)
    max_contour = max(contours, key=cv2.contourArea)
    
    # 获取此轮廓的最紧密包围盒 (Bounding Box)
    x, y, w, h = cv2.boundingRect(max_contour)
    print(f"检测到的主要轮廓边界框为: x={x}, y={y}, w={w}, h={h}")
    
    # 提取这个最紧凑的区域
    cropped_bgr = img_bgr[y:y+h, x:x+w]
    
    # 为原图中的这个轮廓范围单独拉取一个前景 Mask
    # 我们在新画布上将图画出，保留最大的形状内部像素
    mask = np.zeros(img_bgr.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [max_contour], -1, 255, thickness=cv2.FILLED)
    
    # 将此 mask 同样进行紧凑截取
    cropped_mask = mask[y:y+h, x:x+w]

    # 4. 创建一个完美正方形的透明画布，强制所有维度中心对称
    # 为防止旋转匹配或边界溢出，正方形边长以菱形的宽高的最大值再加一定 padding 为准
    max_side = max(w, h)
    padding = 6  # 留白保护边缘
    canvas_size = max_side + padding * 2
    
    # 画布：4通道，全 0（完美透明黑色）
    canvas_bgra = np.zeros((canvas_size, canvas_size, 4), dtype=np.uint8)
    
    # 5. 精确计算居中的起止坐标
    start_y = (canvas_size - h) // 2
    start_x = (canvas_size - w) // 2
    
    # 6. 将原始颜色 (BGR) 放进去，并将抽出来的物体形状作为 (Alpha)
    for c in range(3): # B, G, R 通道
        canvas_bgra[start_y:start_y+h, start_x:start_x+w, c] = cropped_bgr[:, :, c]
        
    # 将计算好的纯粹形状遮罩写入到 Alpha 通道中
    # 由于只有 mask 为 255 的地方（菱形内部）Alpha通道才是不透明的，除此之外全部透明
    canvas_bgra[start_y:start_y+h, start_x:start_x+w, 3] = cropped_mask
    
    # 7. 保存文件
    if output_path is None:
        filename, ext = os.path.splitext(input_path)
        output_path = f"{filename}_centered.png"
        
    cv2.imwrite(output_path, canvas_bgra)
    print(f"成功导出带透明完美居中掩码的模板至 {output_path}")
    print(f"最终模板画板大小为纯正方形: {canvas_size}x{canvas_size}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="图标去背与完美居中生成器")
    parser.add_argument("input", help="原始截屏图标的路径")
    parser.add_argument("-o", "--output", help="导出的目标路径（默认同级加上_centered）")
    
    args = parser.parse_args()
    process_icon(args.input, args.output)
