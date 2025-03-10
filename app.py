import os
import logging
from flask import Flask, render_template, request, send_from_directory, jsonify
from PIL import Image, ImageOps, ImageFilter, ImageEnhance, ImageDraw
import numpy as np
from werkzeug.utils import secure_filename
from scipy.ndimage import maximum_filter
from scipy.cluster.vq import kmeans2

# 配置日志
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def find_light_sources(image, threshold=220, max_sources=8):
    """检测图像中的亮点（潜在光源），并进行聚类"""
    # 转换为灰度图
    gray = image.convert('L')
    # 提高对比度以更好地检测亮点
    enhancer = ImageEnhance.Contrast(gray)
    gray = enhancer.enhance(2.0)
    
    # 获取图像数据
    img_data = np.array(gray)
    
    # 使用最大值滤波器找到局部最大值
    local_max = maximum_filter(img_data, size=15)  # 增加滤波器大小以找到更大的光源区域
    peak_positions = ((img_data == local_max) & (img_data > threshold))
    
    # 获取峰值位置
    y_peaks, x_peaks = np.where(peak_positions)
    
    if len(x_peaks) == 0:
        # 如果没有找到光源，使用图像中最亮的点
        flat_idx = np.argmax(img_data)
        y_idx, x_idx = np.unravel_index(flat_idx, img_data.shape)
        return [(int(x_idx), int(y_idx))]
    
    # 将坐标组合
    coordinates = np.column_stack((x_peaks, y_peaks))
    
    # 如果点太多，使用K-means聚类减少光源数量
    if len(coordinates) > max_sources:
        try:
            centroids, labels = kmeans2(coordinates, max_sources, minit='++')
            # 确保选择的是最亮的点
            centers = []
            for centroid in centroids:
                x, y = int(centroid[0]), int(centroid[1])
                # 在周围5x5区域内找最亮的点
                region = img_data[max(0, y-2):min(y+3, img_data.shape[0]),
                                max(0, x-2):min(x+3, img_data.shape[1])]
                if region.size > 0:
                    local_y, local_x = np.unravel_index(np.argmax(region), region.shape)
                    centers.append((
                        int(max(0, x-2) + local_x),
                        int(max(0, y-2) + local_y)
                    ))
            return centers
        except:
            # 如果聚类失败，返回前max_sources个最亮的点
            brightness_values = img_data[y_peaks, x_peaks]
            brightest_indices = np.argsort(brightness_values)[-max_sources:]
            return [(int(x_peaks[i]), int(y_peaks[i])) for i in brightest_indices]
    
    return [(int(x), int(y)) for x, y in coordinates]

def create_star_ray(image, center, angle, length, width=1, color=(255, 255, 255)):
    """在指定位置创建一条光线"""
    draw_ray = Image.new('RGBA', image.size, (0, 0, 0, 0))  # 使用透明背景
    pixels = draw_ray.load()
    
    x0, y0 = center
    rad = np.radians(angle)
    
    # 使用改进的高斯衰减
    for r in range(length):
        # 计算光线的高斯衰减强度，使中心更亮，衰减更慢
        sigma = length / 2.5
        intensity = int(255 * (0.5 + 0.5 * np.exp(-(r**2)/(2*sigma**2))))
        
        # 计算光线位置
        x = int(x0 + r * np.cos(rad))
        y = int(y0 + r * np.sin(rad))
        
        # 绘制具有宽度的光线
        for w in range(-width, width+1):
            # 横向衰减
            w_factor = np.exp(-(w**2)/(2*(width/2)**2))
            
            y_offset = int(y + w * np.cos(rad))
            x_offset = int(x - w * np.sin(rad))
            
            if (0 <= x_offset < image.size[0] and 
                0 <= y_offset < image.size[1]):
                # 使用光源颜色，考虑横向衰减，并设置alpha通道
                alpha = int(intensity * w_factor)
                ray_color = (*color[:3], alpha)
                pixels[x_offset, y_offset] = ray_color
    
    return draw_ray

def create_preview_image(image, light_sources):
    """创建带有标记的预览图像"""
    preview = image.copy()
    draw = ImageDraw.Draw(preview)
    
    # 在每个光源位置画一个标记
    for x, y in light_sources:
        # 画十字标记
        marker_size = 20
        draw.line([(x - marker_size, y), (x + marker_size, y)], fill='red', width=2)
        draw.line([(x, y - marker_size), (x, y + marker_size)], fill='red', width=2)
        # 画圆圈
        draw.ellipse([(x - marker_size//2, y - marker_size//2),
                     (x + marker_size//2, y + marker_size//2)],
                    outline='yellow', width=2)
    
    return preview

def create_starburst_effect(image, filename):
    try:
        # 转换图像为RGBA模式
        if image.mode != 'RGBA':
            image = image.convert('RGBA')
        
        # 检测光源
        light_sources = find_light_sources(image)
        logger.debug(f"Found {len(light_sources)} light sources")
        
        # 创建预览图像
        preview_image = create_preview_image(image, light_sources)
        preview_path = os.path.join(app.config['UPLOAD_FOLDER'], 'preview_' + filename)
        preview_image.save(preview_path)
        
        # 创建一个新图像用于存储所有星芒效果
        width, height = image.size
        starburst = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        
        # 为每个光源创建星芒效果
        for source in light_sources:
            # 获取光源颜色并增强它
            source_color = image.getpixel(source)
            # 确保颜色足够亮
            source_color = tuple(min(255, int(c * 1.8)) for c in source_color[:3])
            
            # 计算该光源的星芒长度（基于图像大小）
            ray_length = int(min(width, height) * 0.5)  # 增加光线长度
            
            # 在不同角度创建光线
            for angle in range(0, 360, 15):  # 每15度创建一条光线
                ray = create_star_ray(image, source, angle, ray_length, width=4, color=source_color)
                # 使用alpha混合叠加光线
                starburst = Image.alpha_composite(starburst, ray)
        
        # 对星芒效果进行轻微模糊处理
        starburst = starburst.filter(ImageFilter.GaussianBlur(radius=0.5))
        
        # 将星芒效果叠加到原图上
        result = Image.alpha_composite(image, starburst)
        
        return result, 'preview_' + filename
    except Exception as e:
        logger.error(f"Error in create_starburst_effect: {str(e)}")
        raise

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            logger.error("No file part in request")
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['file']
        if file.filename == '':
            logger.error("No selected file")
            return jsonify({'error': 'No selected file'}), 400
        
        if not file or not allowed_file(file.filename):
            logger.error("Invalid file type")
            return jsonify({'error': 'Invalid file type'}), 400
        
        # 确保上传目录存在
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        
        filename = secure_filename(file.filename)
        original_path = os.path.join(app.config['UPLOAD_FOLDER'], 'original_' + filename)
        processed_path = os.path.join(app.config['UPLOAD_FOLDER'], 'processed_' + filename)
        
        # Save original image
        file.save(original_path)
        logger.debug(f"Original image saved to {original_path}")
        
        # Process image
        with Image.open(original_path) as img:
            processed_img, preview_filename = create_starburst_effect(img, filename)
            processed_img.save(processed_path)
            logger.debug(f"Processed image saved to {processed_path}")
        
        return jsonify({
            'original': 'original_' + filename,
            'processed': 'processed_' + filename,
            'preview': preview_filename
        })
    
    except Exception as e:
        logger.error(f"Error processing upload: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # 确保上传目录存在
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True) 