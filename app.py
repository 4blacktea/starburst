import os
import logging
from flask import Flask, render_template, request, send_from_directory, jsonify
from PIL import Image, ImageOps, ImageFilter
import numpy as np
from werkzeug.utils import secure_filename

# 配置日志
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def create_starburst_effect(image):
    try:
        # Convert image to RGB if it's not
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Create a new image with the same size
        width, height = image.size
        
        # Create star rays
        draw_rays = Image.new('RGB', (width, height), (0, 0, 0))
        pixels = draw_rays.load()
        
        center_x = width // 2
        center_y = height // 2
        
        # Draw rays
        for angle in range(0, 360, 30):  # 12 rays
            rad = np.radians(angle)
            for r in range(0, int(min(width, height) * 0.5)):
                x = int(center_x + r * np.cos(rad))
                y = int(center_y + r * np.sin(rad))
                if 0 <= x < width and 0 <= y < height:
                    pixels[x, y] = (255, 255, 255)
        
        # Blur the rays
        draw_rays = draw_rays.filter(ImageFilter.GaussianBlur(radius=3))
        
        # Blend the original image with the rays
        result = Image.blend(image, draw_rays, 0.3)
        return result
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
            processed_img = create_starburst_effect(img)
            processed_img.save(processed_path)
            logger.debug(f"Processed image saved to {processed_path}")
        
        return jsonify({
            'original': 'original_' + filename,
            'processed': 'processed_' + filename
        })
    
    except Exception as e:
        logger.error(f"Error processing upload: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # 确保上传目录存在
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True) 