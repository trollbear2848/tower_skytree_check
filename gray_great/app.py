from flask import Flask, request, render_template, send_from_directory
from werkzeug.utils import secure_filename
from PIL import Image, ImageOps, ImageEnhance, ImageFilter, ImageChops
import os
import zipfile
import random
import cv2
from rembg import remove
import numpy as np

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['PROCESSED_FOLDER'] = 'processed'

# アップロードフォルダと処理済みフォルダを作成
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])
if not os.path.exists(app.config['PROCESSED_FOLDER']):
    os.makedirs(app.config['PROCESSED_FOLDER'])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    if 'file' not in request.files:
        return "ファイルがありません"
    file = request.files['file']
    if file.filename == '':
        return "ファイルが選択されていません"
    
    # 一時的にZIPファイルを保存
    filename = secure_filename(file.filename)
    zip_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(zip_path)
    
    # 解凍先のフォルダ名を明示的に定義
    folder_name = os.path.splitext(filename)[0]
    folder_path = os.path.join(app.config['UPLOAD_FOLDER'], folder_name)
    
    # ZIPファイルを解凍
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(folder_path)
    
    processed_files = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            try:
                img = Image.open(file_path)
                # EXIFデータに基づいて正しい向きに回転
                img = ImageOps.exif_transpose(img)
                # グレースケールに変換
                img = img.convert('L')

                # 各画像をn回処理
                for i in range(1):
                    # ランダムな変換を適用
                    transformed_img = apply_random_transformations(
                        img.copy(),
                        rotate=request.form.get('rotate'),
                        resize=request.form.get('resize'),
                        shift=request.form.get('shift'),
                        blur=request.form.get('blur'),
                        remove_bg=request.form.get('remove_bg')
                    )
                    # 90x160にリサイズ
                    transformed_img = transformed_img.resize((90, 160), Image.Resampling.LANCZOS)
                    # 変換された画像を新しいファイル名で保存
                    new_filename = f"processed_{i}_{random.randint(1000, 9999)}_{filename}"
                    processed_path = os.path.join(app.config['PROCESSED_FOLDER'], new_filename)
                    transformed_img = transformed_img.convert("RGB")  # ここでRGBに変換
                    transformed_img.save(processed_path, format="JPEG")  # ここでJPEG形式で保存
                    processed_files.append(new_filename)
            except Exception as e:
                print(f"{filename} の処理に失敗しました: {e}")
                continue
    return render_template('processed.html', files=processed_files)


@app.route('/processed/<filename>')
def get_processed_file(filename):
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename)

def apply_random_transformations(img, rotate, resize, shift, blur, remove_bg):
    """画像にランダムなシフト、拡大縮小、回転、ぼかしを適用"""
    if rotate and random.random() > 0.5:
        angle = random.uniform(0, 360)
        img = img.rotate(angle)
    if resize and random.random() > 0.5:
        scale_factor = random.uniform(0.8, 1.2)
        new_size = (int(img.size[0] * scale_factor), int(img.size[1] * scale_factor))
        img = img.resize(new_size, Image.Resampling.LANCZOS)
        img = img.crop((0, 0, 90, 160))  # サイズを90x160にトリミング
    if shift and random.random() > 0.5:
        max_shift = 50  # 移動の最大ピクセル数
        dx = random.randint(-max_shift, max_shift)
        dy = random.randint(-max_shift, max_shift)
        img = ImageChops.offset(img, dx, dy)
        img = img.crop((max(0, -dx), max(0, -dy), img.width - max(0, dx), img.height - max(0, dy)))
    if blur and random.random() > 0.5:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0, 5)))
    if remove_bg:
        img = remove(np.array(img))
        img = Image.fromarray(img)
    return img

if __name__ == '__main__':
    app.run(debug=True)
