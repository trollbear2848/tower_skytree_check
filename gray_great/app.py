from flask import Flask, request, render_template, send_from_directory
from werkzeug.utils import secure_filename
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
import os
import zipfile
import random

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['PROCESSED_FOLDER'] = 'processed'

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
    
    # ZIPファイルを解凍
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(app.config['UPLOAD_FOLDER'])

    # 解凍されたフォルダのパス
    folder_name = os.path.splitext(filename)[0]
    folder_path = os.path.join(app.config['UPLOAD_FOLDER'], folder_name)

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
                
                # ランダムでシフト、拡大縮小、回転、ぼかし
                img = apply_random_transformations(img)
                
                # 90x160にリサイズ
                img = img.resize((90, 160))
                
                # 変換された画像を保存
                processed_path = os.path.join(app.config['PROCESSED_FOLDER'], 'processed_' + filename)
                img.save(processed_path)
                
                processed_files.append('processed_' + filename)
            except Exception as e:
                print(f"{filename} の処理に失敗しました: {e}")
                continue
    
    return render_template('processed.html', files=processed_files)

@app.route('/processed/<filename>')
def get_processed_file(filename):
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename)

def apply_random_transformations(img):
    """画像にランダムなシフト、拡大縮小、回転、ぼかしを適用"""
    # ランダムに回転 (0~360度)
    if random.random() > 0.5:
        angle = random.uniform(0, 360)
        img = img.rotate(angle)

    # ランダムに拡大縮小 (80%~120%)
    if random.random() > 0.5:
        scale_factor = random.uniform(0.8, 1.2)
        new_size = (int(img.size[0] * scale_factor), int(img.size[1] * scale_factor))
        img = img.resize(new_size, Image.ANTIALIAS)
        img = img.crop((0, 0, 90, 160))  # サイズを90x160にトリミング

    # ランダムでぼかしを適用
    if random.random() > 0.5:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0, 5)))

    return img

if __name__ == '__main__':
    app.run()
