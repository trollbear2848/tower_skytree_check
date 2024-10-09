from flask import Flask, request, render_template, send_from_directory
from werkzeug.utils import secure_filename
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
import os
import zipfile
import random
import cv2
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
                
                # 背景除去とランダムな変換を適用
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
    """画像にランダムなシフト、拡大縮小、回転、ぼかし、背景除去を適用"""
    
    # 背景除去を適用
    img = remove_background(img)
    
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

def remove_background(img):
    """OpenCVを使って画像から背景を除去する"""
    # PIL画像をOpenCVの形式に変換
    cv_img = np.array(img)
    cv_img = cv_img[:, :, ::-1].copy()  # RGB -> BGR

    # グレースケールに変換
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

    # 背景をマスクするために、しきい値処理を適用
    _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)

    # マスクを使って輪郭を見つける
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 最大の輪郭を取得（建物を囲む輪郭が背景と認識される可能性が高い）
    max_contour = max(contours, key=cv2.contourArea)
    
    # 背景を白に、建物を黒にしたマスクを作成
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [max_contour], -1, 255, thickness=cv2.FILLED)
    
    # 画像にマスクを適用
    result = cv2.bitwise_and(cv_img, cv_img, mask=mask)

    # 背景を白にする
    background = np.full_like(cv_img, 255)
    final_img = np.where(result == 0, background, result)

    # OpenCV形式からPIL画像に戻す
    final_img = Image.fromarray(final_img[:, :, ::-1])  # BGR -> RGB
    return final_img

if __name__ == '__main__':
    app.run()
