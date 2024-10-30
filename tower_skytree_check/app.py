from flask import Flask, request, render_template, send_from_directory
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image, ImageOps
import numpy as np
import os
from werkzeug.utils import secure_filename
import cv2
from rembg import remove

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER

# モデルの読み込み
model = load_model('h5/L3M150MR1350B16E10.h5')

def remove_background(input_path, output_path):
    input_img = cv2.imread(input_path)
    if input_img is None:
        raise ValueError(f"入力画像が見つかりません: {input_path}")
    output_img = remove(input_img)
    cv2.imwrite(output_path, output_img)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return render_template('error.html', error_message='ファイルがアップロードされていません！'), 400
    file = request.files['file']
    if file.filename == '':
        return render_template('error.html', error_message='ファイルが選択されていません！'), 400
    filename = secure_filename(file.filename)
    input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(input_path)

    output_filename = filename.rsplit('.', 1)[0] + '.png'
    output_path = os.path.join(app.config['PROCESSED_FOLDER'], output_filename)

    try:
        # 背景除去
        remove_background(input_path, output_path)
        
        # 画像の読み込みと加工（RGBに変換）
        img = Image.open(output_path).convert('RGB')
        img = ImageOps.exif_transpose(img)
        # 画像の幅と高さを取得
        width, height = img.size
        # 画像が横向きの場合は90度回転
        if width > height:
            img = img.rotate(90, expand=True)
        img = img.resize((90, 160))
        img = img.convert('L')
        # 加工後の画像を保存
        processed_filepath = os.path.join(app.config['PROCESSED_FOLDER'], output_filename)
        img.save(processed_filepath)
        # 画像をnumpy配列に変換（1チャンネルのグレースケール）
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
        # 予測
        prediction = model.predict(img_array)
        classes = ['none', 'skytree', 'tower']
        predicted_class = classes[np.argmax(prediction)]
        # クラスごとの確率をコンソールに出力
        print("各クラスの確率:")
        for i, prob in enumerate(prediction[0]):
            print(f'{classes[i]}: {prob * 100:.2f}%')
        # 結果のテキスト
        result = f'予測結果: {predicted_class}'
        # 結果ページに加工済み画像パスと結果を渡す
        return render_template('result.html', prediction=result, image_url=output_filename)
    except Exception as e:
        return render_template('error.html', error_message=f'エラーが発生しました: {str(e)}'), 500

@app.route('/processed/<filename>')
def processed_file(filename):
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename)

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    if not os.path.exists(app.config['PROCESSED_FOLDER']):
        os.makedirs(app.config['PROCESSED_FOLDER'])
    app.run(debug=True)
