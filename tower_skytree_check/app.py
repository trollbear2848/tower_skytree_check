from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image, ImageOps
import numpy as np
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# アップロードフォルダの設定
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER

# モデルの読み込み
model = load_model('h5/L2N200R800B32D02E20.h5')

# ホームページのルートを定義
@app.route('/')
def home():
    return render_template('index.html')

# 予測を行うルートを定義
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'ファイルがアップロードされていません！', 400
    
    file = request.files['file']
    if file.filename == '':
        return 'ファイルが選択されていません！', 400
    
    # アップロードされたファイルを保存
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # 画像の読み込みと加工（RGBに変換）
    img = Image.open(filepath).convert('RGB')
    img = ImageOps.exif_transpose(img) 

    # 90x160にリサイズ
    img = img.resize((90, 160))

    # 白黒に変換（グレースケール）
    img = img.convert('L')

    # 加工後の画像を保存
    processed_filepath = os.path.join(app.config['PROCESSED_FOLDER'], filename)
    img.save(processed_filepath)

    # 画像をnumpy配列に変換（1チャンネルのグレースケール）
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # 正規化

    # 予測
    prediction = model.predict(img_array)
    tower_prob = round(prediction[0][0] * 100, 2)  
    skytree_prob = round(100 - tower_prob, 2)
    print(prediction[0][0])
    # 結果のテキスト
    result = f'東京スカイツリーが {skytree_prob}%<br>東京タワーが {tower_prob}% と識別されました'

    # 結果ページに加工済み画像パスと結果を渡す
    return render_template('result.html', prediction=result, image_url=processed_filepath)

if __name__ == '__main__':
    app.run()
