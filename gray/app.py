from flask import Flask, request, render_template, send_from_directory
from werkzeug.utils import secure_filename
from PIL import Image, ImageOps
import os
import zipfile

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

if __name__ == '__main__':
    app.run(debug=True)
