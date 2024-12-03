from flask import Flask, request, render_template, send_from_directory
import os
import zipfile
from werkzeug.utils import secure_filename
from remove_bg import remove_background

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'ファイルがアップロードされていません！', 400
    file = request.files['file']
    if file.filename == '':
        return 'ファイルが選択されていません！', 400

    filename = secure_filename(file.filename)
    input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(input_path)

    if zipfile.is_zipfile(input_path):
        with zipfile.ZipFile(input_path, 'r') as zip_ref:
            zip_ref.extractall(app.config['UPLOAD_FOLDER'])

        processed_files = []
        for file_info in zip_ref.infolist():
            extracted_file_path = os.path.join(app.config['UPLOAD_FOLDER'], file_info.filename)
            
            # デバッグメッセージ
            print(f"抽出されたファイルのパス: {extracted_file_path}")

            if not os.path.isfile(extracted_file_path):
                print(f"ファイルが見つかりません: {extracted_file_path}")
                continue

            output_filename = secure_filename(file_info.filename.rsplit('.', 1)[0] + '.png')
            output_path = os.path.join(app.config['PROCESSED_FOLDER'], output_filename)

            # デバッグメッセージ
            print(f"出力ファイルのパス: {output_path}")

            remove_background(extracted_file_path, output_path)
            processed_files.append(output_filename)

        return render_template('result.html', filenames=processed_files)

    return 'アップロードされたファイルはZIPファイルではありません。', 400

@app.route('/processed/<filename>')
def processed_file(filename):
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename)

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    if not os.path.exists(app.config['PROCESSED_FOLDER']):
        os.makedirs(app.config['PROCESSED_FOLDER'])
    app.run(debug=True)
