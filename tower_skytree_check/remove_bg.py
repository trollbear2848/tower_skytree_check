from rembg import remove
import cv2

def remove_background(input_path, output_path):
    input_img = cv2.imread(input_path)
    if input_img is None:
        raise ValueError(f"入力画像が見つかりません: {input_path}")
    output_img = remove(input_img)
    cv2.imwrite(output_path, output_img)
