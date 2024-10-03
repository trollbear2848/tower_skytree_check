import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# データの前処理
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
# データの前処理（グレースケール画像を使用）
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)

# グレースケール画像としてデータをロード
train_generator = train_datagen.flow_from_directory(
    'data/train',
    target_size=(90, 160),
    batch_size=64,
    class_mode='binary',
    color_mode='grayscale'  # ここでグレースケールに設定
)

# モデルの構築
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(90, 160, 1)),  # 1チャンネル（グレースケール）に変更
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# モデルのトレーニング
model.fit(train_generator, epochs=20)
print(train_generator.class_indices)
# モデルの保存
model.save('L2N200R800B64E20.h5')
