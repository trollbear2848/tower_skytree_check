import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# オプティマイザ設定のオン/オフを設定するフラグ
use_custom_optimizer = True
use_early_stopping = False

# データの前処理（グレースケール画像を使用）
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)  # 20%を検証用データに
train_generator = datagen.flow_from_directory(
    'data/train',
    target_size=(90, 160),
    batch_size=16,
    class_mode='categorical',
    color_mode='grayscale',
    subset='training')
validation_generator = datagen.flow_from_directory(
    'data/train',
    target_size=(90, 160),
    batch_size=16,
    class_mode='categorical',
    color_mode='grayscale',
    subset='validation')

# モデルの構築
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(90, 160, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(3, activation='softmax')  # 出力層を3ユニットに変更し、softmaxを使用
])

# オプティマイザの設定
if use_custom_optimizer:
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
else:
    optimizer = 'adam'

# モデルのコンパイル
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# 早期終了の設定
callbacks = []
if use_early_stopping:
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    callbacks.append(early_stopping)

# モデルのトレーニングと検証
history = model.fit(train_generator, epochs=10, validation_data=validation_generator, callbacks=callbacks)

# 学習曲線のプロット
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(1, len(acc) + 1)

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

print(train_generator.class_indices)

# モデルの保存
model.save('L3CM105B16E10.h5')
