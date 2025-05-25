import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from datetime import datetime

# Configuration
VIDEO_PATHS = {
    "celeb_real": "/kaggle/input/celeb-df/black/Celeb-real",
    "youtube_real": "/kaggle/input/celeb-df/black/YouTube-real",
    "celeb_fake": "/kaggle/input/celeb-df/black/Celeb-synthesis"
}
IMG_SIZE = (224, 224)
BATCH_SIZE = 8
EPOCHS = 10

def collect_video_paths():
    def list_mp4(path): return [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.mp4')]

    real = list_mp4(VIDEO_PATHS["celeb_real"]) + list_mp4(VIDEO_PATHS["youtube_real"])
    fake = list_mp4(VIDEO_PATHS["celeb_fake"])
    print(f"🟢 Real videos: {len(real)} | 🔴 Fake videos: {len(fake)}")
    return [(p, 1) for p in real] + [(p, 0) for p in fake]

def grab_frames(video_file, target_size=IMG_SIZE):
    frames_out = []
    try:
        cap = cv2.VideoCapture(video_file)
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % 2 == 0:
                frame = cv2.resize(frame, target_size)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames_out.append(frame)
            idx += 1
        cap.release()
    except Exception as e:
        print(f"❌ Error reading {video_file}: {e}")
    return frames_out

def compute_ela(img, quality=90):
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, enc = cv2.imencode('.jpg', img, encode_params)
    dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    ela_img = cv2.absdiff(img, dec).astype(np.float32)

    for i in range(3):
        c = ela_img[..., i]
        ela_img[..., i] = 255 * (c - c.min()) / (c.max() - c.min() + 1e-6)

    return ela_img.astype(np.float32) / 255.0

class ELADataGen(tf.keras.utils.Sequence):
    def __init__(self, video_label_pairs, batch_size=BATCH_SIZE, shuffle=True):
        self.video_data = video_label_pairs
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.video_data) / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.video_data)

    def __getitem__(self, idx):
        batch = self.video_data[idx * self.batch_size:(idx + 1) * self.batch_size]
        images, labels = [], []

        for video_path, label in batch:
            frames = grab_frames(video_path)
            for frame in frames:
                ela = compute_ela(frame)
                if not np.isnan(ela).any():
                    images.append(ela)
                    labels.append(label)

        return np.array(images), np.array(labels)

class NaNStopper(tf.keras.callbacks.Callback):
    def on_batch_end(self, batch, logs=None):
        if logs:
            if np.isnan(logs.get('loss')) or np.isnan(logs.get('accuracy')):
                print(f"⚠️ NaN detected at batch {batch} - Stopping Training")
                self.model.stop_training = True

def build_model():
    base = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base.trainable = False
    model = models.Sequential([
        base,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def run_training():
    data = collect_video_paths()
    train_set, val_set = train_test_split(data, test_size=0.2, random_state=42)

    train_gen = ELADataGen(train_set)
    val_gen = ELADataGen(val_set)

    strategy = tf.distribute.MirroredStrategy()
    print(f"🚀 Using {strategy.num_replicas_in_sync} GPUs")

    with strategy.scope():
        model = build_model()
    model.summary()

    history = model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS, callbacks=[NaNStopper()])

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"ela_detector_{timestamp}.h5"
    model.save(model_name)
    print(f"✅ Model saved to: {model_name}")

    plt.figure(figsize=(8, 5))
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title("Accuracy Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    run_training()
