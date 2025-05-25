import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from sklearn.model_selection import train_test_split

real_video_dir = r"/kaggle/input/celeb-df/black/Celeb-real"  # INSERT APPLICABLE PATH
youtube_real_dir = r"/kaggle/input/celeb-df/black/YouTube-real"
fake_video_dir = r"/kaggle/input/celeb-df/black/Celeb-synthesis"

# The below code combines he real and fake videos in two lists

real_videos = [os.path.join(real_video_dir, file) for file in os.listdir(real_video_dir) if file.endswith('.mp4')]
real_videos += [
    os.path.join(youtube_real_dir, file) for file in os.listdir(youtube_real_dir) if file.endswith('.mp4')
]
fake_videos = [os.path.join(fake_video_dir, file) for file in os.listdir(fake_video_dir) if file.endswith('.mp4')]

# test if the above step has been successfully executed
print(len(real_videos))
print(len(fake_videos))

# Extract every second frame using computer vision
def extract_every_2nd_frame(video_path, size=(224, 224)):
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0

    while True:
        success, frame_vec = cap.read()
        if not success:
            break  # Break if we can't read the frame

        # Convert the frame to RGB if it is not in that format
        if frame_vec.shape[-1] != 3:  # If not 3 channels, convert to RGB
            frame_vec = cv2.cvtColor(frame_vec, cv2.COLOR_BGR2RGB)
        
        # Resize frame to the specified size
        frame_resized = cv2.resize(frame_vec, size)

        # Only add every 3rd frame
        if frame_count % 2 == 0:
            frames.append(frame_resized)

        frame_count += 1

    cap.release()
    return frames

# Applying ELA (needs further processing)
def apply_ela(image, quality=90):
    if image.shape[-1] != 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, compressed = cv2.imencode('.jpg', image, encode_param)
    compressed_image = cv2.imdecode(compressed, cv2.IMREAD_COLOR)

    ela_image = cv2.absdiff(image, compressed_image).astype(np.float32)

    epsilon = 1e-6
    for c in range(3):
        channel = ela_image[..., c]
        min_val, max_val = channel.min(), channel.max()
        range_val = max_val - min_val
        ela_image[..., c] = 255 * (channel - min_val) / (range_val + epsilon)

    ela_image = ela_image.astype(np.float32) / 255.0

    if np.isnan(ela_image).any() or np.isinf(ela_image).any():
        print("❌ NaN or Inf found in ELA image!")

    return ela_image

# Frame generator(to minimize RAM usage, while maintaining accuracy)
class FrameGenerator(tf.keras.utils.Sequence):
    def __init__(self, video_list, batch_size=8, frame_size=(224, 224), shuffle=True):
        self.video_list = video_list
        self.batch_size = batch_size
        self.frame_size = frame_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.video_list) / self.batch_size))

    def __getitem__(self, index):
        batch_videos = self.video_list[index * self.batch_size : (index + 1) * self.batch_size]
        batch_images = []
        batch_labels = []

        for video_path, label in batch_videos:
            try:
                frames = extract_every_2nd_frame(video_path, size=self.frame_size)
                for frame in frames:
                    if frame is None or frame.shape != (224, 224, 3):
                        print(f"⚠️ Skipping invalid frame: {video_path}")
                        continue

                    ela_frame = apply_ela(frame)

                    if np.isnan(ela_frame).any() or np.isinf(ela_frame).any():
                        print(f"❌ Found NaN/Inf in frame from {video_path}")
                        continue

                    batch_images.append(ela_frame)
                    batch_labels.append(label)

            except Exception as e:
                print(f"💥 Error processing {video_path}: {e}")
                continue

        return np.array(batch_images) / 255.0, np.array(batch_labels)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.video_list)

# Callback to catch NaNs during training
class NaNLogger(tf.keras.callbacks.Callback):
    def on_batch_end(self, batch, logs=None):
        if logs:
            loss = logs.get('loss')
            acc = logs.get('accuracy')
            if loss is None or np.isnan(loss) or np.isnan(acc):
                print(f"🚨 NaN detected on batch {batch}, loss={loss}, acc={acc}")
                self.model.stop_training = True
strategy = tf.distribute.MirroredStrategy()
print(f"✅ Number of GPUs Available: {strategy.num_replicas_in_sync}")

# Gather videos and labels
video_paths = [
    r"/kaggle/input/celeb-df/black/Celeb-synthesis",
    r"/kaggle/input/celeb-df/black/Celeb-real",
    r"/kaggle/input/celeb-df/black/YouTube-real"
]
labels = [0, 1, 1]

# Load video paths
all_videos = []
for path, label in zip(video_paths, labels):
    videos = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.mp4')]
    all_videos.extend([(video, label) for video in videos])

# Train-validation split
train_videos, val_videos = train_test_split(all_videos, test_size=0.2, random_state=42)

# Data Generators (with larger batch size)
train_gen = FrameGenerator(train_videos, batch_size=8)
val_gen = FrameGenerator(val_videos, batch_size=8)

# Model inside strategy scope (For training on kaggle dual GPU's)
with strategy.scope():
    analyzer = tf.keras.applications.ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    analyzer.trainable = False

    model = models.Sequential([
        analyzer,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer='adam',
        # Uncomment below if still NaNs after these fixes:
        # loss=tf.keras.losses.BinaryFocalCrossentropy(),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

# Show model structure
model.summary()

# Train with NaN logging
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=10,
    callbacks=[NaNLogger()],
)

#  Save the model
model.save('ElaCNN.h5')
print("✅ Model Saved Successfully as 'ElaCNN.h5'")

# Plot Accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title("Accuracy over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.show()
