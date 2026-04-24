import os
import sys
import datetime
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
#from build_saan import build_saan_model
from build_saan import build_sota_saan


class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        # utf-8 is required so Windows doesn't crash when saving the emojis!
        self.log = open(filename, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        self.terminal.flush()
        self.log.flush()

timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
log_filename = f"../logs_densenet_{timestamp}.txt"
sys.stdout = Logger(log_filename)
print(f"📝 Live logging activated! Saving everything to {log_filename}")
# 👆 ------------------------------------ 👆

print("🔍 Checking Hardware...")
# --- THE GPU KILL SWITCH ---
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) == 0:
    raise RuntimeError("❌ NO GPU DETECTED! TensorFlow is trying to use the CPU. Aborting.")
else:
    print(f"✅ GPU Detected: {physical_devices[0]}")
    # This prevents TensorFlow from hoarding all 4GB of VRAM instantly
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
# ---------------------------

print("🚀 Initializing GPU and Mixed Precision...")
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# --- 1. Load the GPS Map (CSV) ---
CSV_PATH = '../master_labels_60k.csv'  # <--- Updated to the new 60k file

# ⚠️ PASTE THE EXACT PATH FROM YOUR WINDOWS EXPLORER HERE ⚠️
# Keep the 'r' before the quotes so Windows doesn't break the slashes
EXTERNAL_IMAGES_DIR = r"C:\galaxy_datasets\images_training_rev1"

df = pd.read_csv(CSV_PATH)
df['filepath'] = EXTERNAL_IMAGES_DIR + '/' + df['filename']

print("🔍 Verifying file paths...")
df = df[df['filepath'].apply(os.path.exists)]
print(f"Valid images ready for training: {len(df)}")

# --- 2. Encode Labels ---  
encoder = LabelEncoder()
df['label_encoded'] = encoder.fit_transform(df['Label'])
num_classes = len(encoder.classes_)

train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['label_encoded'], random_state=42)

# --- 3. The Data Streaming Pipeline ---
IMG_SIZE = (64, 64)
BATCH_SIZE = 64 

def process_image(file_path, label):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    
    # --- 🌌 DATA AUGMENTATION (The "Pro" Trick) ---
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    # ----------------------------------------------
    
    #img = img / 255.0
    label = tf.one_hot(label, depth=num_classes)
    return img, label

def create_dataset(dataframe):
    ds = tf.data.Dataset.from_tensor_slices((dataframe['filepath'].values, dataframe['label_encoded'].values))
    ds = ds.map(process_image, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds

print("🌊 Building Data Streams...")
train_dataset = create_dataset(train_df)
val_dataset = create_dataset(val_df)

# --- 4. Build and Train the Model ---
TARGET_MODEL = 'efficientnet' # CHANGE THIS TO 'densenet' OR 'convnext' LATER

print(f"🏗️ Building SAAN Architecture with {TARGET_MODEL}...")
model = build_sota_saan(backbone_name=TARGET_MODEL, input_shape=(64, 64, 3), num_classes=num_classes)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint(f'../models/saan_{TARGET_MODEL}_best.keras', save_best_only=True)
]

from sklearn.utils.class_weight import compute_class_weight

print("⚖️ Calculating Class Weights to prevent lazy predictions...")
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_df['label_encoded']),
    y=train_df['label_encoded']
)
weight_dict = dict(enumerate(class_weights))

print(f"🔥 Starting GPU Training Pipeline...")
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=30,
    class_weight=weight_dict,
    callbacks=callbacks,
    verbose=2 # <--- ADD THIS LINE!
)

print("✅ TRAINING COMPLETE. Best model saved to models/saan_best_model_60k.keras")