import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from build_saan import build_saan_model # Importing your custom architecture!

# --- 1. Load and Prepare Data ---
DATA_DIR = '../data/processed_images'
CLASSES = ['Elliptical', 'Spiral', 'Irregular']

X = []
y = []

print("Loading 6,000 images into memory...")
for label, class_name in enumerate(CLASSES):
    class_dir = os.path.join(DATA_DIR, class_name)
    for img_name in os.listdir(class_dir):
        img_path = os.path.join(class_dir, img_name)
        img = cv2.imread(img_path)
        if img is not None:
            X.append(img)
            y.append(label)

X = np.array(X, dtype='float32')
y = np.array(y)

# Z-Score / Mathematical Normalization (Crucial for Neural Networks)
print("Normalizing pixel values (0 to 1)...")
X = X / 255.0 

# Split into 80% Training, 20% Validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Training shape: {X_train.shape} | Validation shape: {X_val.shape}")

# --- 2. Initialize the Custom SAAN Model ---
print("\nInitializing SAAN Architecture...")
model = build_saan_model(input_shape=(64, 64, 3), num_classes=3)

# The "Punishment" Engine: AdamW optimizer and Label Smoothing
optimizer = tf.keras.optimizers.AdamW(learning_rate=0.001, weight_decay=1e-4)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)

model.compile(
    optimizer=optimizer,
    loss=loss_fn,
    metrics=['accuracy']
)

# --- 3. Training Callbacks ---
# Save the absolute best version of the model automatically
checkpoint = ModelCheckpoint(
    '../models/saan_best_model.keras', 
    monitor='val_accuracy', 
    save_best_only=True, 
    mode='max', 
    verbose=1
)

# Stop training early if the model stops improving (prevents memorizing noise)
early_stopping = EarlyStopping(
    monitor='val_loss', 
    patience=5, 
    restore_best_weights=True
)

# --- 4. Unleash the GPU ---

print("\nStarting Training Phase...")
history = model.fit(
    X_train, y_train,
    epochs=25,
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=[checkpoint, early_stopping]
)

print("\nTraining Complete! Best model saved to 'models/saan_best_model.keras'.")