import os
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# --- Configuration ---
DATA_DIR = '../data/processed_images'
MODEL_DIR = '../models'
BATCH_SIZE = 32
IMG_SIZE = (64, 64)
EPOCHS = 10

os.makedirs(MODEL_DIR, exist_ok=True)

# --- 1. Load and Split the Data ---
print("Loading datasets...")
train_dataset = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=42, # Keeps the split consistent
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

val_dataset = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

class_names = train_dataset.class_names
print(f"Classes found: {class_names}")

# Optimize dataset loading for your hardware
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_dataset = val_dataset.cache().prefetch(buffer_size=AUTOTUNE)

# --- 2. Build the CNN Architecture ---
print("Building CNN model...")
model = models.Sequential([
    layers.Rescaling(1./255, input_shape=(64, 64, 3)), # Normalize pixel values to [0,1]
    
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5), # Helps prevent overfitting
    layers.Dense(len(class_names), activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# --- 3. Train the Model ---
print("\nStarting training...")
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS
)

# --- 4. Save the Model ---
model_path = os.path.join(MODEL_DIR, 'galaxy_cnn_baseline.keras')
model.save(model_path)
print(f"\nModel saved successfully to {model_path}")