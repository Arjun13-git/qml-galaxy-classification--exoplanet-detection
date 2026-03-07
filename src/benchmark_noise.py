# import os
# import numpy as np
# import tensorflow as tf
# import pandas as pd

# # --- Configuration ---
# DATA_DIR = '../data/processed_images'
# MODEL_PATH = '../models/galaxy_cnn_baseline.keras'
# BATCH_SIZE = 32
# IMG_SIZE = (64, 64)
# SNR_LEVELS = [10, 5, 1, 0.1]

# # --- 1. Load Model and Validation Data ---
# print("Loading model and validation data...")
# model = tf.keras.models.load_model(MODEL_PATH)

# val_dataset = tf.keras.utils.image_dataset_from_directory(
#     DATA_DIR,
#     validation_split=0.2,
#     subset="validation",
#     seed=42,
#     image_size=IMG_SIZE,
#     batch_size=BATCH_SIZE,
#     shuffle=True # Keep order so we can calculate exact metrics
# )

# # Extract images and true labels into numpy arrays for easier manipulation
# val_images = []
# val_labels = []
# for images, labels in val_dataset:
#     val_images.append(images.numpy())
#     val_labels.append(labels.numpy())

# val_images = np.concatenate(val_images)
# val_labels = np.concatenate(val_labels)

# # Normalize images to [0, 1] just like the Rescaling layer does internally
# # (We do this here so our noise math scales correctly)
# val_images = val_images / 255.0 

# # --- 2. Noise Injection Function ---
# def inject_gaussian_noise(images, target_snr):
#     """
#     Injects Gaussian noise based on the target SNR.
#     """
#     noisy_images = np.zeros_like(images)
    
#     for i in range(images.shape[0]):
#         img = images[i]
#         # Calculate signal power (variance of the image pixels)
#         signal_power = np.var(img)
        
#         # Calculate required noise variance
#         noise_variance = signal_power / target_snr
        
#         # Generate Gaussian noise
#         noise = np.random.normal(0, np.sqrt(noise_variance), img.shape)
        
#         # Add noise and clip values to keep them in valid pixel range [0, 1]
#         noisy_img = np.clip(img + noise, 0.0, 1.0)
#         noisy_images[i] = noisy_img
        
#     return noisy_images

# # --- 3. Run the SNR Sweep ---
# results = []

# print("\nRunning SNR Benchmark Sweep...")
# for snr in SNR_LEVELS:
#     print(f"Testing SNR = {snr}...")
    
#     # Inject noise
#     noisy_val_images = inject_gaussian_noise(val_images, snr)
    
#     # Evaluate model
#     # (We multiply by 255 because our model's first layer is a Rescaling(1./255) layer, 
#     # so we need to feed it data in the original [0, 255] format)
#     loss, accuracy = model.evaluate(noisy_val_images * 255.0, val_labels, verbose=0)
    
#     results.append({
#         'Model': 'Classical CNN',
#         'Target_SNR': snr,
#         'Accuracy': round(accuracy * 100, 2)
#     })

# # --- 4. Display Results ---
# df_results = pd.DataFrame(results)
# print("\n--- Benchmarking Results ---")
# print(df_results.to_markdown(index=False))

import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score
# Import the custom architecture so TensorFlow knows how to rebuild the saved model
from build_saan import LearnableNoiseGate, SEBlock

# --- 1. Load Data ---
DATA_DIR = '../data/processed_images'
CLASSES = ['Elliptical', 'Spiral', 'Irregular']

X_test = []
y_test = []

print("Loading test images...")
for label, class_name in enumerate(CLASSES):
    class_dir = os.path.join(DATA_DIR, class_name)
    # Just grab 200 from each class for a fast, statistically significant benchmark
    for img_name in os.listdir(class_dir)[:200]: 
        img_path = os.path.join(class_dir, img_name)
        img = cv2.imread(img_path)
        if img is not None:
            X_test.append(img)
            y_test.append(label)

X_test = np.array(X_test, dtype='float32') / 255.0
y_test = np.array(y_test)

# --- 2. Load the SAAN Model ---
print("\nLoading the custom SAAN model...")
model = tf.keras.models.load_model(
    '../models/saan_best_model.keras',
    custom_objects={'LearnableNoiseGate': LearnableNoiseGate, 'SEBlock': SEBlock}
)

# --- 3. The Noise Injection Engine ---
def apply_snr_noise(image, snr):
    """Simulates deep space noise by injecting Gaussian static based on a target SNR."""
    signal_power = np.mean(image ** 2)
    noise_power = signal_power / snr
    noise = np.random.normal(0, np.sqrt(noise_power), image.shape)
    noisy_image = image + noise
    return np.clip(noisy_image, 0.0, 1.0) # Keep pixel values valid

snr_levels = [10, 5, 1, 0.1]

print("\n🚀 Starting the SNR Degradation Benchmark...")

# Baseline Test (No Noise)
baseline_preds = np.argmax(model.predict(X_test, verbose=0), axis=1)
baseline_acc = accuracy_score(y_test, baseline_preds)
print(f"Clean Baseline (SNR ∞): {baseline_acc * 100:.2f}% Accuracy")

# Stress Tests
for snr in snr_levels:
    X_noisy = np.array([apply_snr_noise(img, snr) for img in X_test])
    preds = np.argmax(model.predict(X_noisy, verbose=0), axis=1)
    acc = accuracy_score(y_test, preds)
    print(f"Stress Test (SNR {snr}): {acc * 100:.2f}% Accuracy")