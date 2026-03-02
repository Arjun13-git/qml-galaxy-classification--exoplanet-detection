import os
import numpy as np
import tensorflow as tf
import pandas as pd

# --- Configuration ---
DATA_DIR = '../data/processed_images'
MODEL_PATH = '../models/galaxy_cnn_baseline.keras'
BATCH_SIZE = 32
IMG_SIZE = (64, 64)
SNR_LEVELS = [10, 5, 1, 0.1]

# --- 1. Load Model and Validation Data ---
print("Loading model and validation data...")
model = tf.keras.models.load_model(MODEL_PATH)

val_dataset = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True # Keep order so we can calculate exact metrics
)

# Extract images and true labels into numpy arrays for easier manipulation
val_images = []
val_labels = []
for images, labels in val_dataset:
    val_images.append(images.numpy())
    val_labels.append(labels.numpy())

val_images = np.concatenate(val_images)
val_labels = np.concatenate(val_labels)

# Normalize images to [0, 1] just like the Rescaling layer does internally
# (We do this here so our noise math scales correctly)
val_images = val_images / 255.0 

# --- 2. Noise Injection Function ---
def inject_gaussian_noise(images, target_snr):
    """
    Injects Gaussian noise based on the target SNR.
    """
    noisy_images = np.zeros_like(images)
    
    for i in range(images.shape[0]):
        img = images[i]
        # Calculate signal power (variance of the image pixels)
        signal_power = np.var(img)
        
        # Calculate required noise variance
        noise_variance = signal_power / target_snr
        
        # Generate Gaussian noise
        noise = np.random.normal(0, np.sqrt(noise_variance), img.shape)
        
        # Add noise and clip values to keep them in valid pixel range [0, 1]
        noisy_img = np.clip(img + noise, 0.0, 1.0)
        noisy_images[i] = noisy_img
        
    return noisy_images

# --- 3. Run the SNR Sweep ---
results = []

print("\nRunning SNR Benchmark Sweep...")
for snr in SNR_LEVELS:
    print(f"Testing SNR = {snr}...")
    
    # Inject noise
    noisy_val_images = inject_gaussian_noise(val_images, snr)
    
    # Evaluate model
    # (We multiply by 255 because our model's first layer is a Rescaling(1./255) layer, 
    # so we need to feed it data in the original [0, 255] format)
    loss, accuracy = model.evaluate(noisy_val_images * 255.0, val_labels, verbose=0)
    
    results.append({
        'Model': 'Classical CNN',
        'Target_SNR': snr,
        'Accuracy': round(accuracy * 100, 2)
    })

# --- 4. Display Results ---
df_results = pd.DataFrame(results)
print("\n--- Benchmarking Results ---")
print(df_results.to_markdown(index=False))