import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from build_saan import LearnableNoiseGate, SEBlock

print("🔍 Initializing Hardware for Grad-CAM...")
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# --- 1. Load the Champion Model ---
print("🧠 Awakening SAAN-DenseNet121...")
MODEL_PATH = '../models/saan_densenet_best.keras'
model = tf.keras.models.load_model(
    MODEL_PATH, 
    custom_objects={'LearnableNoiseGate': LearnableNoiseGate, 'SEBlock': SEBlock}
)

# --- 2. The Noise Generator (Same as Stress Test) ---
def add_gaussian_noise(image, snr):
    signal_power = tf.math.reduce_variance(image)
    noise_power = signal_power / snr
    noise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=tf.math.sqrt(noise_power), dtype=tf.float32)
    return tf.clip_by_value(image + noise, 0.0, 1.0)

# --- 3. The Grad-CAM Engine ---
def make_gradcam_heatmap(img_array, model):
    # Target the Squeeze-and-Excitation Block directly (It is Layer Index 3)
    # This bypasses the nested graph bug and shows your custom attention!
    target_layer_output = model.layers[3].output
    
    # Create the gradient tracking model
    grad_model = tf.keras.models.Model(
        [model.inputs], 
        [target_layer_output, model.output]
    )

    # Compute the gradient of the top predicted class
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        top_pred_index = tf.argmax(preds[0])
        top_class_channel = preds[:, top_pred_index]

    # Calculate exactly which pixels triggered the decision
    grads = tape.gradient(top_class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normalize the heatmap
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def overlay_heatmap(img, heatmap, alpha=0.4):
    # Resize heatmap to match image dimensions
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    
    # Convert to RGB colormap
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Convert image back to 0-255 scale for visualization
    img_uint8 = np.uint8(255 * img)
    
    # Superimpose
    superimposed_img = cv2.addWeighted(heatmap, alpha, img_uint8, 1 - alpha, 0)
    return superimposed_img

# --- 4. The Visual Test Flight ---
# Pick a random clean image from your dataset directory
SAMPLE_IMAGE_PATH = r"C:\galaxy_datasets\images_training_rev1\100023.jpg" # Change this number to any valid image file!

print("🎨 Generating Heatmaps...")
raw_img = tf.io.read_file(SAMPLE_IMAGE_PATH)
raw_img = tf.image.decode_jpeg(raw_img, channels=3)
raw_img = tf.image.resize(raw_img, (64, 64)) / 255.0

# Generate the noise variants
img_clean = raw_img
img_snr10 = add_gaussian_noise(raw_img, 10.0)
img_snr1 = add_gaussian_noise(raw_img, 1.0)
img_snr01 = add_gaussian_noise(raw_img, 0.1)

images = [img_clean, img_snr10, img_snr1, img_snr01]
titles = ["Clean (80.8%)", "SNR 10 (78.8%)", "SNR 1 (48.4%)", "SNR 0.1 (28.1%)"]

plt.figure(figsize=(16, 4))

class_names = ['Elliptical', 'Irregular', 'Spiral']
for i, img_tensor in enumerate(images):
    # Expand dimensions to create a "batch" of 1 for the model
    img_batch = tf.expand_dims(img_tensor, axis=0)
    
    preds = model.predict(img_batch, verbose=0)
    pred_idx = np.argmax(preds[0])
    confidence = preds[0][pred_idx] * 100
    pred_label = class_names[pred_idx]
    # Generate Heatmap
    heatmap = make_gradcam_heatmap(img_batch, model)
    
    # Overlay on original image
    overlaid = overlay_heatmap(img_tensor.numpy(), heatmap)
    
    # Plotting
    plt.subplot(1, 4, i + 1)
    # OpenCV uses BGR, Matplotlib uses RGB, so we swap color channels for correct display
    plt.imshow(cv2.cvtColor(overlaid, cv2.COLOR_BGR2RGB))
    dynamic_title = f"{titles[i]}\nGuess: {pred_label} ({confidence:.1f}%)"
    plt.title(dynamic_title, fontsize=12, fontweight='bold')
    plt.axis('off')

plt.tight_layout()
plt.savefig("../gradcam_collapse_proof_with_preds.png", dpi=300)
print("✅ SUCCESS! Heatmap with predictions saved to QML_Galaxy_Project/gradcam_collapse_proof.png")
plt.show()