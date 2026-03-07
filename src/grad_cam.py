import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from build_saan import LearnableNoiseGate, SEBlock

# --- 1. Load the Model ---
print("Loading SAAN Model...")
model = tf.keras.models.load_model(
    '../models/saan_best_model.keras',
    custom_objects={'LearnableNoiseGate': LearnableNoiseGate, 'SEBlock': SEBlock}
)

# We target the last Convolutional layer before the Flatten step to see the final spatial map
TARGET_LAYER = 'conv2d_2' 

# --- 2. The Grad-CAM Engine ---
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # Create a model that maps the input image to the activations of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        model.inputs, 
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Record operations for automatic differentiation
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # Calculate the gradients of the top predicted class with respect to the output feature map
    grads = tape.gradient(class_channel, last_conv_layer_output)
    
    # Pool the gradients over the spatial dimensions
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Multiply each channel in the feature map by "how important" it is
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normalize the heatmap between 0 and 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def superimpose_heatmap(img, heatmap, alpha=0.4):
    # Resize heatmap to match the original image size
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    # Convert heatmap to RGB using a colormap
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Superimpose the heatmap on original image
    superimposed_img = heatmap * alpha + (img * 255.0)
    superimposed_img = np.clip(superimposed_img, 0, 255).astype('uint8')
    return superimposed_img

# --- 3. Noise Function ---
def apply_snr_noise(image, snr):
    signal_power = np.mean(image ** 2)
    noise_power = signal_power / snr
    noise = np.random.normal(0, np.sqrt(noise_power), image.shape)
    noisy_image = image + noise
    return np.clip(noisy_image, 0.0, 1.0)

# --- 4. Execution & Plotting ---
print("Fetching a sample Spiral Galaxy...")
# Let's grab a random Spiral galaxy from your processed data
sample_path = os.listdir('../data/processed_images/Spiral')[10]
img_path = os.path.join('../data/processed_images/Spiral', sample_path)

# Prepare Clean Image
clean_img = cv2.imread(img_path)
clean_img = cv2.cvtColor(clean_img, cv2.COLOR_BGR2RGB) # Convert to RGB for matplotlib
clean_img_normalized = np.expand_dims(clean_img / 255.0, axis=0)

# Prepare Noisy Image (SNR 0.1)
noisy_img_normalized = apply_snr_noise(clean_img_normalized[0], snr=0.1)
noisy_img_normalized = np.expand_dims(noisy_img_normalized, axis=0)

print("Generating Heatmaps...")
# Generate heatmaps
clean_heatmap = make_gradcam_heatmap(clean_img_normalized, model, TARGET_LAYER)
noisy_heatmap = make_gradcam_heatmap(noisy_img_normalized, model, TARGET_LAYER)

# Superimpose
clean_result = superimpose_heatmap(clean_img_normalized[0], clean_heatmap)
noisy_result = superimpose_heatmap(noisy_img_normalized[0], noisy_heatmap)

# Plotting the results
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(clean_result)
axes[0].set_title('Clean Image (SAAN focuses on the core)')
axes[0].axis('off')

axes[1].imshow(noisy_result)
axes[1].set_title('SNR 0.1 (SAAN gets distracted by static)')
axes[1].axis('off')

plt.tight_layout()
plt.show()