import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import pennylane as qml
import matplotlib.pyplot as plt
from build_saan import LearnableNoiseGate, SEBlock

print("🔍 Initializing Quantum Grad-CAM Visualizer...")
# Lock to 32-bit for Quantum Gradient Calculation
tf.keras.mixed_precision.set_global_policy('float32') 
tf.config.run_functions_eagerly(True)

physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# --- 1. Rebuild the Quantum Blueprints ---
n_qubits = 4
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface="tf", diff_method="backprop")
def qnode(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(n_qubits))
    qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(wires=i)) for i in range(3)]

class QuantumKerasLayer(tf.keras.layers.Layer):
    def __init__(self, n_qubits, n_layers, **kwargs):
        # 🛡️ THE HOLY GRAIL: This physically blocks AutoGraph from touching this layer!
         
        super().__init__(**kwargs)
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.q_weights = self.add_weight(
            shape=(n_layers, n_qubits),
            initializer="uniform",
            trainable=True,
            name="q_weights",
            dtype=tf.float32
        )
        
    @tf.autograph.experimental.do_not_convert
    def call(self, inputs, training=None, **kwargs):
        # Because Grad-CAM strictly uses a batch size of 1, 
        # we bypass all loops and just process the single image directly!
        single_input = inputs[0]
        q_out = tf.cast(qnode(single_input, self.q_weights), tf.float32)
        return tf.expand_dims(q_out, 0)
        
    def compute_output_shape(self, input_shape):
        return tf.TensorShape([input_shape[0], 3])
        
    def get_config(self):
        config = super().get_config()
        config.update({
            "n_qubits": self.n_qubits,
            "n_layers": self.n_layers,
        })
        return config

# --- 2. Load the Quantum Champion ---
print("🧠 Awakening SAAN Hybrid Quantum Model...")
MODEL_PATH = '../models/saan_hybrid_qml_best.keras'
model = tf.keras.models.load_model(
    MODEL_PATH, 
    custom_objects={
        'LearnableNoiseGate': LearnableNoiseGate, 
        'SEBlock': SEBlock,
        'QuantumKerasLayer': QuantumKerasLayer
    }
)

# --- 3. The Noise Generator ---
def add_gaussian_noise(image, snr):
    signal_power = tf.math.reduce_variance(image)
    noise_power = signal_power / snr
    noise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=tf.math.sqrt(noise_power), dtype=tf.float32)
    return tf.clip_by_value(image + noise, 0.0, 1.0)

# --- 4. The Quantum Grad-CAM Engine ---
def make_gradcam_heatmap(img_array, model):
    # Target the SE Block (Layer Index 3) just like we did for the classical fix
    target_layer_output = model.layers[3].output
    
    grad_model = tf.keras.models.Model(
        [model.inputs], 
        [target_layer_output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        top_pred_index = tf.argmax(preds[0])
        top_class_channel = preds[:, top_pred_index]

    # Calculate gradients flowing back from the quantum layer
    grads = tape.gradient(top_class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def overlay_heatmap(img, heatmap, alpha=0.4):
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    img_uint8 = np.uint8(255 * img)
    superimposed_img = cv2.addWeighted(heatmap, alpha, img_uint8, 1 - alpha, 0)
    return superimposed_img

# --- 5. The Visual Test Flight ---
# Make sure this points to the exact same image you used for the classical test!
SAMPLE_IMAGE_PATH = r"C:\galaxy_datasets\images_training_rev1\102243.jpg" 

print("🎨 Generating Quantum Heatmaps...")
raw_img = tf.io.read_file(SAMPLE_IMAGE_PATH)
raw_img = tf.image.decode_jpeg(raw_img, channels=3)
raw_img = tf.image.resize(raw_img, (64, 64)) / 255.0

img_clean = raw_img
img_snr10 = add_gaussian_noise(raw_img, 10.0)
img_snr1 = add_gaussian_noise(raw_img, 1.0)
img_snr01 = add_gaussian_noise(raw_img, 0.1)

images = [img_clean, img_snr10, img_snr1, img_snr01]
titles = ["Clean (56.7%)", "SNR 10 (56.7%)", "SNR 1 (56.7%)", "SNR 0.1 (56.7%)"]
class_names = ['Elliptical', 'Irregular', 'Spiral']

plt.figure(figsize=(16, 4))

for i, img_tensor in enumerate(images):
    img_batch = tf.expand_dims(img_tensor, axis=0)
    
    # Get the Quantum Prediction!
    preds = model(img_batch, training=False)
    pred_idx = np.argmax(preds[0])
    confidence = preds[0][pred_idx] * 100
    pred_label = class_names[pred_idx]
    
    # Generate Heatmap
    heatmap = make_gradcam_heatmap(img_batch, model)
    overlaid = overlay_heatmap(img_tensor.numpy(), heatmap)
    
    plt.subplot(1, 4, i + 1)
    plt.imshow(cv2.cvtColor(overlaid, cv2.COLOR_BGR2RGB))
    
    dynamic_title = f"{titles[i]}\nQuantum Guess: {pred_label} ({confidence:.1f}%)"
    plt.title(dynamic_title, fontsize=12, fontweight='bold')
    plt.axis('off')

plt.tight_layout()
plt.savefig("../gradcam_qml_immunity_proof.png", dpi=300)
print("✅ SUCCESS! Quantum Heatmap saved to QML_Galaxy_Project/gradcam_qml_immunity_proof.png")
plt.show()