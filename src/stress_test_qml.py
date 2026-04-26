import os
import pandas as pd
import numpy as np
import tensorflow as tf
import pennylane as qml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from build_saan import LearnableNoiseGate, SEBlock

print("⚛️ Initializing Quantum Stress Test Environment...")
tf.keras.mixed_precision.set_global_policy('float32') # Lock to 32-bit for Quantum

physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# --- 1. Rebuild the Quantum Blueprints for Loading ---
n_qubits = 4

dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface="tf", diff_method="backprop")
def qnode(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(n_qubits))
    qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(wires=i)) for i in range(3)]

class QuantumKerasLayer(tf.keras.layers.Layer):
    def __init__(self, n_qubits, n_layers, **kwargs):
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
        
    def call(self, inputs):
        return tf.map_fn(
            lambda x: tf.cast(tf.convert_to_tensor(qnode(x, self.q_weights)), tf.float32), 
            inputs, 
            fn_output_signature=tf.float32
        )
        
    def get_config(self):
        config = super().get_config()
        config.update({
            "n_qubits": self.n_qubits,
            "n_layers": self.n_layers,
        })
        return config

# --- 2. Load the Hybrid Model ---
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

# --- 3. Load the Data (Strict Validation Set) ---
CSV_PATH = '../master_labels_60k.csv'
IMAGES_DIR = r"C:\galaxy_datasets\images_training_rev1"

df = pd.read_csv(CSV_PATH)
df['filepath'] = IMAGES_DIR + '/' + df['filename']
df = df[df['filepath'].apply(os.path.exists)]

encoder = LabelEncoder()
df['label_encoded'] = encoder.fit_transform(df['Label'])

# Grab the exact same 2000 images we used for the classical test
_, val_df = train_test_split(df, test_size=0.2, stratify=df['label_encoded'], random_state=42)
test_df = val_df.sample(n=2000, random_state=42)
num_classes = len(encoder.classes_)

# --- 4. The Deep Space Noise Generator ---
def add_gaussian_noise(image, snr):
    signal_power = tf.math.reduce_variance(image)
    noise_power = signal_power / snr
    noise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=tf.math.sqrt(noise_power), dtype=tf.float32)
    return tf.clip_by_value(image + noise, 0.0, 1.0)

# --- 5. Testing Pipeline ---
def evaluate_at_snr(dataframe, snr_level, snr_name):
    print(f"\n🌌 Commencing {snr_name} Sweep...")
    
    def process_and_corrupt(file_path, label):
        img = tf.io.read_file(file_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, (64, 64))
        
        if snr_level is not None:
            img = add_gaussian_noise(img, snr_level)
            
        label = tf.one_hot(label, depth=num_classes)
        return img, label

    test_dataset = tf.data.Dataset.from_tensor_slices((dataframe['filepath'].values, dataframe['label_encoded'].values))
    test_dataset = test_dataset.map(process_and_corrupt, num_parallel_calls=tf.data.AUTOTUNE).batch(64)
    
    results = model.evaluate(test_dataset, verbose=0)
    accuracy = results[1] * 100
    print(f"   => Accuracy: {accuracy:.2f}%")
    return accuracy

# --- 6. The Grand Evaluation ---
print("\n" + "="*50)
print("🚀 LAUNCHING QUANTUM SNR STRESS TEST (2,000 Images)")
print("="*50)

acc_clean = evaluate_at_snr(test_df, snr_level=None, snr_name="CLEAN (No Noise)")
acc_snr10 = evaluate_at_snr(test_df, snr_level=10.0, snr_name="SNR 10 (Light Static)")
acc_snr1 = evaluate_at_snr(test_df, snr_level=1.0, snr_name="SNR 1 (Heavy Static)")
acc_snr01 = evaluate_at_snr(test_df, snr_level=0.1, snr_name="SNR 0.1 (Severe Cosmic Radiation)")

print("\n" + "="*50)
print("📉 QUANTUM STRESS TEST RESULTS")
print("="*50)
print(f"Clean Data:    {acc_clean:.2f}%")
print(f"SNR 10:        {acc_snr10:.2f}%")
print(f"SNR 1:         {acc_snr1:.2f}%")
print(f"SNR 0.1:       {acc_snr01:.2f}%")
print("="*50)