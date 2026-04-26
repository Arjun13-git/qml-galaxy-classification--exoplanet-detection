import os
import pandas as pd
import numpy as np
import tensorflow as tf
import pennylane as qml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from build_saan import LearnableNoiseGate, SEBlock
import sys

import sys

# 👇 ADD THE LIVE LOGGER 👇
class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    def flush(self):
        pass

sys.stdout = Logger("../logs_hybrid_qml_2026.txt")
print("📝 Live logging activated! Saving everything to ../logs_hybrid_qml_2026.txt")
# 👆 ---------------------- 👆

print("🌌 Initializing Quantum-Classical Hardware...")
# Disable mixed precision for the Quantum layer (PennyLane prefers float32)
tf.keras.mixed_precision.set_global_policy('float32') 

physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print(f"✅ Classical GPU Detected: {physical_devices[0]}")

# --- 1. Load Data Pipeline (Clean Data) ---
CSV_PATH = '../master_labels_60k.csv'
IMAGES_DIR = r"C:\galaxy_datasets\images_training_rev1"

df = pd.read_csv(CSV_PATH)
df['filepath'] = IMAGES_DIR + '/' + df['filename']
df = df[df['filepath'].apply(os.path.exists)]

encoder = LabelEncoder()
df['label_encoded'] = encoder.fit_transform(df['Label'])
train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['label_encoded'], random_state=42)

def process_image(file_path, label):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (64, 64))
    img = tf.image.random_flip_left_right(img)
    label = tf.one_hot(label, depth=3)
    return img, label # No manual / 255.0 because we are using DenseNet SOTA inside

train_dataset = tf.data.Dataset.from_tensor_slices((train_df['filepath'].values, train_df['label_encoded'].values)).map(process_image, num_parallel_calls=tf.data.AUTOTUNE).batch(64).prefetch(tf.data.AUTOTUNE)
val_dataset = tf.data.Dataset.from_tensor_slices((val_df['filepath'].values, val_df['label_encoded'].values)).map(process_image, num_parallel_calls=tf.data.AUTOTUNE).batch(64)

# --- 2. The Quantum Node (VQC) ---
print("⚛️ Forging the 4-Qubit Quantum Circuit...")
n_qubits = 4
n_layers = 2

# We use the TensorFlow interface for PennyLane
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface="tf", diff_method="backprop")
def qnode(inputs, weights):
    # 1. Embed classical data into quantum rotations
    qml.AngleEmbedding(inputs, wires=range(n_qubits))
    
    # 2. Entangle the qubits to find hidden cosmic correlations
    qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
    
    # 3. Measure the first 3 qubits (matching our 3 galaxy classes)
    return [qml.expval(qml.PauliZ(wires=i)) for i in range(3)]

# Define the shape of the quantum weights so Keras knows how to train it
# 👇 THE FIX: Custom Keras Layer to replace the deleted PennyLane one 👇
# 👇 THE UPDATED 32-BIT QUANTUM LAYER 👇
# 👇 THE FULLY SERIALIZABLE 32-BIT QUANTUM LAYER 👇
class QuantumKerasLayer(tf.keras.layers.Layer):
    def __init__(self, n_qubits, n_layers, **kwargs):
        super().__init__(**kwargs)
        self.n_qubits = n_qubits # Save for serialization
        self.n_layers = n_layers # Save for serialization
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
        
    # 🛡️ THE FIX: Tells Keras exactly how to save this custom layer to the hard drive
    def get_config(self):
        config = super().get_config()
        config.update({
            "n_qubits": self.n_qubits,
            "n_layers": self.n_layers,
        })
        return config

qlayer = QuantumKerasLayer(n_qubits, n_layers, name="Quantum_Entanglement_Layer")
# 👆 ---------------------------------------------------- 👆

# --- 3. Digital Surgery: Building the Hybrid ---
print("🧠 Splicing DenseNet121 into Quantum Processor...")

# Load the Classical Champion
old_model = tf.keras.models.load_model(
    '../models/saan_densenet_best.keras', 
    custom_objects={'LearnableNoiseGate': LearnableNoiseGate, 'SEBlock': SEBlock}
)

# Chop off the final classification head (Extract from the Dense(128) layer)
# We freeze the entire classical network so ONLY the quantum circuit trains
extractor = tf.keras.models.Model(inputs=old_model.inputs, outputs=old_model.layers[-3].output)
extractor.trainable = False 

# Attach the Quantum Brain
x = extractor.output
x = tf.keras.layers.Dense(4, activation='tanh', name="Classical_Bottleneck")(x) # Compress to 4 signals (-1 to 1)
x = qlayer(x) # Pipe into Hilbert Space
outputs = tf.keras.layers.Activation('softmax', name="Final_Measurement")(x) # Collapse to 3 probabilities

hybrid_model = tf.keras.models.Model(inputs=extractor.inputs, outputs=outputs, name="SAAN_Hybrid_Quantum")

# --- 4. Training the Quantum Machine ---
hybrid_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint('../models/saan_hybrid_qml_best.keras', save_best_only=True)
]

print("🚀 INITIATING QUANTUM TRAINING (Expect longer epoch times!)...")
hybrid_model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=10, 
    callbacks=callbacks,
    verbose=2
)
print("✅ QUANTUM HYBRID MODEL SECURED.")