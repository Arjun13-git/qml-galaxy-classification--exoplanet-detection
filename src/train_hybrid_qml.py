import os
import pandas as pd
import numpy as np
import tensorflow as tf
import pennylane as qml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from build_saan import LearnableNoiseGate, SEBlock

print("🔍 Checking Hardware...")
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    print(f"✅ GPU Detected: {physical_devices[0]}")
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# --- 1. Load a SMALL Sample of Data (Quantum is heavy!) ---
CSV_PATH = '../master_labels_60k.csv'
IMAGES_DIR = r"C:\galaxy_datasets\images_training_rev1"

print("📂 Loading data...")
df = pd.read_csv(CSV_PATH)
df['filepath'] = IMAGES_DIR + '/' + df['filename']
df = df[df['filepath'].apply(os.path.exists)]

# ⚠️ WE ONLY TAKE 1,500 IMAGES FOR THE QUANTUM SIMULATOR ⚠️
df = df.sample(n=1500, random_state=42)

encoder = LabelEncoder()
df['label_encoded'] = encoder.fit_transform(df['Label'])
num_classes = len(encoder.classes_)

train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['label_encoded'], random_state=42)

# Data processing
IMG_SIZE = (64, 64)
BATCH_SIZE = 16 # Much smaller batch size for quantum!

def process_image(file_path, label):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = img / 255.0
    label = tf.one_hot(label, depth=num_classes)
    return img, label

train_dataset = tf.data.Dataset.from_tensor_slices((train_df['filepath'].values, train_df['label_encoded'].values)).map(process_image).batch(BATCH_SIZE)
val_dataset = tf.data.Dataset.from_tensor_slices((val_df['filepath'].values, val_df['label_encoded'].values)).map(process_image).batch(BATCH_SIZE)

# --- 2. Load the Classical Model & "Chop off the Head" ---
print("🧠 Loading Classical SAAN...")
base_model = tf.keras.models.load_model(
    '../models/saan_best_model_60k.keras', 
    custom_objects={'LearnableNoiseGate': LearnableNoiseGate, 'SEBlock': SEBlock}
)

# Freeze the classical layers (we already trained them perfectly!)
for layer in base_model.layers:
    layer.trainable = False

# Extract features up to the Flatten layer
feature_extractor = tf.keras.Model(inputs=base_model.input, outputs=base_model.get_layer('flatten').output)

# --- 3. Build the Quantum Circuit ---
n_qubits = 4
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface="tf")
def qnode(inputs, weights):
    # Embed the classical features into the quantum state (Angle Embedding)
    qml.AngleEmbedding(inputs, wires=range(n_qubits))
    
    # Apply entangling layers (The Quantum "Hidden Layers")
    qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
    
    # Measure the expectation value of the first 3 qubits (for our 3 classes)
    return [qml.expval(qml.PauliZ(wires=i)) for i in range(num_classes)]

# Define the Quantum Keras Layer
weight_shapes = {"weights": (3, n_qubits)} # 3 layers of quantum entanglement
qlayer = qml.qnn.KerasLayer(qnode, weight_shapes, output_dim=num_classes)

# --- 4. Assemble the Hybrid Model ---
print("🌌 Assembling Hybrid Quantum-Classical Architecture...")
inputs = tf.keras.Input(shape=(64, 64, 3))
x = feature_extractor(inputs)

# Compress the thousands of classical features down to 4 variables for the 4 qubits
x = tf.keras.layers.Dense(n_qubits, activation='tanh')(x) 

# Pass the 4 variables into the Quantum Realm
x = qlayer(x) 

# Convert the quantum measurements (-1 to 1) into standard probabilities (0 to 1)
outputs = tf.keras.layers.Activation('softmax')(x)

hybrid_model = tf.keras.Model(inputs=inputs, outputs=outputs)

hybrid_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), # Higher learning rate for quantum layer
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# --- 5. Train the Hybrid Model ---
print("🚀 Launching Quantum Training Loop...")
hybrid_model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=5
)

print("✅ HYBRID QML TRAINING COMPLETE.")