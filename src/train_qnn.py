import os
import cv2
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Qiskit imports
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.algorithms import NeuralNetworkClassifier
from qiskit_algorithms.optimizers import COBYLA

# --- Configuration ---
DATA_DIR = '../data/processed_images'
NUM_COMPONENTS = 6  # Down to 6 features to match our 6 qubits
SAMPLE_LIMIT = 50   # MINI-BATCH TEST: Only grab 50 per class for now!

# --- 1. Load, Flatten, and Subsample Data ---
print("Loading and flattening images for PCA...")
classes = ['Elliptical', 'Spiral']
X = []
y = []

for label, class_name in enumerate(classes):
    class_dir = os.path.join(DATA_DIR, class_name)
    image_files = os.listdir(class_dir)[:SAMPLE_LIMIT] # Limiting for the quick test
    
    for img_name in image_files:
        img_path = os.path.join(class_dir, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) # Grayscale simplifies PCA
        X.append(img.flatten()) # Flatten 64x64 into a 1D array of 4096 pixels
        y.append(label)

X = np.array(X)
# Qiskit classifiers expect labels as -1 and 1 for binary classification
y = np.array(y) * 2 - 1 

# --- 2. Dimensionality Reduction (PCA) ---
print(f"Applying PCA to reduce from {X.shape[1]} pixels to {NUM_COMPONENTS} components...")
pca = PCA(n_components=NUM_COMPONENTS)
X_pca = pca.fit_transform(X)

# Quantum circuits love data mapped between -1 and 1 (or 0 and pi) for angle rotations
scaler = MinMaxScaler(feature_range=(-1, 1))
X_scaled = scaler.fit_transform(X_pca)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# --- 3. Build the Quantum Neural Network ---
print("Building the 6-qubit QNN...")

# Feature Map: Encodes classical data into quantum states
feature_map = ZZFeatureMap(feature_dimension=NUM_COMPONENTS, reps=1)

# Ansatz: The trainable quantum layers (RY rotations and CNOT entanglements)
ansatz = RealAmplitudes(num_qubits=NUM_COMPONENTS, reps=2)

# Combine them into a QNN
qnn = EstimatorQNN(
    circuit=feature_map.compose(ansatz),
    input_params=feature_map.parameters,
    weight_params=ansatz.parameters
)

# Callback function to print progress during training
def callback_graph(weights, obj_func_eval):
    print(f"COBYLA Iteration | Objective Function Value: {obj_func_eval:.4f}")

# Wrap it in a Classifier
classifier = NeuralNetworkClassifier(
    neural_network=qnn,
    optimizer=COBYLA(maxiter=30), # Only 30 iterations for the fast test
    callback=callback_graph
)

# --- 4. Train and Evaluate ---
print("\nStarting Quantum Training (This might take a few minutes)...")
classifier.fit(X_train, y_train)

print("\nEvaluating Quantum Model...")
y_pred = classifier.predict(X_test)

# Convert predictions back to standard 0 and 1 for accuracy score
y_pred_binary = np.where(y_pred > 0, 1, 0)
y_test_binary = np.where(y_test > 0, 1, 0)

acc = accuracy_score(y_test_binary, y_pred_binary)
print(f"\nMini-Batch Quantum Accuracy: {acc * 100:.2f}%")