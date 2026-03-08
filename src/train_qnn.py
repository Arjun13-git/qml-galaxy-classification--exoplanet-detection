import pickle
import numpy as np
import os
from qiskit_algorithms.optimizers import COBYLA
from qiskit_algorithms.utils import algorithm_globals
# Using the updated function calls to avoid warnings
from qiskit.circuit.library import zz_feature_map, real_amplitudes
from qiskit_machine_learning.algorithms import VQC
from qiskit.primitives import StatevectorSampler as Sampler

# Set seed for reproducibility
algorithm_globals.random_seed = 42

# --- 1. Load PCA Data ---
with open('../data/quantum/processed_data.pkl', 'rb') as f:
    data = pickle.load(f)

X_train, y_train = data['X_train'], data['y_train']
X_test, y_test = data['X_test'], data['y_test']

num_classes = 3
y_train_oh = np.eye(num_classes)[y_train]
y_test_oh = np.eye(num_classes)[y_test]

# --- 2. Define Quantum Circuit ---
num_qubits = 6
# Using functions instead of Classes to satisfy Qiskit 2.3+
f_map = zz_feature_map(num_qubits, reps=1)
ansatz_circ = real_amplitudes(num_qubits, reps=1)

# --- 3. Initialize VQC ---
sampler = Sampler()

vqc = VQC(
    feature_map=f_map,
    ansatz=ansatz_circ,
    loss='cross_entropy',
    optimizer=COBYLA(maxiter=50),
    sampler=sampler
)

# --- 4. Training ---
print(f"Starting Quantum Training on 100 samples...")
vqc.fit(X_train[:100], y_train_oh[:100])
print("\nQuantum Training Complete!")

# --- 5. Evaluation ---
score = vqc.score(X_test[:50], y_test_oh[:50])
print(f"Quantum Test Accuracy (Clean Data): {score * 100:.2f}%")

# --- 6. Professional Saving (Weights Only) ---
os.makedirs('../models', exist_ok=True)
# We save the underlying neural network weights to bypass the pickle error
np.save('../models/vqc_weights.npy', vqc.weights)
print("Quantum weights saved to '../models/vqc_weights.npy'")