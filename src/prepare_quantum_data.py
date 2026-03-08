import os
import cv2
import numpy as np
import pickle
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# --- 1. Load Data ---
DATA_DIR = '../data/processed_images'
CLASSES = ['Elliptical', 'Spiral', 'Irregular']

X = []
y = []

print("Gathering images for Quantum dimensionality reduction...")
for label, class_name in enumerate(CLASSES):
    class_dir = os.path.join(DATA_DIR, class_name)
    for img_name in os.listdir(class_dir):
        img_path = os.path.join(class_dir, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) # Quantum works best with grayscale intensity
        if img is not None:
            # Flatten 64x64 into a 4096-length vector
            X.append(img.flatten())
            y.append(label)

X = np.array(X, dtype='float32')
y = np.array(y)

# --- 2. Scaling & PCA ---
print("Standardizing features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# We reduce 4,096 pixels down to 6 core principal components for our 6 qubits
n_components = 6
print(f"Running PCA to extract {n_components} principal components...")
pca = PCA(n_components=n_components)
X_quantum = pca.fit_transform(X_scaled)

# Check how much information we kept
variance_ratio = np.sum(pca.explained_variance_ratio_)
print(f"Retained Variance: {variance_ratio:.2%} (Information kept from original images)")

# --- 3. Save for the QNN ---
os.makedirs('../data/quantum', exist_ok=True)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_quantum, y, test_size=0.2, random_state=42)

quantum_data = {
    'X_train': X_train,
    'X_test': X_test,
    'y_train': y_train,
    'y_test': y_test,
    'pca_object': pca,
    'scaler_object': scaler
}

with open('../data/quantum/processed_data.pkl', 'wb') as f:
    pickle.dump(quantum_data, f)

print("\nQuantum-ready data saved to '../data/quantum/processed_data.pkl'")