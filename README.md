# Quantum Machine Learning for Galaxy Classification and Exoplanet Detection 🌌⚛️

**Institution:** Sahyadri College of Engineering & Management, Mangaluru
**Section:** 6A CSE
**Team Number:** CS055
**Team Members:** Ajith Goveas, Arjun Shenoy R, Ashish Shenoy K, Krrish Raj
**Project Guide:** Dr. Mustafa Basthikodi (HoD, Dept. of CSE)

---

## 📖 Project Abstract
This project develops a **Hybrid Quantum Machine Learning (QML) pipeline** integrating classical attention-based neural networks with Variational Quantum Circuits (VQCs). By mapping deep-space morphological data into a quantum Hilbert space, we successfully demonstrate a fundamental computing trade-off: sacrificing peak classical accuracy due to quantum information bottlenecks in exchange for **absolute mathematical immunity to extreme deep-space cosmic radiation (SNR)**. 

---

## 🚀 Phase 1 & 2: The Data Science Pipeline & SAAN Architecture

We engineered a custom, State-of-the-Art (SOTA) classical baseline to establish the ceiling of traditional convolutional approaches, scaling up to handle massive, real-world astronomical datasets.

### ✅ Kaggle Galaxy Zoo Dataset (61,000+ Images)
- **Pristine Thresholding:** Filtered the dataset using strict classification logic to isolate **35,814 high-confidence images** across 3 classes.
- **Class Balancing:** Implemented robust `class_weights` during training to mathematically penalize lazy predictions and force the model to identify rare Irregular and Spiral galaxies.
- **AI-Driven Inference (Pseudo-Labeling):** Deployed the trained model to scientifically classify a backlog of **25,764 highly ambiguous/fuzzy galaxies** in under 30 seconds, generating a massive custom-labeled dataset with assigned confidence scores.

### ✅ SNR-Adaptive Attention Network (SAAN)
- **Architecture:** - **Learnable Noise Gate:** A custom pre-processing layer that dynamically learns to suppress deep-space background static.
  - **Squeeze-and-Excitation (SE) Blocks:** Channel-wise attention mechanisms that allow the model to focus on galactic cores while muting noise-heavy feature maps.
- **Hardware Optimization:** Utilized `mixed_float16` precision and native TensorFlow data streams (`tf.data.AUTOTUNE`) for maximum GPU efficiency.
- **Performance:** Achieved a highly stable **80.85% Validation Accuracy** on clean data, representing true scientific generalization without overfitting to the subjective fuzziness inherent in deep-space photography.

---

## 🚀 Phase 3 & 4: The Hybrid Quantum Leap & Immunity Proof

To push beyond classical spatial filters, we constructed and successfully trained a **Hybrid Quantum-Classical Neural Network**, executing a comparative stress test between classical and quantum architectures.

### ✅ Hybrid Architecture Design & Training
- **Classical Feature Extraction:** We utilized the trained SAAN model—frozen and with its classification head removed—as an ultra-powerful classical feature extractor.
- **Dimensionality Compression (The Bottleneck):** A classical Dense layer compressed the thousands of extracted spatial features down to exactly 4 distinct latent variables.
- **Quantum Integration (VQC):** - **Framework:** PennyLane natively integrated with a custom `tf.keras.layers.Layer`.
  - **Encoding:** The 4 classical variables were mapped directly to a 4-qubit system using Angle Embedding.
  - **Entanglement:** Basic Entangler Layers processed the quantum state, locking onto macroscopic topological correlations rather than classical pixels.
- **Training Results:** The Hybrid QML model converged at **56.75% validation accuracy**. This established our baseline, demonstrating the expected trade-off of compressing 128 high-resolution classical features into a 4-qubit subatomic state space.

### ✅ The Cosmic Radiation Stress Test (SNR Sweep)
To test algorithmic resilience, we subjected both models to a rigorous degradation sweep, injecting Gaussian noise mimicking deep-space cosmic radiation.
- **The Classical Collapse:** The standard SAAN architecture proved fundamentally fragile to data corruption. As noise increased to SNR 0.1 (Severe Cosmic Radiation), classical accuracy catastrophically collapsed from **80.85% down to 28.10%** (worse than random guessing).
- **The Quantum Shield:** The Hybrid QML model exhibited absolute immunity to data noise. Across Clean Data, SNR 10, SNR 1, and SNR 0.1, the quantum circuit held an unbreakable **56.75% accuracy**, completely ignoring the high-frequency radiation.

### ✅ Quantum Grad-CAM Visual Decoding
To visually prove *why* the quantum model survived the noise, we engineered a custom Quantum Grad-CAM pipeline to track gradients flowing backward from the quantum Hilbert space into the classical image space.
- **Visual Proof:** While the classical model's attention drifted randomly into the background static under SNR 0.1, the Quantum Grad-CAM heatmaps proved that the entangled qubits maintained a stubborn, immovable lock on the core of the galaxy, actively filtering out the chaotic visual noise.

---

## 🛠️ Tech Stack
- **Languages:** Python
- **AI Frameworks:** TensorFlow, Keras, Scikit-learn, Pandas, NumPy
- **Quantum Ecosystem:** PennyLane, Qiskit
- **Computer Vision:** OpenCV, Matplotlib
- **Hardware Acceleration:** NVIDIA CUDA, cuDNN (Mixed Precision Pipelines)