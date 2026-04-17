# Quantum Machine Learning for Galaxy Classification and Exoplanet Detection 🌌⚛️

**Institution:** Sahyadri College of Engineering & Management, Mangaluru
**Section:** 6A CSE
**Team Number:** CS055
**Team Members:** Ajith Goveas, Arjun Shenoy R, Ashish Shenoy K, Krrish Raj
**Project Guide:** Dr. Mustafa Basthikodi (HoD, Dept. of CSE)

---

## 📖 Project Abstract
This project develops a **Hybrid Quantum Machine Learning (QML) pipeline** integrating classical attention-based neural networks with Variational Quantum Circuits (VQCs). By mapping deep-space morphological data into quantum Hilbert space, we aim to improve sensitivity, structural pattern recognition, and robustness in low Signal-to-Noise Ratio (SNR) regimes for galaxy classification (Elliptical, Spiral, Irregular).

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
- **Hardware Optimization:** Utilized `mixed_float16` precision and native TensorFlow data streams (`tf.data.AUTOTUNE`) for maximum RTX 3050 GPU efficiency.
- **Performance:** Achieved a highly stable **81.28% Validation Accuracy**. This represents true scientific generalization without overfitting to the subjective fuzziness inherent in deep-space photography.

---

## 🚀 Phase 3 & 4: The Hybrid Quantum Leap (PennyLane)

To push beyond classical spatial filters, we constructed a **Hybrid Quantum-Classical Neural Network**.

### ✅ Hybrid Architecture Design
- **Classical Feature Extraction:** We utilize the trained SAAN model—frozen and with its classification head removed—as an ultra-powerful classical feature extractor.
- **Dimensionality Compression:** A classical Dense layer compresses the thousands of extracted spatial features down to exactly 4 distinct latent variables.
- **Quantum Integration (VQC):** - **Framework:** PennyLane (`qml.node`) integrated natively with Keras.
  - **Encoding:** The 4 classical variables are mapped directly to a 4-qubit system using Angle Embedding.
  - **Entanglement:** Basic Entangler Layers process the quantum state, testing if subatomic probability and entanglement can discover correlations invisible to classical math.
  - **Measurement:** Pauli-Z expectation values dictate the final probability distribution across the 3 galaxy classes.

---

## 🛠️ Tech Stack
- **Languages:** Python
- **AI Frameworks:** TensorFlow, Keras, Scikit-learn, Pandas, NumPy
- **Quantum Ecosystem:** PennyLane, Qiskit
- **Hardware Acceleration:** NVIDIA CUDA, cuDNN (Mixed Precision Pipelines)