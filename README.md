# Quantum Machine Learning for Galaxy Classification and Exoplanet Detection 🌌⚛️

**Institution:** Sahyadri College of Engineering & Management, Mangaluru  
**Team Number:** CS055  
**Team Members:** Ajith Goveas, Arjun Shenoy R, Ashish Shenoy K, Krrish Raj  
**Project Guide:** Dr. Mustafa Basthikodi (HoD, Dept of CSE)  

---

## 📖 Project Abstract
Modern astronomy produces massive, high-dimensional datasets from telescopes like SDSS, Galaxy Zoo, Kepler, and TESS. While classical machine learning methods have achieved strong results in classifying galaxies and detecting exoplanets, they reach a performance ceiling in low signal-to-noise ratio (SNR) regimes (e.g., faint transits or distant galaxies). 

This project develops a **Hybrid Quantum Machine Learning (QML) pipeline** integrating Quantum Neural Networks (QNNs) and Quantum Support Vector Machines (QSVMs) to improve sensitivity and robustness against extreme noise. 

---

## 🚀 Current Progress: The Galaxy Classification Pipeline

We are currently building out the morphological galaxy classification half of the project using the Galaxy Zoo dataset. 

### ✅ Phase 1: Data Ingestion & Preprocessing
- **Source:** Galaxy Zoo citizen science dataset (~240,000 images).
- **Labeling:** Filtered crowd-sourced voting fractions for >80% consensus to create hard labels for Elliptical and Spiral galaxies.
- **Processing:** Applied center-cropping (200x200) to remove background artifacts, followed by resizing to 64x64x3 and z-score normalization.
- **Result:** A perfectly balanced, clean dataset of 4,000 images ready for training.

### ✅ Phase 2: Classical ML Baseline (CNN)
- **Architecture:** Built a custom 2D Convolutional Neural Network (CNN) using TensorFlow/Keras.
- **Performance:** Achieved a **92.25% validation accuracy** on the clean dataset, successfully establishing our classical benchmark.

### ✅ Phase 3: Noise Stress Testing & Benchmarking
To justify the quantum advantage, we systematically corrupted our test dataset to simulate deep-space noise and evaluated the CNN's degradation:
- **SNR 10:** 90.38% Accuracy
- **SNR 5:** 89.25% Accuracy
- **SNR 1:** 87.88% Accuracy
- **SNR 0.1:** **53.50% Accuracy** 📉
- **Conclusion:** As predicted, the classical CNN completely fails (dropping to coin-flip probability) in the extreme low-SNR regime.

### ✅ Phase 4: Quantum Neural Network (QNN) - Proof of Concept
- **Dimensionality Reduction:** Flattened the 64x64 images and applied Principal Component Analysis (PCA) to compress 4,096 pixels down to 6 core features.
- **Quantum Architecture:** Constructed a 6-qubit QNN using Qiskit, utilizing an RY-RZ-CNOT variational ansatz and the COBYLA optimizer.
- **Validation:** Executed a mini-batch stress test (100 images, 30 iterations) on the quantum simulator. The circuit successfully compiled, entangled, and optimized, verifying the hybrid quantum-classical pipeline's plumbing and establishing a baseline for full-scale training.

---

## ⏳ Next Steps
1. **Full-Scale QNN Training:** Scale the Qiskit QNN to process the entire 4,000-image dataset and evaluate its resilience in the SNR 0.1 regime.
2. **QSVM Implementation:** Build and benchmark a Quantum Support Vector Machine using PennyLane.
3. **Exoplanet Integration:** Apply the pipeline to Kepler/TESS transit photometry data using LSTM and QSVM models.