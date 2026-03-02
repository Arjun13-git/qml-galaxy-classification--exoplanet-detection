# Quantum Machine Learning for Galaxy Classification and Exoplanet Detection 🌌⚛️


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
- **Processing:** Applied center-cropping ($200 \times 200$) to remove background artifacts, followed by resizing to $64 \times 64 \times 3$ and z-score normalization.
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

---

## ⏳ Next Steps: Phase 4 (Quantum Integration)
We are now moving into the quantum realm to beat the classical failure point. 
1. **Dimensionality Reduction:** Apply PCA ($n=8$) to flatten and compress the images.
2. **Quantum Neural Network:** Build a 6-qubit QNN in Qiskit using an RY-RZ-CNOT variational ansatz and a COBYLA optimizer.
3. **Validation:** Prove the QNN maintains >90% accuracy in the SNR 0.1 regime where the classical model failed.