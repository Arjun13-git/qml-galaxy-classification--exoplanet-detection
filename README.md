# Quantum Machine Learning for Galaxy Classification and Exoplanet Detection 🌌⚛️


---
## 📖 Project Abstract
[cite_start]Modern astronomy produces massive, high-dimensional datasets from telescopes like SDSS, Galaxy Zoo, Kepler, and TESS[cite: 24, 231, 249]. [cite_start]While classical machine learning methods have achieved strong results in classifying galaxies and detecting exoplanets, they reach a performance ceiling in low signal-to-noise ratio (SNR) regimes (e.g., faint transits or distant galaxies)[cite: 26, 233, 255]. 

[cite_start]This project develops a **Hybrid Quantum Machine Learning (QML) pipeline** integrating Quantum Neural Networks (QNNs) and Quantum Support Vector Machines (QSVMs) to improve sensitivity and robustness against extreme noise[cite: 28, 35, 235]. 

---

## 🚀 Current Progress: The Galaxy Classification Pipeline

[cite_start]We are currently building out the morphological galaxy classification half of the project using the Galaxy Zoo dataset[cite: 418]. 

### ✅ Phase 1: Data Ingestion & Preprocessing
- [cite_start]**Source:** Galaxy Zoo citizen science dataset (~240,000 images)[cite: 418, 443].
- **Labeling:** Filtered crowd-sourced voting fractions for >80% consensus to create hard labels for Elliptical and Spiral galaxies.
- [cite_start]**Processing:** Applied center-cropping ($200 \times 200$) to remove background artifacts, followed by resizing to $64 \times 64 \times 3$ and z-score normalization[cite: 421].
- **Result:** A perfectly balanced, clean dataset of 4,000 images ready for training.

### ✅ Phase 2: Classical ML Baseline (CNN)
- [cite_start]**Architecture:** Built a custom 2D Convolutional Neural Network (CNN) using TensorFlow/Keras[cite: 424, 554].
- [cite_start]**Performance:** Achieved a **92.25% validation accuracy** on the clean dataset, successfully establishing our classical benchmark[cite: 424, 589].

### ✅ Phase 3: Noise Stress Testing & Benchmarking
[cite_start]To justify the quantum advantage, we systematically corrupted our test dataset to simulate deep-space noise and evaluated the CNN's degradation[cite: 381, 574]:
- **SNR 10:** 90.38% Accuracy
- **SNR 5:** 89.25% Accuracy
- **SNR 1:** 87.88% Accuracy
- **SNR 0.1:** **53.50% Accuracy** 📉
- [cite_start]**Conclusion:** As predicted, the classical CNN completely fails (dropping to coin-flip probability) in the extreme low-SNR regime[cite: 258, 388, 591].

---

## ⏳ Next Steps: Phase 4 (Quantum Integration)
We are now moving into the quantum realm to beat the classical failure point. 
1. [cite_start]**Dimensionality Reduction:** Apply PCA ($n=8$) to flatten and compress the images[cite: 422].
2. [cite_start]**Quantum Neural Network:** Build a 6-qubit QNN in Qiskit using an RY-RZ-CNOT variational ansatz and a COBYLA optimizer[cite: 428, 557, 558, 560].
3. [cite_start]**Validation:** Prove the QNN maintains >90% accuracy in the SNR 0.1 regime where the classical model failed[cite: 432, 595].