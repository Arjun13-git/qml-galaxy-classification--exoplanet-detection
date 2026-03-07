# Quantum Machine Learning for Galaxy Classification and Exoplanet Detection 🌌⚛️

**Institution:** Sahyadri College of Engineering & Management, Mangaluru
**Team Number:** CS055
**Team Members:** Ajith Goveas, Arjun Shenoy R, Ashish Shenoy K, Krrish Raj
**Project Guide:** Dr. Mustafa Basthikodi (HoD, Dept. of CSE)

---

## 📖 Project Abstract
This project develops a **Hybrid Quantum Machine Learning (QML) pipeline** integrating Quantum Neural Networks (QNNs) and Quantum Support Vector Machines (QSVMs) to improve sensitivity and robustness in low Signal-to-Noise Ratio (SNR) regimes. We specifically target morphological galaxy classification and exoplanet detection where classical models plateau in performance.

---

## 🚀 Current Progress: SNR-Adaptive Attention Network (SAAN)

Following evaluator feedback, we have moved beyond standard CNNs to develop a custom **State-of-the-Art (SOTA)** classical baseline.

### ✅ Phase 1: 3-Class Balanced Data Pipeline
- **Classes:** Elliptical, Spiral, and Irregular.
- **Augmentation:** Rare Irregular galaxies were oversampled using random rotations and flips to reach a balanced dataset of 6,000 images.
- **Quality:** Switched to lossless `.png` format with anti-aliasing blurring to preserve structural integrity for noise analysis.

### ✅ Phase 2: SAAN Architecture (The Proprietary Baseline)
We engineered the **SNR-Adaptive Attention Network (SAAN)**, featuring:
- **Learnable Noise Gate:** A custom pre-processing layer that dynamically learns to suppress background static.
- **Squeeze-and-Excitation (SE) Blocks:** Channel-wise attention mechanisms that allow the model to "focus" on galactic structures while muting noise-heavy feature maps.
- **Performance:** Achieved a **94.67% Validation Accuracy** on the 3-class problem.

### ✅ Phase 3: Stress Testing & Grad-CAM Visual Proof
We benchmarked the SAAN against simulated deep-space noise (SNR Sweep):
- **SNR 10:** 93.17% Accuracy
- **SNR 1:** 84.33% Accuracy
- **SNR 0.1:** **57.17% Accuracy** 📉 (The Classical Ceiling)

**Visual Proof:** Used **Grad-CAM** to visualize the attention maps. In SNR 0.1 conditions, the heatmap shows the model getting distracted by rainbow static, providing a visual justification for the necessity of Quantum state-space mapping.

---

## ⏳ Next Steps
- **Quantum Integration:** Deploy the 6-qubit QNN to determine if Quantum Hilbert space mapping provides superior noise resilience in the SNR 0.1 regime.