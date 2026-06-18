# 🩺 Efficient and Reliable Skin Lesion Classification for Mobile Deployment using Knowledge Distillation

## Overview

Skin cancer is one of the most common forms of cancer worldwide, and early detection plays a critical role in improving patient outcomes. Deep learning models have achieved remarkable performance in skin lesion classification, but many high-performing architectures are too computationally expensive for deployment on resource-constrained devices such as smartphones and handheld diagnostic systems.

This project investigates whether knowledge distillation can be used to transfer diagnostic knowledge from a large teacher model to lightweight student models while preserving accuracy, interpretability, reliability, and deployment efficiency.

---

## 🌐 Live Application 

Frontend: https://skin-lesion-distillation.vercel.app/

Backend API: https://skin-lesion-api-p7ga.onrender.com/

API Documentation:
https://skin-lesion-api-p7ga.onrender.com/docs

### 🚀 Current Deployment Features

✅ Real-time skin lesion classification

✅ Confidence score estimation

✅ Cloud-hosted inference API

❌ Grad-CAM visualization (disabled in public deployment due to cloud memory constraints)

✅ Full Grad-CAM support available in the research version

## 🚀 Key Contributions

* Knowledge distillation from EfficientNet-B3 to lightweight MobileNet architectures.
* Deployment-oriented evaluation including latency, VRAM usage, and model size.
* Calibration analysis using Expected Calibration Error (ECE) and reliability diagrams.
* Interpretability assessment using Grad-CAM visualizations.
* Temperature ablation study to determine the optimal distillation configuration.
* External validation on the PH2 skin lesion dataset.

---

## 📊 Dataset

### HAM10000

* Total Classes: 7
* Classes:

  * akiec
  * bcc
  * bkl
  * df
  * mel
  * nv
  * vasc
* Image Resolution: 300 × 300

### PH2 (External Validation)

* Total Images: 200
* Nevus: 160
* Melanoma: 40

Datasets are not included in this repository and must be downloaded separately.

---

## 🧠 Model Architecture

### Teacher Model

* EfficientNet-B3
* ImageNet pretrained
* Fine-tuned on HAM10000

### Student Models

* MobileNetV2
* MobileNetV3-Small

Knowledge transfer is performed using a combination of:

* Cross Entropy Loss
* KL Divergence Distillation Loss

---

## 🏆 Best Student Model (MobileNetV2, T = 2)

| Metric            | Value   |
| ----------------- | ------- |
| Accuracy          | 90.62%  |
| Balanced Accuracy | 82.18%  |
| F1 Macro          | 85.13%  |
| Inference Latency | 1.06 ms |
| Peak VRAM         | 0.70 GB |
| Model Size        | 33.4 MB |

### Compression Achieved

| Metric     | Teacher  | Student |
| ---------- | -------- | ------- |
| Accuracy   | 89.09%   | 90.62%  |
| Model Size | 176.7 MB | 33.4 MB |
| Latency    | 2.71 ms  | 1.06 ms |

* 5.3× model compression
* 2.5× faster inference

---

## 🔬 Temperature Ablation Study

| Temperature | Validation Accuracy |
| ----------- | ------------------- |
| T = 2       | 90.42%              |
| T = 4       | 90.15%              |
| T = 6       | 90.15%              |

The best performance was obtained using a distillation temperature of T = 2.

---

## 📏 Calibration Analysis

| Model             | ECE    |
| ----------------- | ------ |
| EfficientNet-B3   | 0.0634 |
| MobileNetV2 (T=2) | 0.0745 |

The distilled model remained well calibrated despite significant compression.

---

## 🌍 External Validation (PH2)

The best distilled MobileNetV2 model was evaluated on the external PH2 dataset without any additional training or fine-tuning.

| Metric            | Value  |
| ----------------- | ------ |
| Accuracy          | 88.00% |
| Balanced Accuracy | 70.94% |
| Precision         | 94.44% |
| Recall            | 42.50% |
| F1 Score          | 58.62% |

Confusion Matrix:

```text
[[159   1]
 [ 23  17]]
```

These results demonstrate meaningful cross-dataset generalization.

---

## Interpretability

Grad-CAM analysis was performed on melanoma, basal cell carcinoma, and nevus samples.

Key observations:

* Distilled models preserved diagnostically relevant lesion attention.
* Student models showed more focused lesion localization.
* Background activation was reduced compared to the teacher model.

---

## Repository Structure

```text
models/
training/
utils/

main.py
config.py
eval_distillation.py
evaluate_ph2.py
calibration_analysis.py
reliability_diagram.py
gradcam_analysis.py
```

---

## Requirements

```bash
pip install -r requirements.txt
```

---

## Future Work

* Additional external validation datasets
* Clinical deployment studies
* Real-time mobile implementation
* Multi-center evaluation

---

## Citation

If you use this repository in your research, please cite the corresponding publication once available.

---

## Author

Adhiraj Singh Bhadauria

B.Tech Adhiraj Singh Bhadauria

Madhav Institute of Technology and Science (Gwalior)
