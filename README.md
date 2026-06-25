# Deep Learning Fundamentals

> Two independent deep learning systems built in a single GPU-enabled Google Colab notebook:
> **(1)** AI-assisted chest X-ray diagnosis (classification + lung segmentation) and
> **(2)** automatic speech recognition (ASR) with Connectionist Temporal Classification (CTC).

---

## Overview

This repository contains the full implementation, experiments, and technical paper for a project that applies modern deep learning to two distinct domains:

| Module | Task | Core models |
|---|---|---|---|
| **1 — Computer Vision** | Chest X-ray multi-class classification | ResNet-50, DenseNet-121 (transfer learning) |
| **1 — Computer Vision** | Lung segmentation | U-Net |
| **2 — Speech (CTC)** | Spoken-command recognition | CNN + BiGRU + CTC |
| **2 — Speech (CTC)** | Sentence-level speech-to-text | CNN + BiGRU + CTC |
| **2 — Speech (CTC)** | Sentence-level speech-to-text | CNN + BiGRU + CTC |

Everything runs cloud-first (no local downloads required): datasets are pulled via the Kaggle API / `kagglehub`, and checkpoints are saved to Google Drive.

---

## Module 1 - Chest X-Ray Classification & Segmentation

An end-to-end computer vision system for respiratory-disease screening.

- **Classification** — distinguishes *Normal*, *Pneumonia*, *COVID-19*, and *Tuberculosis* using two ImageNet-pretrained backbones fine-tuned for the task.
- **Segmentation** — a U-Net delineates the lung region, providing an anatomical region of interest.
- **Interpretability** — Grad-CAM heatmaps verify the classifiers attend to lung fields rather than artifacts.

### Highlights
- Class imbalance handled with a **weighted cross-entropy loss** + **`WeightedRandomSampler`**.
- **Mixed-precision** training (AMP), cosine LR decay, and **early stopping** with best-checkpoint restoration.
- Evaluation via accuracy, precision, recall, F1, confusion matrices, ROC curves (classification) and IoU / Dice (segmentation).

### Results (test set, 771 images)

| Metric | ResNet-50 | DenseNet-121 |
|---|---|---|
| Accuracy | 0.8988 | **0.9092** |
| Precision (macro) | 0.8914 | **0.9268** |
| Recall (macro) | 0.9312 | **0.9373** |
| F1 (macro) | 0.9096 | **0.9313** |
| F1 (weighted) | 0.8991 | **0.9097** |

**Per-class F1:** DenseNet-121 reaches **0.988** on Tuberculosis and **0.959** on COVID-19. The dominant residual error for both models is the *Normal ↔ Pneumonia* boundary.

**Lung segmentation (U-Net, 106 test images):** mean **IoU = 0.9290**, mean **Dice = 0.9625** — above the ~0.85 Dice threshold generally considered clinically acceptable.

---

## Module 2 - Automatic Speech Recognition with CTC

A speech-to-text system that converts audio to text **without frame-level alignment**, using a CNN + bidirectional GRU acoustic model trained with `nn.CTCLoss` and greedy decoding.

**Architecture:** 3-layer 1-D CNN encoder (40 → 128 → 128 → 256) → 3-layer BiGRU (hidden 256) → linear projection → log-softmax over a 29-token character vocabulary (blank + a–z + space + unknown). ~3.3M parameters.

The system was evaluated on three settings of increasing difficulty:

| Setting | Dataset |
|---|---|---|---|
| Single-speaker sentences | LJ Speech (open vocabulary) |
| Unseen-speaker sentences † | LibriTTS (speaker-disjoint) |

† Warm-started from the LJ Speech checkpoint and fine-tuned with audio augmentation + SpecAugment on **speaker-disjoint** splits. The large WER–CER gap on sentences is characteristic of a compact, character-level model with no language model: predictions are near-phonetic (e.g., *"orlians"* → *"orleans"*) and mostly correct character-by-character.

---

## Datasets

| Dataset | Use | Size | Source |
|---|---|---|---|
| Chest X-Ray (Pneumonia/COVID-19/TB) | Classification | 7,135 images, 4 classes | [Kaggle](https://www.kaggle.com/datasets/jtiptj/chest-xray-pneumoniacovid19tuberculosis) |
| Montgomery County | Segmentation | 138 images + lung masks | NIH / Mendeley |
| Shenzhen Hospital | Segmentation | 566 usable images + masks | NIH / Mendeley |
| LJ Speech | ASR (sentences) | 13,100 clips, 1 speaker | [Kaggle](https://www.kaggle.com/datasets/mathurinache/the-lj-speech-dataset) |
| LibriTTS | ASR (multi-speaker) | 247 speakers (subset used) | [Kaggle](https://www.kaggle.com/datasets/pratt3000/libritts) |

---

## Repository Structure

```
deep-learning-fundamentals-project/
├── Deep_Learning_Project.ipynb   # Main notebook (both modules, fully documented)                   
├── Ai_Project2_Deep_Learning_Paper.pdf # Technical paper (LaTeX)
├── README.md
```

---

## Getting Started

The project is designed to run in **Google Colab** with a **T4 GPU**.

1. **Open the notebook** via the *Open in Colab* badge above.
2. Set the runtime: `Runtime → Change runtime type → GPU (T4)`.
3. **Mount Google Drive** and place your `KaggleApi.json` (Kaggle API key) and the segmentation `.zip` files under
   `MyDrive/Proyecto-2-IA-2026/`.
4. **Run the cells top to bottom.** Each module is self-contained; datasets download automatically and checkpoints are written back to Drive.

### Key dependencies
```
torch  torchvision  torchmetrics  torchaudio
segmentation-models-pytorch  albumentations  grad-cam
librosa  soundfile  jiwer  kagglehub
scikit-learn  matplotlib  seaborn  pandas  numpy
```

---

## Technical Paper

A full write-up in is available in [`AI_Project2_Deep_Learning_Paper.pdf`](AI_Project2_Deep_Learning_Paper.pdf), covering the introduction, theoretical framework (transfer learning, U-Net, Grad-CAM, CTC), methodology, results and analysis, and conclusions.

---

## Authors

- **Dylan Molina Arroyo**
- **Fabricio Alfaro Cabezas**
- **Andrés Esquivel Gómez**

---

## Acknowledgments

Built on PyTorch, [segmentation-models-pytorch](https://github.com/qubvel-org/segmentation_models.pytorch), and [pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam). Datasets courtesy of their respective authors (NIH, Montgomery County, Shenzhen Hospital, LJ Speech, LibriTTS).
