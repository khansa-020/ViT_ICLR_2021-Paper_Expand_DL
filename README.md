# ViT_ICLR_2021-Paper_Expand_DL
# ViT Vision-Transformer — Reproduction & Extension to Medical Imaging

PyTorch reimplementation of [ViT (ICLR 2021)](https://arxiv.org/abs/1510.00149) — reproduced on CIFAR10, then extended to 8-class Blood Cell classification on Blood-MNIST dataset with Generalization and robustness analysis.


---

## What We Did

Reproduced the Vision Transformer pipeline on CIFAR 10 (B-16 and B-32), then applied the same pipeline — with no changes to core principles — to a real-world Medical Dataset: Blood cells type classification on the BloodMNIST Heamatology dataset. The central question explored is whether model pre-Trained on Natural images can work on cross Domain Medical Dataset and classify Blood Cell types (Lekucytes, Esnophyll, granulytes...).

---

## Pipeline

*Baseline* → Original ViT results (paper)
*Reproduced* → pipeline implementation on CIFAR 10-100
*Extension* → Fine-tuned ViT on BloodMNIST Dataset

**Dataset → Preprocessing → ViT Model (pretrained)→ Fine-tuning → Validation →  Testing → Metrics + Visualizations**

- **Stage 1 — Preprocessing** Resize medical images to 224×224, apply augmentation (flip, rotation), normalize with ImageNet stats, and handle class imbalance using weighted sampling
- **Stage 2 — ViT Fine-Tuning** Load pretrained ViT (ImageNet21k), replace classification head, freeze 6 layers, and fine-tune using AdamW + cosine scheduler with LLRD and mixed precision
- **Stage 3 — Evaluation**
Evaluate best model on test set using Accuracy, F1-score, ROC-AUC, and generate confusion matrix and classification report


## Pipeline Architecture
Raw Dataset (MedMNIST / CIFAR)
        ↓
Preprocessing & Augmentation
        ↓
ViT (Pretrained on ImageNet21k)
        ↓
Fine-Tuning (LLRD + FP16 + Layer Freezing)
        ↓
Validation & Checkpoint Selection
        ↓
Testing & Performance Evaluation
        ↓
Visualization & Reporting

---

## Datasets

- **BloodMNIST** — 56,000 blood cells types images, 8 classes, medical Heamatology dataset
- **ImageNet21k** — 21k natural images, 1k classes,(high accuracy)
- **CIFAR-10** — 50000 nature images, 10 classes

---

## Models

| Model (Pretrained on ImageNet21k) | Type       | Dataset       | Accuracy |
|-----------------------------------|------------|---------------|----------|
| ViT B-16/32                       | Baseline   | CIFAR-10/100  | 97.7%    |
| ViT B-32                          | Reproduced | CIFAR-10/100  | 97.6%    |
| ViT B-16                          | Extension  | BloodMNIST    | 91.6%    |

---

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/ViT_ICLR_2021-Paper_Expand_DL .git
pip install -r requirements.txt
```

Notebooks were developed on Kaggle Tesla T4 GPU. For extention BloodMNIST dataset is used, download the dataset [here](https://www.kaggle.com/datasets/michaelyousef/blood-mnist) and  path is in cell at the top of the notebook.

---

## References

@article{dosovitskiy2020vit,
  title={An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale},
  author={Dosovitskiy, Alexey and Beyer, Lucas and Kolesnikov, Alexander and Weissenborn, Dirk and Zhai, Xiaohua and Unterthiner, Thomas and  Dehghani, Mostafa and Minderer, Matthias and Heigold, Georg and Gelly, Sylvain and Uszkoreit, Jakob and Houlsby, Neil},
  journal={ICLR},
  year={2021}
}

@article{tolstikhin2021mixer,
  title={MLP-Mixer: An all-MLP Architecture for Vision},
  author={Tolstikhin, Ilya and Houlsby, Neil and Kolesnikov, Alexander and Beyer, Lucas and Zhai, Xiaohua and Unterthiner, Thomas and Yung, Jessica and Steiner, Andreas and Keysers, Daniel and Uszkoreit, Jakob and Lucic, Mario and Dosovitskiy, Alexey},
  journal={arXiv preprint arXiv:2105.01601},
  year={2021}
}

[The BloodMNIST Dataset] this is medical imaging heamatology dataset having difffrent type of blood cells (https://www.kaggle.com/datasets/michaelyousef/blood-mnist). Scientific Data.
 google repository for Vision-Transformer is here: https://github.com/google-research/vision_transformer/tree/main/vit_jax
Results from the original ViT paper (https://arxiv.org/abs/2010.11929) have been replicated using the models from gs://vit_models/imagenet21k in my repo
