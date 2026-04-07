# HMotionGPT: Aligning Hand Motions and Natural Language for Activity Understanding with Smart Rings

<p align="center">
  <a href="https://dl.acm.org/doi/10.1145/3729543"><img src="https://img.shields.io/badge/Paper-IMWUT%202025-blue" alt="Paper"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green" alt="License"></a>
</p>

> **HMotionGPT: Aligning Hand Motions and Natural Language for Activity Understanding with Smart Rings**
>
> Accepted in *Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies* (**IMWUT 2025**)

---

## Overview

**HMotionGPT** is a multimodal framework that bridges raw hand-motion signals captured by commodity **smart rings** with natural language, enabling fine-grained activity understanding. Inspired by the success of large language models, HMotionGPT trains a motion tokenizer to discretize continuous IMU-based hand-motion sequences and then aligns these motion tokens with a pre-trained language model through instruction tuning. This joint representation allows the model to:

- **Recognize** everyday hand-based activities from wrist-worn smart ring sensor data.
- **Describe** detected activities in natural language (motion captioning).
- **Answer** open-ended questions about hand motions (motion question answering).
- **Retrieve** relevant motion clips given a natural-language query.

The framework is evaluated on a newly collected smart-ring activity dataset and demonstrates strong performance across multiple language-motion understanding tasks.

---

## Key Features

- 📡 **Smart-ring IMU input** — works with the compact, always-on sensor suite of a smart ring (accelerometer + gyroscope).
- 🔤 **Language-motion alignment** — motion sequences are tokenized and projected into the token space of a large language model.
- 💬 **Instruction-tuned LLM backbone** — supports zero-shot and few-shot generalization to unseen activities.
- 🏆 **IMWUT 2025** — peer-reviewed and published in a top-tier ubiquitous computing venue.

---

## Framework

```
Smart Ring IMU Data
        │
        ▼
 Motion Tokenizer (VQ-VAE)
        │
        ▼
  Motion Token Sequence
        │
        ▼
  LLM (Instruction Tuning)  ◄──── Natural Language Prompt
        │
        ▼
  Activity Label / Caption / QA Answer
```

---

## Requirements

```
Python >= 3.8
PyTorch >= 1.13
transformers >= 4.30
numpy
scipy
tqdm
```

Install all dependencies via:

```bash
pip install -r requirements.txt
```

---

## Dataset

The dataset and pre-processed features used in our experiments will be released at this repository. Please check back soon or **star** the repo to get notified.

---

## Model Checkpoints

Pre-trained model checkpoints will be released here upon acceptance formality completion. Stay tuned.

---

## Usage

### Training

```bash
python train.py --config configs/hmotiongpt.yaml
```

### Evaluation

```bash
python evaluate.py --config configs/hmotiongpt.yaml --checkpoint checkpoints/hmotiongpt.pth
```

> **Note:** Detailed configuration options and data-preparation scripts will be uploaded along with the full code release.

---

## Citation

If you find this work useful, please cite our paper:

```bibtex
@article{hmotiongpt2025,
  title     = {HMotionGPT: Aligning Hand Motions and Natural Language for Activity Understanding with Smart Rings},
  journal   = {Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies},
  year      = {2025},
  publisher = {ACM}
}
```

---

## License

This project is released under the [MIT License](LICENSE).

---

## Acknowledgments

We sincerely thank all participants who volunteered in our data collection study, and the anonymous reviewers for their insightful feedback.

