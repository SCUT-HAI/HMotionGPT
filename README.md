# HMotionGPT: Aligning Hand Motions and Natural Language for Activity Understanding with Smart Rings

<p align="center">
  <a href="https://dl.acm.org/doi/10.1145/3729543"><img src="https://img.shields.io/badge/Paper-IMWUT%202025-blue" alt="Paper"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green" alt="License"></a>
</p>

> **HMotionGPT: Aligning Hand Motions and Natural Language for Activity Understanding with Smart Rings**
>
> Accepted in *Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies* (**IMWUT 2025**)

## Overview

HMotionGPT is a multimodal framework that bridges smart-ring IMU signals and natural language for hand-centric activity understanding. This repository releases the core open-source training code for the IMU-conditioned language model, including:

- stage-1 alignment training for the IMU projector
- stage-2 supervised fine-tuning with a frozen projector
- configurable LLM backbones
- a minimal runnable example for smoke testing

This release focuses on the model construction and training pipeline. Internal evaluation scripts, rebuttal code, private data, and large checkpoints are not included.

## Repository Structure

```text
HMotionGPT/
├── configs/
├── example_data/
├── example_models/
├── hmotiongpt/
│   ├── cli/
│   ├── data/
│   ├── models/
│   ├── training/
│   └── utils/
├── scripts/
├── LICENSE
├── README.md
└── requirements.txt
```

## Installation

```bash
git clone https://github.com/SCUT-HAI/HMotionGPT.git
cd HMotionGPT
pip install -r requirements.txt
```

Recommended environment:

- Python >= 3.8
- PyTorch with CUDA support
- transformers

## Quick Start

The repository includes a tiny local backbone and a tiny example dataset for smoke testing. This is useful for verifying that the code path is runnable before switching to a real LLM backbone and your own IMU features.

Stage 1 smoke test:

```bash
PYTHONPATH=. python -m hmotiongpt.cli.train_alignment --config configs/smoke_local_alignment.yaml
```

Stage 2 smoke test:

```bash
PYTHONPATH=. python -m hmotiongpt.cli.train_sft --config configs/smoke_local_sft.yaml
```

You can also use the shell wrappers:

```bash
bash scripts/train_alignment.sh
bash scripts/train_sft.sh
```

## Training With Your Own Backbone

### 1. Configure the backbone

Edit [configs/alignment.yaml](configs/alignment.yaml) and [configs/sft.yaml](configs/sft.yaml):

```yaml
model:
  name_or_path: "/path/to/your/llm"
```

Users should download the language model backbone by themselves and set `name_or_path` to a local path or a Hugging Face model identifier.

### 2. Run stage-1 alignment

```bash
PYTHONPATH=. python -m hmotiongpt.cli.train_alignment --config configs/alignment.yaml
```

This stage freezes the LLM and trains the IMU projector.

### 3. Run stage-2 SFT

Set the projector checkpoint path in [configs/sft.yaml](configs/sft.yaml):

```yaml
model:
  name_or_path: "/path/to/your/llm"
  projector_path: "../outputs/alignment/your_run/projector.pt"
```

Then run:

```bash
PYTHONPATH=. python -m hmotiongpt.cli.train_sft --config configs/sft.yaml
```

This stage loads the projector from stage 1, freezes it, and fine-tunes the LLM.

## Data Format

### Alignment JSONL

```json
{
  "imu_vec_path": "data/imu_windows/example.npy",
  "text": "右手将物体向左移动。"
}
```

### SFT JSONL

```json
{
  "imu_vec_path": "data/imu_windows/example.npy",
  "instruction": "给定一个IMU动作片段，请选择最合适的动作标签。",
  "input": "",
  "output": "炒菜"
}
```

Supported IMU path keys include:

- `imu_vec_path`
- `imu_path`
- `imu_file`
- `imu`

The IMU feature file should be a `.npy` array with shape `[T, D]`.

## Output

Stage 1 saves:

- projector checkpoints
- tokenizer files
- `metrics.jsonl`
- tensorboard logs when available

Stage 2 saves:

- LLM checkpoints
- projector checkpoint copy
- tokenizer files
- `metrics.jsonl`
- tensorboard logs when available

## Notes

- The toy assets under `example_data/` and `example_models/` are only for minimal verification.
- Real experiments should use your actual IMU feature files and your target LLM backbone.
- This repository does not include private datasets, rebuttal code, evaluation-only pipelines, or full released checkpoints.

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

## License

This project is released under the [MIT License](LICENSE).

## Acknowledgments

We sincerely thank all participants who volunteered in our data collection study, and the anonymous reviewers for their insightful feedback.
