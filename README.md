# PromptFolio

Official code for the NeurIPS 2024 paper:
**[Federated Learning from Vision-Language Foundation Models: Theoretical Analysis and Method](https://arxiv.org/abs/2409.19610)**

PromptFolio combines CLIP-based prompt learning with portfolio-theory-inspired federated aggregation to improve personalization across heterogeneous clients.

---

## Table of Contents

- [Installation](#installation)
- [Quick Start (CIFAR-100)](#quick-start-cifar-100)
- [Datasets](#datasets)
- [Running All Methods](#running-all-methods)
- [Key Arguments](#key-arguments)
- [Results](#results)

---

## Installation

**Requirements:** CUDA 12.x, conda

```bash
# 1. Create and activate environment
conda create -n PromptFolio python=3.8 yacs tqdm tabulate ftfy regex tensorboard -y
conda activate PromptFolio

# 2. Install PyTorch (CUDA 12.1 build)
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 \
    --index-url https://download.pytorch.org/whl/cu121

# 3. Install remaining dependencies
pip install timm gdown prettytable scikit-learn einops

# 4. Clone the repository
git clone https://github.com/YOUR_USERNAME/PromptFolio.git
cd PromptFolio
```

> If your CUDA version differs, find the matching PyTorch wheel at https://pytorch.org/get-started/previous-versions/

---

## Quick Start (CIFAR-100)

CIFAR-100 is the easiest dataset to start with — it downloads automatically via torchvision (~160 MB), so **no manual setup is needed**.

### Minimal run (copy-paste ready)

```bash
conda create -n PromptFolio python=3.8 yacs tqdm tabulate ftfy regex tensorboard -y
conda activate PromptFolio
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
pip install timm gdown prettytable scikit-learn einops

python federated_main.py \
  --dataset-config-file configs/datasets/cifar100.yaml \
  --config-file configs/trainers/PromptFolio/rn50.yaml \
  --trainer PromptFolio --model fedavg \
  --num_users 5 --frac 1.0 --frac_p 0.4 \
  --beta 0.3 --round 3 \
  --root ~/data
```

Expected runtime: ~5–10 min on a single GPU. Results are written to `result.json`.

### Step 1 — Create a data directory

```bash
mkdir -p /path/to/datasets
```

Replace `/path/to/datasets` with any local folder (e.g., `~/data` on Linux or `C:\data` on Windows).

### Step 2 — Run PromptFolio on CIFAR-100

```bash
python federated_main.py \
  --dataset-config-file configs/datasets/cifar100.yaml \
  --config-file configs/trainers/PromptFolio/rn50.yaml \
  --trainer PromptFolio \
  --model fedavg \
  --num_users 5 \
  --frac 1.0 \
  --frac_p 0.4 \
  --beta 0.3 \
  --round 3 \
  --root /path/to/datasets
```

**What this does:**
- 5 federated clients, all selected each round (`--frac 1.0`)
- Non-IID split with Dirichlet β=0.3 (moderate heterogeneity)
- 3 communication rounds (fast smoke test; paper uses 10)
- ResNet-50 CLIP backbone
- Results saved to `result.json`

**Expected runtime:** ~5–10 min on a single GPU for 3 rounds.

### Step 3 — Check results

```bash
# result.json is written after every round
cat result.json
```

Fields: `accuracy`, `error`, `f1` per round.

---

## Datasets

For full experiments the paper uses: **Caltech101, Oxford Pets, Food101, DTD, Oxford Flowers**.

See [DATASETS.md](DATASETS.md) for download and setup instructions for each dataset.

Set up your data folder like this:

```
/path/to/datasets/
├── cifar-100-python/     # auto-downloaded
├── caltech-101/
├── oxford_pets/
├── food-101/
├── dtd/
└── oxford_flowers/
```

---

## Running All Methods

All commands below use CIFAR-100. Replace `cifar100.yaml` with another dataset config to switch datasets.

```bash
DATA=/path/to/datasets
CFG=configs/datasets/cifar100.yaml

# PromptFolio (proposed method)
python federated_main.py --dataset-config-file $CFG --config-file configs/trainers/PromptFolio/rn50.yaml \
  --trainer PromptFolio --model fedavg --num_users 10 --beta 0.3 --round 10 --frac_p 0.4 --root $DATA

# CoOp (local only, no federation)
python federated_main.py --dataset-config-file $CFG --config-file configs/trainers/PromptFolio/rn50.yaml \
  --trainer PromptFL --model local --num_users 10 --beta 0.3 --round 10 --root $DATA

# PromptFL
python federated_main.py --dataset-config-file $CFG --config-file configs/trainers/PromptFolio/rn50.yaml \
  --trainer PromptFL --model fedavg --num_users 10 --beta 0.3 --round 10 --root $DATA

# PromptFL + Fine-Tuning
python federated_main.py --dataset-config-file $CFG --config-file configs/trainers/PromptFolio/rn50.yaml \
  --trainer PromptFLFT --model fedavg --num_users 10 --beta 0.3 --round 10 --root $DATA

# PromptFL + FedProx (mu=1.0)
python federated_main.py --dataset-config-file $CFG --config-file configs/trainers/PromptFolio/rn50.yaml \
  --trainer PromptFL --model fedavg --num_users 10 --beta 0.3 --round 10 --fedprox_mu 1.0 --root $DATA

# PromptFL + FedPer
python federated_main.py --dataset-config-file $CFG --config-file configs/trainers/PromptFolio/rn50.yaml \
  --trainer PromptFLFedPer --model fedavg --num_users 10 --beta 0.3 --round 10 --root $DATA

# PromptFL + FedAMP
python federated_main.py --dataset-config-file $CFG --config-file configs/trainers/PromptFolio/rn50.yaml \
  --trainer PromptFLFedAMP --model fedavg --num_users 10 --beta 0.3 --round 10 --root $DATA

# FedTPG
python federated_main.py --dataset-config-file $CFG --config-file configs/trainers/PromptFolio/rn50.yaml \
  --trainer FedTPG --model fedavg --num_users 10 --beta 0.3 --round 10 --root $DATA

# pFedPrompt
python federated_main.py --dataset-config-file $CFG --config-file configs/trainers/PromptFolio/rn50.yaml \
  --trainer pFedPrompt --model fedavg --num_users 10 --beta 0.3 --round 10 --root $DATA
```

To use a ViT-B/16 backbone instead of RN50, replace `rn50.yaml` with `vit_b16.yaml`.

---

## Key Arguments

| Argument | Default | Description |
|---|---|---|
| `--root` | — | **Required.** Path to the dataset root directory |
| `--dataset-config-file` | — | **Required.** Dataset config (e.g., `configs/datasets/cifar100.yaml`) |
| `--config-file` | — | **Required.** Trainer config (e.g., `configs/trainers/PromptFolio/rn50.yaml`) |
| `--trainer` | — | Method: `PromptFolio`, `PromptFL`, `PromptFLFT`, `PromptFLFedPer`, `PromptFLFedAMP`, `FedTPG`, `pFedPrompt` |
| `--model` | `fedavg` | Aggregation: `fedavg` or `local` |
| `--num_users` | `1` | Total number of federated clients |
| `--frac` | `1.0` | Fraction of clients selected per round |
| `--frac_p` | — | Fraction for portfolio clustering (PromptFolio only) |
| `--beta` | — | Dirichlet β for non-IID partitioning (lower = more heterogeneous) |
| `--round` | `10` | Number of communication rounds |
| `--fedprox_mu` | — | Proximal term coefficient for FedProx |

---

## Results

After training, `result.json` contains per-round metrics:

```json
[
  {"round": 1, "accuracy": 0.72, "error": 0.28, "f1": 0.71},
  {"round": 2, "accuracy": 0.75, "error": 0.25, "f1": 0.74},
  ...
]
```

---

## Citation

```bibtex
@inproceedings{promptfolio2024,
  title     = {Federated Learning from Vision-Language Foundation Models: Theoretical Analysis and Method},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2024},
  url       = {https://arxiv.org/abs/2409.19610}
}
```
