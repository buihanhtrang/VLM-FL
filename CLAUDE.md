# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PromptFolio is a federated learning framework for vision-language foundation models, implementing the method from the NeurIPS 2024 paper ["Federated Learning from Vision-Language Foundation Models: Theoretical Analysis and Method"](https://arxiv.org/abs/2409.19610). It combines CLIP-based prompt learning with federated optimization strategies.

## Environment Setup

```bash
conda create -n PromptFolio python=3.8 yacs tqdm tabulate ftfy regex tensorboard
conda activate PromptFolio
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install timm gdown prettytable scikit-learn einops
```

Dataset setup is documented in `DATASETS.md`. The dataset root directory must be specified via `--root` when running experiments.

## Running Experiments

```bash
# PromptFolio on Caltech101 with FedAvg, 10 clients, non-IID beta=0.3
python federated_main.py \
  --dataset-config-file configs/datasets/caltech101.yaml \
  --config-file configs/trainers/PromptFolio/vit_b16.yaml \
  --model fedavg \
  --trainer PromptFolio \
  --frac_p 0.4 \
  --num_users 10 \
  --beta 0.3 \
  --round 10 \
  --root /path/to/datasets
```

Results are written to `result.json` (accuracy, error, F1 per round).

## Key Arguments (`federated_main.py`)

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | `fedavg` | Aggregation: `fedavg` or `local` |
| `--trainer` | — | Algorithm: `PromptFolio`, `PromptFL`, `PromptFLFT`, `PromptFLFedPer`, `PromptFLFedAMP`, `FedTPG`, `pFedPrompt` |
| `--round` | `10` | Number of communication rounds |
| `--num_users` | `1` | Total number of federated clients |
| `--frac` | `1.0` | Fraction of clients selected per round |
| `--beta` | — | Dirichlet concentration for non-IID partitioning |
| `--mu` | — | FedProx proximal term coefficient |

## Architecture

### Training Loop (`federated_main.py`)
Central orchestrator: initializes a local trainer, runs federated rounds with client selection, model upload/download, and FedAvg or local aggregation. Evaluation happens per round on selected clients.

### Trainers (`trainers/`)
Each trainer wraps a CLIP-based prompt learning strategy:
- `PromptFolio.py` — main method using portfolio-based prompt aggregation
- `promptfl.py` — PromptFL baseline
- `promptflFT.py` — PromptFL + fine-tuning
- `promptflFedPer.py` / `promptflFedAMP.py` — PromptFL + personalization variants
- `FedTPG.py` — FedTPG baseline
- `coop.py` — CoOp (local-only, no federation)
- `clip.py` — plain CLIP zero-shot baseline

### Federated Utilities (`fed_utils.py`)
- `average_weights()` — data-weighted FedAvg aggregation over prompt parameters
- `cluster_weights()` — agglomerative clustering of client prompts (used by PromptFolio)
- `count_parameters()` — counts trainable parameters for specific model components

### Configuration System
Uses YACS (via `Dassl/`). Dataset configs live in `configs/datasets/*.yaml`; trainer configs in `configs/trainers/PromptFolio/*.yaml` (backbones: `vit_b16`, `rn50`).

### Dassl Framework (`Dassl/`)
Internal dependency providing: config management, data managers/samplers, backbone definitions (ResNet, ViT), optimizers, evaluation metrics, and checkpointing. Not intended to be modified directly.

### CLIP Integration (`clip/`)
Houses the CLIP model loading, tokenizer, and architectural components used by all trainers as the vision-language backbone.

## Data Partitioning

Non-IID data splits across clients use Dirichlet distribution (`--beta`). Lower beta = more heterogeneous. `datasplit.py` handles the partitioning logic; `dataset.py` provides truncated dataset classes that accept `dataidxs` to restrict to a client's local data subset.
