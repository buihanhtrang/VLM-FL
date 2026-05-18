# PromptFolio — Study Notes

---

## Q1: How does the code work?

### Big Picture
PromptFolio is a federated learning system where multiple clients (simulated on a single machine) each hold private data. Instead of sharing raw data or full model weights, they only share small **learned text prompts** for CLIP. The server aggregates these prompts each round.

### What Gets Trained
CLIP is a large frozen vision-language model. Only a tiny `PromptLearner` is trained — a set of learnable context vectors prepended to each class name in CLIP's text encoder. Everything else (image encoder, text encoder weights) is frozen.

```python
for name, param in self.model.named_parameters():
    if "prompt_learner" not in name:
        param.requires_grad_(False)
```

### The Two-Prompt Design
Each client holds 2 prompts:
- `ctx[0]` — the **global prompt** (shared/aggregated across clients)
- `ctx[1]` — the **local prompt** (private to each client)

They are blended at inference:
```python
text_features = (1 - frac) * text_features0 + frac * text_features1
logits = logit_scale * image_features @ text_features.T
```
`frac_p` controls the blend: 0 = fully global, 1 = fully local.

### Federated Training Loop (each round)
1. Randomly select `m = frac * num_users` clients
2. Each client downloads global prompt + its local state
3. Client trains for 1 epoch on private data (cross-entropy)
4. Client uploads updated weights
5. Server averages only `ctx[0]` (global prompts), weighted by dataset size
6. Evaluate test accuracy on selected clients

### Data Flow Summary
```
Dataset (partitioned by Dirichlet)
    │
    ├─ Client 0: trains prompt → uploads ctx[0]
    ├─ Client 1: trains prompt → uploads ctx[0]
    └─ Client N: ...
                      │
            Server averages ctx[0]
                      │
            Broadcasts new global ctx[0]
                      │
            Each client: ctx = [global_ctx[0], local_ctx[1]]
                      │
            Prediction: blend(global, local) via frac_p
```

---

## Q2: What are the hyperparameters?

### Federated Setup
| Argument | Default | Description |
|---|---|---|
| `--num_users` | 1 | Number of clients |
| `--frac` | 1.0 | Fraction of clients selected per round |
| `--round` | 10 | Communication rounds |
| `--fedprox_mu` | -1 (off) | FedProx proximal term; set positive to enable |

### PromptFolio-Specific
| Argument | Default | Description |
|---|---|---|
| `--frac_p` | 0.4 | Blend ratio between global and local prompt |
| `--n_ctx` | 8 | Number of learnable context tokens |
| `--num_prompt` | 2 | Number of prompts per client (global + local) |

### Optimization
| Argument | Default | Description |
|---|---|---|
| `--lr` | 0.001 | Learning rate |
| `--gamma` | 1 | Step decay factor for LR scheduler |
| `--train_batch_size` | 16 | Training batch size |
| `--test_batch_size` | 16 | Test batch size |

### Data Partitioning
| Argument | Default | Description |
|---|---|---|
| `--beta` | 0.3 | Dirichlet concentration; lower = more non-IID |
| `--num_shots` | 16 | Shots per class in few-shot mode |
| `--useall` | False | Use all training data instead of few-shot |

Most impactful to tune: `--frac_p`, `--beta`, `--n_ctx`, `--lr`.

---

## Q3: Is the code fair for FL and what FL problem does it solve?

### What FL Problem It Solves
The core problem addressed is **statistical heterogeneity (non-IID data)**. In standard FedAvg, a single global model performs poorly on any individual client because the averaged model is a compromise that fits no one well.

PromptFolio addresses this via **personalized federated learning (pFL)** — the local prompt `ctx[1]` is never aggregated, so each client retains a personalized component.

### Fairness Assessment

**What it does well:**
- Each client gets a personalized model (local prompt never aggregated)
- FedAvg aggregation is data-weighted — clients with more data contribute proportionally

**What is missing for true fairness:**

| FL Challenge | Handled? |
|---|---|
| Non-IID data | Yes — via personalized local prompt |
| Communication efficiency | Yes — only tiny prompts are shared |
| Privacy (no raw data sharing) | Yes |
| Fairness across clients | No — no fairness objective or metric |
| Adaptive personalization | No — fixed `frac_p` for all clients |
| Client drift | Optional via `--fedprox_mu` |

Specifically missing:
- No min-max fairness objective (e.g., maximizing worst-performing client's accuracy)
- Metric is mean accuracy only — a client at 10% accuracy is invisible in a high mean
- Client selection is uniform random — struggling clients are never prioritized
- `frac_p` is global and fixed — no per-client adaptation

**Bottom line:** Solid solution for accuracy under heterogeneity, but not a fairness-aware FL system.

---

## Q4: What are task-relevant and task-irrelevant features?

These terms come from the **paper** (NeurIPS 2024, arxiv 2409.19610), not the code. They are used to explain theoretically why standard FedAvg fails.

### Task-Relevant Features
Features in the CLIP embedding space **useful for the classification task** — shape, texture, object parts. These are consistent across clients and should be shared.

### Task-Irrelevant Features
Features **spuriously correlated with labels on a specific client** but don't generalize — background color, lighting, domain style. These vary per client due to different data distributions.

### Why This Matters
When FedAvg averages prompts:
- Client A: "cats have fur + dark background" (task-relevant + task-irrelevant mixed)
- Client B: "cats have fur + bright background"
- After averaging: task-relevant (fur) averages well, but task-irrelevant parts **conflict and cancel**, degrading the global prompt

### How PromptFolio Maps Onto This

| Prompt | Role | Captures |
|---|---|---|
| `ctx[0]` — global, aggregated | Shared knowledge | Task-relevant features |
| `ctx[1]` — local, private | Client-specific bias | Task-irrelevant features |

`frac_p` is a knob for how much task-irrelevant local information to trust.

---

## Q5: Are task-relevant/irrelevant features for image, text, or both?

**Text only.**

The image encoder is completely frozen — it never adapts. Only the `PromptLearner` (text context vectors `ctx`) is trained. So the global/local split is entirely in the **text embedding space**:

```
ctx[0] (global) → text encoder → task-relevant text features
ctx[1] (local)  → text encoder → task-irrelevant text features
```

### Why Only Text?
- Images are fixed — a cat photo looks the same on every client
- What varies across clients is the **language/context** describing the task (e.g., professional photos vs. sketches)
- The text prompt shifts to match each client's visual style without touching the image encoder

---

## Q6: Can my computer run this code?

### System Specs
| Component | Spec | Required |
|---|---|---|
| GPU | NVIDIA GeForce, 4 GB VRAM | ~6–8 GB recommended |
| CUDA Driver | 11.7 | 12.1 (for default PyTorch build) |
| RAM | 16 GB | ~8 GB minimum |

### Problems
1. **VRAM too small** — CLIP ViT-B/16 alone takes ~3 GB in fp16; training will likely OOM
2. **CUDA version mismatch** — must install PyTorch with `--index-url .../cu117` instead of `cu121`

### Workaround: Minimal Experiment
Use ResNet50 backbone and small settings:
```bash
python federated_main.py \
  --config-file configs/trainers/PromptFolio/rn50.yaml \
  --dataset-config-file configs/datasets/caltech101.yaml \
  --model fedavg --trainer PromptFolio \
  --num_users 2 --frac 1.0 --beta 0.3 --round 3 \
  --train_batch_size 4 --test_batch_size 4 --num_shots 4
```

---

## Q7: What are the advanced methods of PromptFolio?

### 1. Portfolio Prompt Design (core)
Two-prompt blend in `CustomCLIP.forward()` — active every forward pass.

### 2. Selective Aggregation (`PromptFolio.py:412`)
Overrides base FedAvg to only aggregate `ctx[0]` (global slice), leaving `ctx[1]` private:
```python
global_list = [self.local_info[idx]['prompt_learner.ctx'][:1] for idx in idx_users]
self.global_info = average_weights(global_list, idx_users, islist=True)
```

### 3. Portfolio Stitching on Download (`PromptFolio.py:419`)
Before each round, global and local prompts are stitched:
```python
folio = torch.cat([self.global_info, self.local_info[idx]['prompt_learner.ctx'][1:]], dim=0)
```

### 4. Structure Loss (`PromptFolio.py:398`) — defined, not active
Regularizes the prompt to produce logits that respect image similarity structure:
```python
S = img_feature @ img_feature.T
result = -sum(S * (temp @ temp.T)) / N²
```

### 5. Generalized Cross Entropy (`PromptFolio.py:247`) — defined, not active
Noise-robust loss using Box-Cox transformation (`q=0.7`):
```python
loss = (1 - p^q) / q
```
Standard `F.cross_entropy` is used in practice instead.

### Summary

| Method | Status |
|---|---|
| Two-prompt portfolio blend | **Active** |
| Selective global aggregation | **Active** |
| Portfolio stitching on download | **Active** |
| Structure loss | Defined, not called |
| Generalized cross entropy | Defined, not called |

---

## Q8: How to run a small experiment on my computer

### Setup Steps

**Step 1 — Install Miniconda**
```
winget install -e --id Anaconda.Miniconda3
```
Check "Add to PATH" during install, then restart terminal.

**Step 2 — Create environment**
```bash
conda create -n PromptFolio python=3.8 yacs tqdm tabulate ftfy regex tensorboard -y
conda activate PromptFolio
```

**Step 3 — Install PyTorch (CUDA 11.7 compatible)**
```bash
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu117
pip install timm gdown prettytable scikit-learn einops
```

**Step 4 — Install Dassl**
```bash
cd C:\Users\Trang\PromptFolio
pip install -e Dassl/
```

**Step 5 — Run minimal experiment**
```bash
python federated_main.py \
  --dataset-config-file configs/datasets/caltech101.yaml \
  --config-file configs/trainers/PromptFolio/rn50.yaml \
  --model fedavg --trainer PromptFolio \
  --num_users 2 --frac 1.0 --beta 0.3 --round 3 \
  --train_batch_size 4 --test_batch_size 4 --num_shots 4 \
  --root C:\path\to\your\datasets
```

**Why these values:**
- `rn50` — less VRAM than ViT-B/16
- `--num_users 2` — only 2 clients
- `--round 3` — just 3 rounds to verify it runs
- `--train_batch_size 4` — fits in 4 GB VRAM
- `--num_shots 4` — tiny data per client
