# NAICS Gemini Quick Start Guide

Welcome to NAICS Gemini! This guide will get you up and running with hierarchical contrastive learning on NAICS codes in under 30 minutes.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Data Pipeline](#data-pipeline)
- [Training Your First Model](#training-your-first-model)
- [Evaluating Results](#evaluating-results)
- [Next Steps](#next-steps)
- [Common Issues](#common-issues)

---

## Prerequisites

### System Requirements

**Hardware:**
- **GPU:** NVIDIA GPU with 16GB+ VRAM (recommended) or Apple Silicon Mac with 16GB+ unified memory
- **CPU:** 8+ cores recommended for data streaming
- **Storage:** 10GB free space for datasets and checkpoints
- **RAM:** 32GB+ recommended

**Software:**
- Python 3.10 or higher
- [uv](https://github.com/astral-sh/uv) package manager (recommended) or pip
- Git

### Quick uv Installation

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/lowmason/naics-gemini.git
cd naics-gemini
```

### 2. Install Dependencies

Using uv (recommended):
```bash
uv sync
```

Using pip:
```bash
pip install -e .
```

### 3. Verify Installation

Check that your GPU backend is detected:

```bash
uv run python -m naics_gemini.utils.backend
```

**Expected output:**
```
Python 3.10.x on Darwin arm64
Torch version: 2.5.1
MPS (Apple Silicon Metal) available
Ran tensor test on: MPS — success!
```

Or for NVIDIA GPUs:
```
CUDA available: NVIDIA GeForce RTX 4090
  CUDA version: 12.1
```

---

## Data Pipeline

The NAICS Gemini training system requires three datasets generated sequentially. Each step validates and enriches the data.

### Option 1: Generate All Data (Recommended for First Run)

```bash
uv run naics-gemini data all
```

This runs the complete pipeline:
1. **Preprocess** (2-3 minutes): Downloads and cleans NAICS data from Census Bureau
2. **Distances** (5-10 minutes): Computes pairwise graph distances for 2,125 codes
3. **Triplets** (10-15 minutes): Generates 263M+ training triplets with hardness labels

**Total time:** ~20 minutes

### Option 2: Run Steps Individually

Useful for debugging or when you've already run earlier steps:

```bash
# Step 1: Download and preprocess NAICS codes
uv run naics-gemini data preprocess

# Step 2: Compute graph distances between all code pairs
uv run naics-gemini data distances

# Step 3: Generate (anchor, positive, negative) triplets
uv run naics-gemini data triplets
```

### Verify Data Generation

After completion, check that all files exist:

```bash
ls -lh data/
```

**Expected output:**
```
naics_descriptions.parquet   (1.2 MB)  - Text data for 2,125 codes
naics_distances.parquet      (48 MB)   - 3M pairwise distances
naics_training_pairs.parquet (3.2 GB)  - 263M triplets
```

---

## Training Your First Model

### Quick Training Run (5 Epochs, Easy Curriculum)

Start with the easiest curriculum to verify everything works:

```bash
uv run naics-gemini train --curriculum 01_stage_easy
```

**What happens during training:**
- Loads pretrained `all-mpnet-base-v2` transformer
- Applies LoRA fine-tuning (r=8, alpha=16)
- Trains 4-channel encoder (title, description, excluded, examples)
- Routes channels through Mixture of Experts (4 experts, Top-2)
- Projects to hyperbolic space (Lorentz model)
- Computes contrastive loss on filtered triplets

**Training configuration:**
- **Curriculum:** Level 6 codes, unrelated negatives (easiest)
- **Batch size:** 32
- **K negatives:** 16 per positive
- **Max epochs:** 5
- **Precision:** bf16-mixed

**Expected duration:** 2-4 hours on RTX 4090 or Apple M1/M2 Max

### Monitor Training Progress

Training logs are written to:
- **Console:** Real-time progress with loss metrics
- **TensorBoard:** `outputs/01_stage_easy/`

View TensorBoard logs:
```bash
tensorboard --logdir outputs/01_stage_easy
```

Open http://localhost:6006 to see:
- Train/validation contrastive loss
- MoE load balancing loss
- Learning rate schedule

### Key Metrics to Watch

| Metric | Good Value | Warning Sign |
|--------|-----------|--------------|
| `train/contrastive_loss` | Decreasing steadily | Flat or increasing |
| `val/contrastive_loss` | < train loss, decreasing | Diverging from train |
| `train/load_balancing_loss` | Small (< 0.05) | Large (> 0.2) indicates mode collapse |

---

## Evaluating Results

### Checkpoint Location

Best model checkpoint is saved to:
```
checkpoints/01_stage_easy/naics-epoch=XX-val_contrastive_loss=Y.YYYY.ckpt
```

### Load and Use Your Model

```python
import torch
from naics_gemini.model.naics_model import NAICSContrastiveModel

# Load checkpoint
checkpoint_path = "checkpoints/01_stage_easy/naics-epoch=04-val_contrastive_loss=2.3456.ckpt"
model = NAICSContrastiveModel.load_from_checkpoint(checkpoint_path)
model.eval()

# TODO: Add inference example when inference API is ready
```

### Quick Quality Check

A well-trained model should:
1. **Cluster siblings closely:** Codes like 541511 and 541512 have small Lorentzian distance
2. **Separate sectors:** Codes from different sectors (e.g., 11 vs 62) have large distance
3. **Preserve hierarchy:** Parent-child relationships reflected in distance

---

## Next Steps

### 1. Progress to Harder Curricula

Once the easy curriculum trains successfully:

```bash
# Medium difficulty: Levels 5-6, mixed hardness
uv run naics-gemini train --curriculum 02_stage_medium

# Hard difficulty: All levels, including exclusions
uv run naics-gemini train --curriculum 03_stage_hard
```

See [Curriculum Design Guide](curriculum_design_guide.md) for details.

### 2. Customize Training

Override any configuration parameter:

```bash
# Train longer
uv run naics-gemini train -c 01_stage_easy training.trainer.max_epochs=10

# Increase batch size (if you have GPU memory)
uv run naics-gemini train -c 01_stage_easy data.batch_size=64

# Adjust learning rate
uv run naics-gemini train -c 01_stage_easy training.learning_rate=5e-5
```

### 3. Create Custom Curriculum

Copy and modify an existing curriculum:

```bash
cp conf/curriculum/01_stage_easy.yaml conf/curriculum/my_custom.yaml
# Edit my_custom.yaml
uv run naics-gemini train -c my_custom
```

See [Curriculum Design Guide](curriculum_design_guide.md) for configuration options.

### 4. Distributed Training

For multi-GPU training:

```bash
uv run naics-gemini train -c 01_stage_easy \
  training.trainer.devices=2 \
  training.trainer.strategy=ddp
```

---

## Common Issues

### 1. Out of Memory (OOM)

**Symptom:** `CUDA out of memory` or process killed

**Solutions:**
```bash
# Reduce batch size
uv run naics-gemini train -c 01_stage_easy data.batch_size=16

# Reduce number of negatives
uv run naics-gemini train -c 01_stage_easy curriculum.k_negatives=8

# Use gradient accumulation
uv run naics-gemini train -c 01_stage_easy \
  training.trainer.accumulate_grad_batches=2
```

### 2. Data Pipeline Fails

**Symptom:** Download errors or missing files

**Solution:** Retry individual steps:
```bash
# If download fails, retry just that step
uv run naics-gemini data preprocess

# Check network access to Census Bureau
curl -I https://www.census.gov/naics/2022NAICS/2-6%20digit_2022_Codes.xlsx
```

### 3. Loss is NaN

**Symptom:** `train/contrastive_loss = nan`

**Causes:**
- Numerical instability in hyperbolic projections
- Learning rate too high
- Corrupted checkpoint

**Solutions:**
```bash
# Reduce learning rate
uv run naics-gemini train -c 01_stage_easy training.learning_rate=5e-5

# Start fresh (delete checkpoints)
rm -rf checkpoints/01_stage_easy/
```

### 4. MoE Load Imbalance

**Symptom:** `train/load_balancing_loss > 0.2` (mode collapse)

**Explanation:** All tokens routing to 1-2 experts

**Solutions:**
```bash
# Increase load balancing coefficient
uv run naics-gemini train -c 01_stage_easy \
  model.moe.load_balancing_coef=0.02

# Or disable MoE temporarily for debugging
uv run naics-gemini train -c 01_stage_easy \
  model.moe.enabled=false
```

### 5. Slow Training

**Symptom:** < 1 batch/second

**Causes:**
- CPU bottleneck in data loading
- Inefficient negative sampling

**Solutions:**
```bash
# Increase data workers
uv run naics-gemini train -c 01_stage_easy data.num_workers=8

# Enable persistent workers
uv run naics-gemini train -c 01_stage_easy \
  data.persistent_workers=true
```

### 6. Validation Loss Not Decreasing

**Symptom:** Train loss decreases, val loss flat

**Possible causes:**
- Overfitting to easy examples
- Curriculum too restrictive
- Need to progress to harder curriculum

**Solutions:**
- Check if ready for next curriculum stage
- Reduce model capacity (lower LoRA rank)
- Add more positives to curriculum

---

## Getting Help

- **Documentation:** See [docs/](../docs/) for detailed guides
- **Issues:** Open an issue on [GitHub](https://github.com/lowmason/naics-gemini/issues)
- **Architecture:** Read [architecture.md](architecture.md) for system design
- **Troubleshooting:** See [troubleshooting.md](troubleshooting.md) for detailed debugging

---

## Summary Checklist

- [ ] Installation verified with `backend.py`
- [ ] Data pipeline completed: 3 Parquet files in `data/`
- [ ] First training run started with `01_stage_easy`
- [ ] TensorBoard monitoring shows decreasing loss
- [ ] Checkpoint saved in `checkpoints/01_stage_easy/`
- [ ] Ready to experiment with custom curricula

**Congratulations!** You've successfully set up NAICS Gemini and trained your first hierarchical embedding model. 🎉
