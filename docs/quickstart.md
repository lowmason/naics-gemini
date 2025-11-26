# Quickstart Guide

This guide provides a quick introduction to the NAICS Embedder command-line interface.
For complete command reference, see the [CLI Usage Guide](usage.md).

## Installation

After cloning the repository, install dependencies with `uv`:

```bash
git clone https://github.com/lowmason/naics-embedder.git
cd naics-embedder
uv sync
```

Verify the installation:

```bash
uv run naics-embedder --help
```

---

## Common Workflows

### 1. Prepare Training Data

Generate all required data files for training:

```bash
uv run naics-embedder data all
```

This runs the complete data pipeline:

1. **Preprocess** - Downloads and cleans raw NAICS taxonomy files
2. **Relations** - Computes pairwise graph relationships
3. **Distances** - Computes pairwise tree distances  
4. **Triplets** - Generates training triplets for contrastive learning

!!! tip "First-time setup"
    The data pipeline only needs to run once. Generated files are cached in `data/`.

### 2. Train the Model

Start training with default configuration:

```bash
uv run naics-embedder train
```

**Common options:**

```bash
# Resume from last checkpoint
uv run naics-embedder train --ckpt-path last

# Override hyperparameters
uv run naics-embedder train training.learning_rate=1e-5 data_loader.batch_size=16

# Skip validation (when you know data is valid)
uv run naics-embedder train --skip-validation
```

### 3. Monitor Training

View current configuration:

```bash
uv run naics-embedder tools config
```

Visualize training metrics:

```bash
uv run naics-embedder tools visualize --stage 02_text
```

Investigate low hierarchy preservation:

```bash
uv run naics-embedder tools investigate
```

---

## Command Groups

The CLI is organized into three main groups:

| Group | Description | Example |
|-------|-------------|---------|
| `data` | Data generation and preprocessing | `data all`, `data preprocess` |
| `tools` | Configuration and metrics utilities | `tools config`, `tools visualize` |
| `train` | Model training (main command) | `train`, `train --ckpt-path last` |

Use `--help` on any command for detailed options:

```bash
uv run naics-embedder data --help
uv run naics-embedder train --help
uv run naics-embedder tools --help
```

---

## Quick Reference

### Data Commands

| Command | Description |
|---------|-------------|
| `data all` | Run complete data pipeline |
| `data preprocess` | Download and preprocess NAICS files |
| `data relations` | Compute pairwise relationships |
| `data distances` | Compute pairwise distances |
| `data triplets` | Generate training triplets |

### Training Commands

| Command | Description |
|---------|-------------|
| `train` | Train with current configuration |
| `train --ckpt-path last` | Resume from last checkpoint |
| `train-seq --legacy` | Sequential training (deprecated) |

### Tools Commands

| Command | Description |
|---------|-------------|
| `tools config` | Display current configuration |
| `tools visualize` | Visualize training metrics |
| `tools investigate` | Analyze hierarchy preservation |

---

## Configuration Overrides

Override any configuration value at runtime using dot notation:

```bash
uv run naics-embedder train \
    training.learning_rate=1e-4 \
    training.trainer.max_epochs=20 \
    data_loader.batch_size=32 \
    loss.hierarchy_weight=0.2
```

Common overrides:

| Parameter | Description |
|-----------|-------------|
| `training.learning_rate` | Optimizer learning rate |
| `training.trainer.max_epochs` | Maximum training epochs |
| `data_loader.batch_size` | Training batch size |
| `loss.hierarchy_weight` | Weight for hierarchy loss |
| `loss.temperature` | Contrastive loss temperature |

---

## Output Files

After training, find outputs in:

| Path | Contents |
|------|----------|
| `checkpoints/<experiment>/` | Model checkpoints and config |
| `checkpoints/<experiment>/training_summary.yaml` | Training results summary |
| `outputs/<experiment>/` | TensorBoard logs |
| `logs/train.log` | Detailed training log |

---

## Next Steps

- [CLI Usage Guide](usage.md) - Complete command reference
- [Text Training Guide](text_training.md) - Detailed training documentation
- [Configuration Reference](api/config.md) - All configuration options
