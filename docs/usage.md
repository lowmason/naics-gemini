# CLI Usage Guide

This guide covers all available CLI commands for the NAICS Embedder system.

## Overview

The NAICS Embedder CLI is organized into three main command groups:

- **`data`** - Data generation and preprocessing commands
- **`tools`** - Utility tools for configuration, GPU optimization, and metrics
- **`train`** / **`train-seq`** - Model training commands

## Installation

The CLI is available as the `naics-embedder` command after installation:

```bash
uv run naics-embedder --help
```

---

## Data Commands

### `data preprocess`

Download and preprocess all raw NAICS data files.

**Generates:** `data/naics_descriptions.parquet`

```bash
uv run naics-embedder data preprocess
```

### `data relations`

Compute pairwise graph relationships between all NAICS codes.

**Requires:** `data/naics_descriptions.parquet`  
**Generates:** `data/naics_relations.parquet`

```bash
uv run naics-embedder data relations
```

### `data distances`

Compute pairwise graph distances between all NAICS codes.

**Requires:** `data/naics_descriptions.parquet`  
**Generates:** `data/naics_distances.parquet`

```bash
uv run naics-embedder data distances
```

### `data triplets`

Generate (anchor, positive, negative) training triplets.

**Requires:** 
- `data/naics_descriptions.parquet`
- `data/naics_distances.parquet`

**Generates:** `data/naics_training_pairs.parquet`

```bash
uv run naics-embedder data triplets
```

### `data all`

Run the full data generation pipeline: preprocess, distances, and triplets.

```bash
uv run naics-embedder data all
```

---

## Tools Commands

### `tools config`

Display current training configuration, including the Structure-Aware Dynamic Curriculum (SADC) schedule.

```bash
uv run naics-embedder tools config
```

**Options:**
- `--config PATH` - Path to base config YAML file (default: `conf/config.yaml`)

```bash
uv run naics-embedder tools config --config conf/config.yaml
```

### `tools gpu`

Optimize training configuration for available GPU memory. Suggests optimal `batch_size` and `accumulate_grad_batches` based on your GPU.

```bash
# Auto-detect GPU memory
uv run naics-embedder tools gpu --auto

# Specify GPU memory manually
uv run naics-embedder tools gpu --gpu-memory 24

# Apply suggested configuration
uv run naics-embedder tools gpu --auto --apply
```

**Options:**
- `--gpu-memory FLOAT` - GPU memory in GB (e.g., 24 for RTX 6000, 80 for A100)
- `--auto` - Auto-detect GPU memory
- `--target-effective-batch INT` - Target effective batch size (default: 256)
- `--apply` - Apply suggested configuration to config files
- `--config PATH` - Path to base config YAML file (default: `conf/config.yaml`)

### `tools visualize`

Visualize training metrics from log files. Creates comprehensive visualizations and analysis of training metrics including:
- Hyperbolic radius over time
- Hierarchy preservation correlations
- Embedding diversity metrics

```bash
uv run naics-embedder tools visualize --stage 02_text
```

**Options:**
- `--stage, -s STR` - Stage name to filter (e.g., `02_text`, default: `02_text`)
- `--log-file PATH` - Path to log file (default: `logs/train_sequential.log`)
- `--output-dir PATH` - Output directory for plots (default: `outputs/visualizations/`)

### `tools investigate`

Investigate why hierarchy preservation correlations might be low. Analyzes ground truth distances, evaluation configuration, and provides recommendations.

```bash
uv run naics-embedder tools investigate
```

**Options:**
- `--distance-matrix PATH` - Path to ground truth distance matrix (default: `data/naics_distance_matrix.parquet`)
- `--config PATH` - Path to config file (default: `conf/config.yaml`)

---

## Training Commands

### `train`

Train the text encoder with the dynamic SADC scheduler defined in `conf/config.yaml`.

```bash
uv run naics-embedder train
```

**Options:**
- `--config PATH` - Path to base config YAML file (default: `conf/config.yaml`)
- `--ckpt-path PATH` - Path to checkpoint file to resume from, or `"last"` to auto-detect the most recent checkpoint
- `--skip-validation` - Skip pre-flight validation of data files and tokenization cache
- `OVERRIDES...` - Config overrides (e.g., `training.learning_rate=1e-4 data_loader.batch_size=64 curriculum.phase2_end=0.65`)

**Examples:**

```bash
# Train with defaults
uv run naics-embedder train

# Train with config overrides
uv run naics-embedder train training.learning_rate=1e-4 data_loader.batch_size=32

# Resume from last checkpoint
uv run naics-embedder train --ckpt-path last

# Adjust the SADC schedule without editing YAML
uv run naics-embedder train curriculum.phase1_end=0.25 curriculum.phase2_end=0.6
```

### `train-seq`

Legacy sequential training retained for backward compatibility. The dynamic SADC scheduler replaces stage-based YAMLs; use this command only if you must reproduce an older multi-stage workflow.

```bash
uv run naics-embedder train-seq --legacy --num-stages 3
```

**Options:**
- `--legacy` - Required acknowledgement to run the deprecated workflow
- `--num-stages INT` - Number of sequential stages to execute
- `--config PATH` - Path to base config YAML file (default: `conf/config.yaml`)
- `OVERRIDES...` - Config overrides applied to every stage

**Examples:**

```bash
# Reproduce a historical 3-stage run
uv run naics-embedder train-seq --legacy --num-stages 3
```

---

## Common Workflows

### Complete Data Pipeline

```bash
# Generate all required data files
uv run naics-embedder data all
```

### Standard Training

```bash
# Train with the default dynamic curriculum
uv run naics-embedder train
```

### Dynamic SADC Training

```bash
# Train with the dynamic curriculum defined in conf/config.yaml
uv run naics-embedder train
```

### View Configuration

```bash
# Display current configuration
uv run naics-embedder tools config
```

### Analyze Training Metrics

```bash
# Visualize training metrics
uv run naics-embedder tools visualize --stage 02_text

# Investigate hierarchy preservation issues
uv run naics-embedder tools investigate
```

---

## Getting Help

For help on any command, use the `--help` flag:

```bash
uv run naics-embedder --help
uv run naics-embedder data --help
uv run naics-embedder tools --help
uv run naics-embedder train --help
```

---

## Configuration Files

The CLI reads a single configuration in `conf/config.yaml`:

- **Base Config:** Paths, model hyperparameters, and trainer settings
- **Curriculum:** `curriculum.*` fields configure SADC phase boundaries and false-negative elimination cadence

See the [Configuration Documentation](api/config.md) for details on configuration structure.

