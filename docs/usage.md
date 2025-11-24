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

Display current training and curriculum configuration.

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

Train a single curriculum stage.

```bash
uv run naics-embedder train --curriculum 01_text
```

**Options:**
- `--curriculum, -c STR` - Curriculum config name (e.g., `01_text`, `02_text`, `01_graph`, `02_graph`, default: `default`)
- `--curriculum-type STR` - Curriculum type: `text` or `graph` (default: `text`)
- `--config PATH` - Path to base config YAML file (default: `conf/config.yaml`)
- `--list-curricula` - List available curricula and exit
- `--ckpt-path PATH` - Path to checkpoint file to resume from, or `"last"` to auto-detect last checkpoint
- `OVERRIDES...` - Config overrides (e.g., `training.learning_rate=1e-4 data.batch_size=64`)

**Examples:**

```bash
# List available curricula
uv run naics-embedder train --list-curricula

# Train with a specific curriculum
uv run naics-embedder train --curriculum 01_text

# Train with config overrides
uv run naics-embedder train --curriculum 01_text training.learning_rate=1e-4 data.batch_size=32

# Resume from last checkpoint
uv run naics-embedder train --curriculum 01_text --ckpt-path last

# Train graph curriculum
uv run naics-embedder train --curriculum 01_graph --curriculum-type graph
```

### `train-seq`

Run sequential curriculum training with automatic checkpoint handoff. Trains through multiple curriculum stages, automatically loading the best checkpoint from each stage as the initialization for the next stage.

```bash
uv run naics-embedder train-seq --curricula 01_text 02_text 03_text
```

**Options:**
- `--curricula, -c LIST` - List of curriculum stages to run sequentially (e.g., `01_text 02_text 03_text`)
- `--curriculum-type STR` - Curriculum type: `text` or `graph` (default: `text`)
- `--config PATH` - Path to base config YAML file (default: `conf/config.yaml`)
- `--resume` - Resume from last checkpoint if available
- `OVERRIDES...` - Config overrides (e.g., `training.learning_rate=1e-4`)

**Examples:**

```bash
# Train default sequence (01_text, 02_text, 03_text)
uv run naics-embedder train-seq

# Train custom sequence
uv run naics-embedder train-seq --curricula 01_text 02_text 03_text 04_text 05_text

# Train with overrides applied to all stages
uv run naics-embedder train-seq --curricula 01_text 02_text training.learning_rate=1e-4

# Resume from last checkpoint
uv run naics-embedder train-seq --resume
```

---

## Common Workflows

### Complete Data Pipeline

```bash
# Generate all required data files
uv run naics-embedder data all
```

### Single Stage Training

```bash
# Train a single curriculum stage
uv run naics-embedder train --curriculum 01_text
```

### Sequential Multi-Stage Training

```bash
# Train multiple stages sequentially
uv run naics-embedder train-seq --curricula 01_text 02_text 03_text 04_text 05_text
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

The CLI uses configuration files located in the `conf/` directory:

- **Base Config:** `conf/config.yaml` - Main training configuration
- **Text Curricula:** `conf/text_curriculum/*.yaml` - Text training curriculum stages
- **Graph Curricula:** `conf/graph_curriculum/*.yaml` - Graph training curriculum stages
- **Chain Configs:** `conf/text_curriculum/chain_text.yaml` - Sequential training chains

See the [Configuration Documentation](api/config.md) for details on configuration structure.

