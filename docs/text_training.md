# Training Guide

This guide explains how to train the NAICS Hyperbolic Embedding System with the dynamic Structure-Aware Dynamic Curriculum (SADC) scheduler. The current workflow uses a single configuration file (`conf/config.yaml`) to control model, data, trainer, and curriculum settings.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Training Command](#training-command)
3. [Dynamic SADC Schedule](#dynamic-sadc-schedule)
4. [Resuming from Checkpoints](#resuming-from-checkpoints)
5. [Configuration Reference](#configuration-reference)
6. [Common Workflows](#common-workflows)
7. [Troubleshooting](#troubleshooting)
8. [Distributed Training](#distributed-training)

---

## Quick Start

Run the recommended dynamic curriculum training loop:

```bash
uv run naics-embedder train
```

Inspect the active configuration, including SADC phase boundaries:

```bash
uv run naics-embedder tools config
```

Apply ad-hoc overrides without editing YAML:

```bash
uv run naics-embedder train \
  training.learning_rate=1e-4 \
  data_loader.batch_size=16 \
  curriculum.phase2_end=0.65
```

---

## Training Command

The `train` command orchestrates data loading, dynamic curriculum scheduling, and model optimization. It reads `conf/config.yaml` and applies any command-line overrides.

```bash
uv run naics-embedder train [OPTIONS] [OVERRIDES...]
```

**Options**

- `--config PATH` — Path to the base config (default: `conf/config.yaml`).
- `--ckpt-path PATH` — Resume from a checkpoint (`last` auto-detects the latest checkpoint in `checkpoints/<experiment_name>`).
- `--skip-validation` — Skip pre-flight validation of data files and tokenization cache.
- `OVERRIDES...` — Dot-notation overrides (e.g., `training.learning_rate=5e-5 curriculum.phase1_end=0.25`).

**Examples**

```bash
# Resume from the last checkpoint
uv run naics-embedder train --ckpt-path last

# Start fresh with a shorter run
uv run naics-embedder train training.trainer.max_epochs=8

# Run with a different experiment name and output folder
uv run naics-embedder train \
  experiment_name=sadc_ablation \
  dirs.output_dir=./outputs/ablation
```

> **Legacy sequential training**: The `train-seq` command is kept for historical multi-stage experiments. Modern runs should rely on the single dynamic SADC schedule.

---

## Dynamic SADC Schedule

SADC is a three-phase scheduler embedded in `curriculum.*` fields within `conf/config.yaml`:

- **Phase 1: Structural Initialization (`phase1_end`)**
  - Masks siblings and weights negatives by inverse tree distance (`tree_distance_alpha`, `sibling_distance_threshold`).
- **Phase 2: Geometric Refinement (`phase2_end`)**
  - Enables Lorentzian hard-negative mining and router-guided sampling for the MoE encoder.
- **Phase 3: False Negative Mitigation (`phase3_end`)**
  - Activates clustering-based false-negative elimination, refreshed every `fn_cluster_every_n_epochs` epochs using `fn_num_clusters` clusters, starting at `fn_curriculum_start_epoch`.

Adjust these boundaries or cadences via YAML or command-line overrides to tailor the schedule to your run length.

---

## Resuming from Checkpoints

Checkpoints are stored under `checkpoints/<experiment_name>/`.

- **Auto-resume**: `--ckpt-path last` loads the most recent checkpoint for the configured experiment.
- **Explicit path**: Point `--ckpt-path` to any `.ckpt` file to warm-start a new run.
- **Fresh start**: Omit `--ckpt-path` to ignore existing checkpoints.

---

## Configuration Reference

All settings live in `conf/config.yaml`:

- **Experiment**: `experiment_name`, `seed`.
- **Curriculum**: `curriculum.phase{1,2,3}_end`, `tree_distance_alpha`, `sibling_distance_threshold`, `fn_curriculum_start_epoch`, `fn_cluster_every_n_epochs`, `fn_num_clusters`.
- **Paths**: `dirs.*` for data, logs, outputs, checkpoints, and docs.
- **Data Loader**: Batch size, workers, validation split, and tokenization/streaming inputs.
- **Model**: Base encoder, LoRA, and MoE settings; evaluation cadence.
- **Loss**: Temperature, curvature, margins, and hierarchy/rank/radius regularization weights.
- **Training**: Optimizer hyperparameters and Lightning trainer settings.

Use `tools config` to render a condensed view of these values.

---

## Common Workflows

**Complete data pipeline then train**

```bash
uv run naics-embedder data all
uv run naics-embedder train
```

**Memory-aware tuning**

```bash
uv run naics-embedder tools gpu --auto
uv run naics-embedder train data_loader.batch_size=<suggested>
```

**Adjust curriculum boundaries for shorter runs**

```bash
uv run naics-embedder train \
  training.trainer.max_epochs=8 \
  curriculum.phase1_end=0.2 \
  curriculum.phase2_end=0.55
```

---

## Troubleshooting

- **Out of memory**: Lower `data_loader.batch_size`, increase `training.trainer.accumulate_grad_batches`, or reduce `fn_num_clusters` to lighten Phase 3 clustering.
- **Slow convergence**: Shorten `fn_curriculum_start_epoch` so false-negative mitigation starts earlier; reduce `training.learning_rate` if loss oscillates.
- **Scheduler transitions**: If phases feel too short or long, adjust `curriculum.phase*` boundaries so each phase covers the desired fraction of epochs.

---

## Distributed Training

Enable multi-GPU runs by updating the trainer configuration:

```yaml
training:
  trainer:
    devices: 4
    accelerator: gpu
    precision: "16-mixed"
```

Or via overrides:

```bash
uv run naics-embedder train training.trainer.devices=4 training.trainer.accelerator=gpu
```

When distributed training is active, the model gathers negatives across GPUs for richer hard-negative mining and logs memory metrics (`train/global_batch/*`) to TensorBoard.
