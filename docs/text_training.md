# Training Guide

This guide explains how to train the NAICS Hyperbolic Embedding System with the dynamic
Structure-Aware Dynamic Curriculum (SADC) scheduler. Training now happens in a single run—the
scheduler flips curriculum flags automatically, so you no longer need stage files or chain
configs.

## Table of Contents

1. [Quick Start](#quick-start)
2. [SADC Scheduler](#sadc-scheduler)
3. [CLI Reference](#cli-reference)
4. [Resuming and Overrides](#resuming-and-overrides)
5. [Migration for Legacy Chains](#migration-for-legacy-chains)
6. [Troubleshooting](#troubleshooting)

---

## Quick Start

Preprocess data first (see `docs/usage.md`), then launch training with the default config:

```bash
uv run naics-embedder train --config conf/config.yaml
```

Apply overrides inline—SADC will stay active and adapt phases automatically:

```bash
uv run naics-embedder train --config conf/config.yaml \
  training.learning_rate=1e-4 training.trainer.max_epochs=20
```

---

## SADC Scheduler

The scheduler runs three phases within a single training invocation:

1. **Structural Initialization (0–30%)**
   - Flags: `use_tree_distance`, `mask_siblings`
   - Effect: weights negatives by inverse tree distance and masks siblings.
2. **Geometric Refinement (30–70%)**
   - Flags: `enable_hard_negative_mining`, `enable_router_guided_sampling`
   - Effect: activates Lorentzian hard-negative mining and router-guided MoE sampling.
3. **False Negative Mitigation (70–100%)**
   - Flags: `enable_clustering`
   - Effect: enables clustering-driven false-negative elimination.

Phase boundaries are derived from the trainer's `max_epochs`. Flag transitions are logged to help
verify when each mechanism is active.

---

## CLI Reference

Use the `train` command for all new runs:

```bash
uv run naics-embedder train --config conf/config.yaml
```

Key options:
- `--config PATH` — Base config file (default: `conf/config.yaml`).
- `--ckpt-path PATH` — Resume from a checkpoint or use `last` to pick the most recent run artifact.
- `--skip-validation` — Bypass pre-flight validation when inputs are already verified.
- `OVERRIDES...` — Space-separated config overrides (e.g., `training.learning_rate=1e-4`).

---

## Resuming and Overrides

Resume the latest checkpoint produced under the current experiment name:

```bash
uv run naics-embedder train --ckpt-path last
```

Override trainer settings without editing YAML:

```bash
uv run naics-embedder train \
  training.trainer.max_epochs=15 training.trainer.accumulate_grad_batches=4
```

---

## Migration for Legacy Chains

The legacy stage-by-stage curriculum files and chain configs are retired. To reproduce an old
multi-stage job, acknowledge the deprecated workflow explicitly:

```bash
uv run naics-embedder train-seq --legacy --num-stages 3 --config conf/config.yaml
```

New work should rely on `train` plus overrides—the dynamic SADC scheduler replaces manual chains and
static curriculum files.

---

## Troubleshooting

- **Dataset checks** — Use `uv run naics-embedder tools config` to confirm paths before training.
- **Flag visibility** — Curriculum phase transitions and flag values are emitted in training logs.
- **Memory pressure** — Lower `data_loader.batch_size` or increase `accumulate_grad_batches` via
  overrides.
