# Training Guide

This guide covers how to train the NAICS Hyperbolic Embedding System using the contrastive learning pipeline. The system supports single-stage training, sequential curriculum training, and chain-based training with checkpoint resumption.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Single-Stage Training](#single-stage-training)
3. [Sequential Curriculum Training](#sequential-curriculum-training)
4. [Chain Configuration](#chain-configuration)
5. [Resuming from Checkpoints](#resuming-from-checkpoints)
6. [Configuration Files](#configuration-files)
7. [Common Workflows](#common-workflows)
8. [Troubleshooting](#troubleshooting)

---

## Quick Start

### List Available Curricula

```bash
uv run naics-embedder train --list-curricula
```

### Run a Single Training Stage

```bash
uv run naics-embedder train --curriculum 01_stage
```

### Run Sequential Training with Chain Config

```bash
uv run naics-embedder train-curriculum --chain chain_default
```

---

## Single-Stage Training

Train a single curriculum stage with a specific configuration.

### Basic Usage

```bash
uv run naics-embedder train \
  --curriculum 01_stage \
  --config conf/config.yaml
```

### With Configuration Overrides

Override specific parameters without editing config files:

```bash
uv run naics-embedder train \
  --curriculum 01_stage \
  training.learning_rate=1e-4 \
  data_loader.batch_size=8 \
  training.trainer.max_epochs=15
```

### Command Options

- `--curriculum, -c`: Curriculum config name (e.g., `01_stage`, `default`)
- `--config`: Path to base config YAML file (default: `conf/config.yaml`)
- `--list-curricula`: List all available curriculum configs and exit
- Overrides: Space-separated key-value pairs using dot notation

### Example: Training with Custom Learning Rate

```bash
uv run naics-embedder train \
  --curriculum 02_stage \
  training.learning_rate=5e-5 \
  training.trainer.max_epochs=20
```

---

## Sequential Curriculum Training

Train multiple curriculum stages sequentially, with each stage building on the previous one's learned representations.

### Basic Usage

Train all default stages (01-05) sequentially:

```bash
uv run naics-embedder train-curriculum
```

### Specify Stages Manually

```bash
uv run naics-embedder train-curriculum \
  --curricula 01_stage,02_stage,03_stage
```

### With Custom Config

```bash
uv run naics-embedder train-curriculum \
  --curricula 01_stage,02_stage,03_stage \
  --config conf/config.yaml
```

### Command Options

- `--curricula, -c`: Comma-separated list of curriculum stages
- `--chain`: Chain configuration file (overrides `--curricula`)
- `--config`: Path to base config YAML file
- `--resume`: Resume from last checkpoint if available (default: `true`)
- `--from-checkpoint`: Resume from a specific checkpoint (see [Resuming from Checkpoints](#resuming-from-checkpoints))

### How Sequential Training Works

1. **Stage 1**: Trains from scratch (random initialization)
2. **Stage 2**: Loads the best checkpoint from Stage 1
3. **Stage 3**: Loads the best checkpoint from Stage 2
4. And so on...

Each stage uses the **learned encoder weights** from the previous stage, allowing progressive refinement of the embedding space.

---

## Chain Configuration

Chain configurations allow you to define a sequence of stages with stage-specific training parameter overrides (e.g., `max_epochs`, `learning_rate`).

### Chain Config Format

Create a chain config file in `conf/text_curriculum/`:

```yaml
chain_name: progressive_difficulty

stages:
  - name: 01_stage
    max_epochs: 5
    learning_rate: 2e-4
  - name: 02_stage  
    max_epochs: 8
    learning_rate: 1e-4
  - name: 03_stage
    max_epochs: 10
    learning_rate: 5e-5
```

### Using a Chain Config

```bash
uv run naics-embedder train-curriculum --chain chain_default
```

This will:

1. Load the chain configuration
2. Run stages in order: `01_stage`, `02_stage`, `03_stage`
3. Apply stage-specific overrides:
   - Stage 01: `max_epochs=5`, `learning_rate=2e-4`
   - Stage 02: `max_epochs=8`, `learning_rate=1e-4`
   - Stage 03: `max_epochs=10`, `learning_rate=5e-5`

### Chain Config Fields

- `chain_name`: Name of the training chain (for logging/identification)
- `stages`: List of stage configurations
  - `name`: Curriculum stage name (must match a file in `conf/text_curriculum/`)
  - `max_epochs` (optional): Override max epochs for this stage
  - `learning_rate` (optional): Override learning rate for this stage

**Note**: If `max_epochs` or `learning_rate` are not specified, the values from the base config or curriculum config are used.

---

## Resuming from Checkpoints

The training system supports resuming from checkpoints in several ways:

### 1. Automatic Resumption (Default)

When running sequential training, the system automatically resumes from the previous stage's checkpoint:

```bash
uv run naics-embedder train-curriculum --curricula 01_stage,02_stage,03_stage
```

Stage 2 will automatically load Stage 1's best checkpoint, and Stage 3 will load Stage 2's best checkpoint.

### 2. Resume from Specific Stage

Resume training from a specific stage's checkpoint by stage name:

```bash
uv run naics-embedder train-curriculum \
  --curricula 04_stage \
  --from-checkpoint 03_stage
```

This will:

- Search for the best checkpoint from `03_stage` in any previous sequential run
- Use that checkpoint to initialize training for `04_stage`

### 3. Resume from Specific Checkpoint File

Resume from an exact checkpoint file path:

```bash
uv run naics-embedder train-curriculum \
  --curricula 04_stage \
  --from-checkpoint checkpoints/sequential_01-02-03/03_stage/03_stage-epoch=10-val_loss=0.1234.ckpt
```

### 4. Re-run a Stage with Different Parameters

Re-train a stage starting from its own previous checkpoint:

```bash
# First, modify 03_stage.yaml or use overrides
uv run naics-embedder train-curriculum \
  --curricula 03_stage \
  --from-checkpoint 03_stage \
  training.learning_rate=1e-5
```

### Checkpoint Discovery

When using `--from-checkpoint` with a stage name (not a file path), the system:

1. Searches all `sequential_*/` directories in the checkpoint folder
2. Finds checkpoints from the specified stage
3. Selects the checkpoint with the **lowest validation loss** (best model)
4. Uses that checkpoint to resume training

### Checkpoint Directory Structure

Checkpoints are organized as follows:

```
checkpoints/
  sequential_01-02-03/          # Sequential run identifier
    01_stage/
      01_stage-epoch=05-val_loss=0.1234.ckpt
      last.ckpt
    02_stage/
      02_stage-epoch=08-val_loss=0.0987.ckpt
      last.ckpt
    03_stage/
      03_stage-epoch=10-val_loss=0.0876.ckpt
      last.ckpt
```

---

## Configuration Files

### Base Configuration (`conf/config.yaml`)

Main training configuration including:

- Model architecture (LoRA, MoE, etc.)
- Training hyperparameters (learning rate, weight decay, etc.)
- Data loader settings (batch size, num workers, etc.)
- Trainer settings (max epochs, precision, etc.)

### Curriculum Configuration (`conf/text_curriculum/*.yaml`)

Stage-specific curriculum settings:

- `name`: Curriculum name
- `anchor_level`: Filter anchors by hierarchy level
- `positive_level`: Filter positives by hierarchy level
- `positive_relation`: Filter positives by relation type
- `n_positives`: Number of positive samples per anchor
- `n_negatives`: Number of negative samples per anchor

### Example Curriculum Config

```yaml
name: 01_stage

# Anchor filtering
anchor_level: [2, 3]
relation_margin: null
distance_margin: null

# Positive filtering
positive_level: [2, 3]
positive_relation: [1, 2]
positive_distance: null

# Negative sampling
negative_level: null
negative_relation: null
negative_distance: null

# Sample counts
n_positives: 64
n_negatives: 24
```

---

## Common Workflows

### Workflow 1: Full Chain Training

Train a complete chain from scratch:

```bash
uv run naics-embedder train-curriculum --chain chain_default
```

### Workflow 2: Continue Training After a Chain

After running `chain_default` (stages 01-03), continue with stage 04:

```bash
uv run naics-embedder train-curriculum \
  --curricula 04_stage \
  --from-checkpoint 03_stage
```

### Workflow 3: Experiment with Different Parameters

Re-run a stage with different learning rate:

```bash
uv run naics-embedder train-curriculum \
  --curricula 02_stage \
  --from-checkpoint 01_stage \
  training.learning_rate=5e-5 \
  training.trainer.max_epochs=15
```

### Workflow 4: Custom Sequential Run

Run a custom sequence of stages:

```bash
uv run naics-embedder train-curriculum \
  --curricula 01_stage,03_stage,05_stage
```

### Workflow 5: Single Stage with Overrides

Train a single stage with parameter overrides:

```bash
uv run naics-embedder train \
  --curriculum 01_stage \
  training.learning_rate=1e-4 \
  data_loader.batch_size=16 \
  training.trainer.max_epochs=10
```

---

## Troubleshooting

### Checkpoint Not Found

**Problem**: `--from-checkpoint` can't find a checkpoint

**Solutions**:

1. Verify the checkpoint file exists:

   ```bash
   ls -la checkpoints/sequential_*/03_stage/*.ckpt
   ```

2. Use the full checkpoint file path instead of stage name
3. Check that the checkpoint directory structure matches expected format

### Stage Already Complete

**Problem**: Training skips a stage because it's already complete

**Solution**: This is expected behavior when `--resume` is enabled. To re-train:

- Delete the checkpoint directory for that stage, or
- Use `--resume false` to force re-training

### Out of Memory

**Problem**: GPU runs out of memory during training

**Solutions**:

1. Reduce batch size: `data_loader.batch_size=4`
2. Increase gradient accumulation: `training.trainer.accumulate_grad_batches=8`
3. Reduce number of positives/negatives in curriculum config
4. Use mixed precision: `training.trainer.precision=16-mixed` (already default)
5. **For Distributed Training**: Monitor global batch memory usage in TensorBoard
   - Check `train/global_batch/global_negatives_memory_mb` metric
   - If memory is high, reduce `n_negatives` in curriculum config

### Learning Rate Too High/Low

**Problem**: Training diverges or converges too slowly

**Solutions**:

1. Use chain config to set stage-specific learning rates
2. Use command-line overrides: `training.learning_rate=1e-5`
3. Adjust in base config file

### Model Not Improving

**Problem**: Validation loss plateaus or increases

**Solutions**:

1. Check curriculum config - may need more diverse positives/negatives
2. Reduce learning rate for later stages
3. Increase training epochs
4. Check data quality and filtering settings

---

## Best Practices

1. **Start with Chain Configs**: Use chain configurations for reproducible multi-stage training
2. **Monitor TensorBoard**: Check training progress and metrics:

   ```bash
   tensorboard --logdir outputs/
   ```

3. **Save Checkpoints**: Always keep checkpoints from each stage for flexibility
4. **Use Stage-Specific Learning Rates**: Gradually reduce learning rate across stages
5. **Validate Checkpoints**: Test that checkpoints load correctly before long training runs
6. **Document Experiments**: Keep notes on which configs/checkpoints produced best results

---

## Distributed Training

The system supports multi-GPU distributed training with automatic global batch sampling for improved hard negative mining.

### Enabling Distributed Training

Configure the number of GPUs in your config file:

```yaml
training:
  trainer:
    devices: 4  # Number of GPUs to use
    accelerator: 'gpu'
```

Or override via command line:

```bash
uv run naics-embedder train \
  --curriculum 01_text \
  training.trainer.devices=4
```

### Global Batch Sampling

When distributed training is enabled with hard negative mining, the system automatically:

1. **Gathers negatives from all GPUs** using `torch.distributed.all_gather`
2. **Selects hard negatives from the global pool** (not just local batch)
3. **Preserves gradient flow** through the all_gather operation
4. **Monitors memory usage** and logs metrics to TensorBoard

**Benefits:**

- **Larger Negative Pool**: Access to negatives from all GPUs
- **Better Hard Negatives**: More likely to find meaningful "Cousin" relationships
- **Improved Training**: Higher quality negative samples lead to better representations

**Memory Considerations:**

- Global negatives: ~9MB per GPU (for batch_size=32, world_size=4, k_negatives=24)
- Similarity matrix: ~393KB per batch
- Monitor via TensorBoard: `train/global_batch/global_negatives_memory_mb`

### Monitoring Distributed Training

TensorBoard provides metrics for distributed training:

- `train/global_batch/global_batch_size`: Effective global batch size
- `train/global_batch/global_k_negatives`: Number of negatives per anchor globally
- `train/global_batch/global_negatives_memory_mb`: Memory usage for global negatives
- `train/global_batch/similarity_matrix_memory_mb`: Memory usage for similarity matrix
- `train/global_batch/global_hard_negatives_used`: Whether global hard negatives are active

---