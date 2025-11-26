# Training Utilities

Helper functions for training orchestration and result collection.

## Overview

The `training` utilities module provides reusable components for the training
workflow, including hardware detection, configuration parsing, checkpoint
management, and summary artifact generation.

## Usage

```python
from naics_embedder.utils.training import (
    detect_hardware,
    parse_config_overrides,
    resolve_checkpoint,
    save_training_summary,
)

# Detect hardware
hardware = detect_hardware(log_info=True)
print(f"Training on {hardware.accelerator}")

# Parse overrides
overrides, invalid = parse_config_overrides(['lr=1e-4', 'epochs=10'])

# Resolve checkpoint
checkpoint_info = resolve_checkpoint('last', Path('checkpoints'), 'experiment')
```

## Data Classes

### HardwareInfo

Container for detected hardware configuration.

### CheckpointInfo

Resolved checkpoint path and metadata.

### TrainingResult

Structured result from a completed training run.

## API Reference

::: naics_embedder.utils.training
    options:
      show_source: false
      members:
        - HardwareInfo
        - CheckpointInfo
        - TrainingResult
        - detect_hardware
        - get_gpu_memory_info
        - parse_config_overrides
        - resolve_checkpoint
        - create_trainer
        - collect_training_result
        - save_training_summary

