# -------------------------------------------------------------------------------------------------
# Training Utilities
# -------------------------------------------------------------------------------------------------
'''
Training orchestration utilities for NAICS Embedder.

This module provides helper functions to simplify training setup by extracting
common operations into reusable components. These utilities handle hardware
detection, configuration parsing, checkpoint management, and trainer creation.

Functions:
    detect_hardware: Detect available accelerators and optimal precision.
    get_gpu_memory_info: Query current GPU memory usage.
    parse_config_overrides: Parse and validate command-line config overrides.
    resolve_checkpoint: Resolve checkpoint path from user input.
    create_trainer: Create a configured PyTorch Lightning Trainer.
    TrainingResult: Structured result from a training run.
'''

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytorch_lightning as pyl
import torch
from pytorch_lightning.callbacks import Callback, EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from naics_embedder.utils.backend import get_device
from naics_embedder.utils.config import Config, parse_override_value

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------------------------------
# Data Classes
# -------------------------------------------------------------------------------------------------

@dataclass
class HardwareInfo:
    '''
    Hardware configuration detected for training.

    Attributes:
        accelerator: The accelerator type (cuda, mps, cpu).
        precision: Recommended precision setting (16-mixed, 32-true).
        num_devices: Number of available devices.
        gpu_memory: Optional GPU memory information dictionary.
    '''

    accelerator: str
    precision: str
    num_devices: int
    gpu_memory: Optional[Dict[str, float]] = None

@dataclass
class CheckpointInfo:
    '''
    Resolved checkpoint information.

    Attributes:
        path: Resolved filesystem path to the checkpoint, or None.
        is_same_stage: Whether checkpoint is from the same experiment stage.
        exists: Whether the checkpoint file exists.
    '''

    path: Optional[str]
    is_same_stage: bool
    exists: bool

@dataclass
class TrainingResult:
    '''
    Structured result from a training run.

    Provides a clean interface for accessing training outputs, metrics, and
    paths for downstream processing or testing.

    Attributes:
        best_checkpoint_path: Path to the best model checkpoint.
        last_checkpoint_path: Path to the last model checkpoint.
        config_path: Path to the saved configuration file.
        best_loss: Best validation loss achieved.
        stopped_epoch: Epoch at which training stopped (early stopping or max).
        early_stopped: Whether early stopping was triggered.
        metrics: Dictionary of final metrics.
    '''

    best_checkpoint_path: Optional[str] = None
    last_checkpoint_path: Optional[str] = None
    config_path: Optional[str] = None
    best_loss: Optional[float] = None
    stopped_epoch: int = 0
    early_stopped: bool = False
    metrics: Dict[str, Any] = field(default_factory=dict)

# -------------------------------------------------------------------------------------------------
# Hardware Detection
# -------------------------------------------------------------------------------------------------

def detect_hardware(log_info: bool = False) -> HardwareInfo:
    '''
    Detect available hardware and recommend training settings.

    Queries the system for CUDA, MPS, or CPU availability and returns
    appropriate accelerator and precision settings for PyTorch Lightning.

    Args:
        log_info: If True, log detailed hardware information.

    Returns:
        HardwareInfo with detected accelerator, precision, device count,
        and optional GPU memory information.

    Example:
        >>> hw = detect_hardware(log_info=True)
        >>> print(f'Training on {hw.accelerator} with {hw.precision} precision')
    '''
    accelerator, precision, num_devices = get_device(log_info=log_info)
    gpu_memory = None

    if accelerator in ['cuda', 'gpu'] and torch.cuda.is_available():
        gpu_memory = get_gpu_memory_info()

    return HardwareInfo(
        accelerator=accelerator,
        precision=precision,
        num_devices=num_devices,
        gpu_memory=gpu_memory
    )

def get_gpu_memory_info() -> Optional[Dict[str, float]]:
    '''
    Query current GPU memory usage.

    Returns memory statistics including total, reserved, allocated, and free
    memory in gigabytes, along with utilization percentage.

    Returns:
        Dictionary with memory statistics, or None if CUDA unavailable.

    Example:
        >>> info = get_gpu_memory_info()
        >>> if info:
        ...     print(f'GPU Memory: {info["free_gb"]:.1f} GB free')
    '''
    if not torch.cuda.is_available():
        return None

    try:
        device = torch.cuda.current_device()
        total = torch.cuda.get_device_properties(device).total_memory / (1024**3)
        reserved = torch.cuda.memory_reserved(device) / (1024**3)
        allocated = torch.cuda.memory_allocated(device) / (1024**3)
        free = total - reserved

        return {
            'total_gb': total,
            'reserved_gb': reserved,
            'allocated_gb': allocated,
            'free_gb': free,
            'utilization_pct': (reserved / total) * 100 if total > 0 else 0,
        }
    except Exception as e:
        logger.debug(f'Could not get GPU memory info: {e}')
        return None

# -------------------------------------------------------------------------------------------------
# Configuration Parsing
# -------------------------------------------------------------------------------------------------

def parse_config_overrides(overrides: Optional[List[str]]) -> Tuple[Dict[str, Any], List[str]]:
    '''
    Parse and validate command-line configuration overrides.

    Converts a list of ``key=value`` strings into a dictionary suitable for
    use with ``Config.override()``. Invalid overrides are collected and
    returned for warning messages.

    Args:
        overrides: List of override strings like ``training.learning_rate=1e-4``.

    Returns:
        Tuple of (valid_overrides_dict, list_of_invalid_override_strings).

    Example:
        >>> overrides = ['training.learning_rate=1e-4', 'invalid', 'batch_size=32']
        >>> valid, invalid = parse_config_overrides(overrides)
        >>> print(valid)  # {'training.learning_rate': 0.0001, 'batch_size': 32}
        >>> print(invalid)  # ['invalid']
    '''
    if not overrides:
        return {}, []

    override_dict: Dict[str, Any] = {}
    invalid_overrides: List[str] = []

    for override in overrides:
        if '=' not in override:
            invalid_overrides.append(override)
            continue

        key, value_str = override.split('=', 1)
        value = parse_override_value(value_str)
        override_dict[key] = value
        logger.info(f'  • {key} = {value} ({type(value).__name__})')

    return override_dict, invalid_overrides

# -------------------------------------------------------------------------------------------------
# Checkpoint Resolution
# -------------------------------------------------------------------------------------------------

def resolve_checkpoint(
    ckpt_path: Optional[str], checkpoint_dir: Path, experiment_name: str
) -> CheckpointInfo:
    '''
    Resolve a checkpoint path from user input.

    Handles three cases:
    1. ``None``: No checkpoint specified, start fresh.
    2. ``"last"`` or ``"last.ckpt"``: Auto-detect last checkpoint in experiment dir.
    3. Explicit path: Validate and resolve the provided path.

    Args:
        ckpt_path: User-provided checkpoint path or keyword.
        checkpoint_dir: Base directory for checkpoints.
        experiment_name: Name of the current experiment.

    Returns:
        CheckpointInfo with resolved path and metadata.

    Example:
        >>> info = resolve_checkpoint('last', Path('checkpoints'), '01_text')
        >>> if info.exists:
        ...     print(f'Resuming from {info.path}')
    '''
    if not ckpt_path:
        return CheckpointInfo(path=None, is_same_stage=False, exists=False)

    experiment_dir = checkpoint_dir / experiment_name
    ckpt_path_lower = ckpt_path.lower()

    # Handle 'last' keyword
    if ckpt_path_lower in ('last', 'last.ckpt'):
        last_ckpt = experiment_dir / 'last.ckpt'
        if last_ckpt.exists():
            logger.info(f'Auto-detected last checkpoint: {last_ckpt}')
            return CheckpointInfo(path=str(last_ckpt), is_same_stage=True, exists=True)
        else:
            logger.warning(f'Last checkpoint not found at {last_ckpt}')
            return CheckpointInfo(path=None, is_same_stage=False, exists=False)

    # Handle explicit path
    checkpoint_path_obj = Path(ckpt_path)
    if checkpoint_path_obj.exists():
        resolved_path = str(checkpoint_path_obj.resolve())

        # Determine if this is from the same stage
        try:
            is_same_stage = checkpoint_path_obj.resolve().parent == experiment_dir.resolve()
        except Exception:
            is_same_stage = False

        logger.info(f'Using checkpoint: {resolved_path}')
        return CheckpointInfo(path=resolved_path, is_same_stage=is_same_stage, exists=True)
    else:
        logger.warning(f'Checkpoint not found at {ckpt_path}')
        return CheckpointInfo(path=None, is_same_stage=False, exists=False)

# -------------------------------------------------------------------------------------------------
# Trainer Creation
# -------------------------------------------------------------------------------------------------

def create_trainer(
    cfg: Config,
    hardware: HardwareInfo,
    checkpoint_dir: Path,
    callbacks: Optional[List[Callback]] = None,
    tb_logger: Optional[TensorBoardLogger] = None,
) -> Tuple[pyl.Trainer, ModelCheckpoint, EarlyStopping]:
    '''
    Create a configured PyTorch Lightning Trainer.

    Sets up the trainer with appropriate callbacks, logging, and hardware
    settings based on the provided configuration.

    Args:
        cfg: Training configuration.
        hardware: Detected hardware information.
        checkpoint_dir: Directory for saving checkpoints.
        callbacks: Optional additional callbacks to include.
        tb_logger: Optional TensorBoard logger (created if not provided).

    Returns:
        Tuple of (Trainer, ModelCheckpoint callback, EarlyStopping callback).

    Example:
        >>> hw = detect_hardware()
        >>> trainer, ckpt_cb, es_cb = create_trainer(cfg, hw, Path('checkpoints'))
        >>> trainer.fit(model, datamodule)
    '''
    # Setup checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename='naics-{epoch:02d}-{val/contrastive_loss:.4f}',
        monitor='val/contrastive_loss',
        mode='min',
        save_top_k=3,
        save_last=True,
    )

    # Setup early stopping
    early_stopping = EarlyStopping(
        monitor='val/contrastive_loss', patience=3, mode='min', min_delta=0.0001, verbose=True
    )

    # Setup TensorBoard logger if not provided
    if tb_logger is None:
        tb_log_dir = Path(cfg.dirs.output_dir) / cfg.experiment_name
        tb_log_dir.mkdir(parents=True, exist_ok=True)
        tb_logger = TensorBoardLogger(save_dir=cfg.dirs.output_dir, name=cfg.experiment_name)

    # Combine callbacks
    all_callbacks: List[Callback] = [checkpoint_callback, early_stopping]
    if callbacks:
        all_callbacks.extend(callbacks)

    # Determine devices and strategy
    devices_to_use = getattr(cfg.training.trainer, 'devices', 1)
    strategy = 'auto'

    if devices_to_use > 1 and hardware.accelerator in ['cuda', 'gpu']:
        from pytorch_lightning.strategies import DDPStrategy

        strategy = DDPStrategy(find_unused_parameters=True)

    # Create trainer
    trainer = pyl.Trainer(
        max_epochs=cfg.training.trainer.max_epochs,
        accelerator=hardware.accelerator,
        devices=devices_to_use,
        strategy=strategy,
        precision=hardware.precision,  # type: ignore
        gradient_clip_val=cfg.training.trainer.gradient_clip_val,
        accumulate_grad_batches=cfg.training.trainer.accumulate_grad_batches,
        log_every_n_steps=cfg.training.trainer.log_every_n_steps,
        val_check_interval=cfg.training.trainer.val_check_interval,
        callbacks=all_callbacks,
        logger=tb_logger,
        default_root_dir=cfg.dirs.output_dir,
    )

    return trainer, checkpoint_callback, early_stopping

def collect_training_result(
    checkpoint_callback: ModelCheckpoint,
    early_stopping: EarlyStopping,
    config_path: Optional[str] = None,
) -> TrainingResult:
    '''
    Collect results from a completed training run.

    Gathers checkpoint paths, metrics, and early stopping information into
    a structured result object for downstream processing.

    Args:
        checkpoint_callback: The ModelCheckpoint callback from training.
        early_stopping: The EarlyStopping callback from training.
        config_path: Optional path where config was saved.

    Returns:
        TrainingResult with all training outputs and metrics.

    Example:
        >>> trainer.fit(model, datamodule)
        >>> result = collect_training_result(ckpt_cb, es_cb, 'config.yaml')
        >>> print(f'Best loss: {result.best_loss:.4f}')
    '''
    early_stopped = early_stopping.stopped_epoch > 0
    best_loss = float(early_stopping.best_score) if early_stopping.best_score is not None else None

    # Handle the checkpoint directory path
    ckpt_dir = checkpoint_callback.dirpath
    last_ckpt_path = str(Path(ckpt_dir) / 'last.ckpt') if ckpt_dir else None

    return TrainingResult(
        best_checkpoint_path=checkpoint_callback.best_model_path,
        last_checkpoint_path=last_ckpt_path,
        config_path=config_path,
        best_loss=best_loss,
        stopped_epoch=early_stopping.stopped_epoch if early_stopped else -1,
        early_stopped=early_stopped,
        metrics={'best_val_loss': best_loss},
    )

# -------------------------------------------------------------------------------------------------
# Summary Artifacts
# -------------------------------------------------------------------------------------------------

def save_training_summary(
    result: TrainingResult,
    config: Config,
    hardware: HardwareInfo,
    output_dir: Path,
    format: str = 'both',
) -> Dict[str, str]:
    '''
    Save training summary artifacts for downstream evaluation and documentation.

    Creates YAML and/or JSON summary files containing training results,
    configuration snapshot, and hardware information. These artifacts can
    be used for evaluation scripts, MkDocs documentation, or CI/CD pipelines.

    Args:
        result: TrainingResult from the completed training run.
        config: Configuration used for training.
        hardware: Hardware information used during training.
        output_dir: Directory to save summary files.
        format: Output format - 'yaml', 'json', or 'both'.

    Returns:
        Dictionary mapping format to output file path.

    Example:
        >>> result = collect_training_result(ckpt_cb, es_cb)
        >>> paths = save_training_summary(result, cfg, hw, Path('outputs'))
        >>> print(f'Summary saved to: {paths}')
    '''
    import json
    from datetime import datetime

    import yaml

    output_dir.mkdir(parents=True, exist_ok=True)

    # Build summary data
    summary = {
        'training_run': {
            'experiment': config.experiment_name,
            'timestamp': datetime.now().isoformat(),
            'completed': True,
            'early_stopped': result.early_stopped,
            'stopped_epoch': result.stopped_epoch,
        },
        'results': {
            'best_loss': result.best_loss,
            'best_checkpoint': result.best_checkpoint_path,
            'last_checkpoint': result.last_checkpoint_path,
            'config_path': result.config_path,
        },
        'metrics': result.metrics,
        'hardware': {
            'accelerator': hardware.accelerator,
            'precision': hardware.precision,
            'num_devices': hardware.num_devices,
            'gpu_memory_gb': (hardware.gpu_memory['total_gb'] if hardware.gpu_memory else None),
        },
        'config_snapshot': {
            'model': {
                'base_model': config.model.base_model_name,
                'lora_rank': config.model.lora.r,
                'num_experts': config.model.moe.num_experts,
            },
            'training': {
                'learning_rate': config.training.learning_rate,
                'max_epochs': config.training.trainer.max_epochs,
                'batch_size': config.data_loader.batch_size,
            },
            'loss': {
                'temperature': config.loss.temperature,
                'curvature': config.loss.curvature,
                'hierarchy_weight': config.loss.hierarchy_weight,
            },
        },
    }

    output_paths: Dict[str, str] = {}

    if format in ('yaml', 'both'):
        yaml_path = output_dir / 'training_summary.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(summary, f, default_flow_style=False, sort_keys=False)
        output_paths['yaml'] = str(yaml_path)
        logger.info(f'Saved training summary (YAML): {yaml_path}')

    if format in ('json', 'both'):
        json_path = output_dir / 'training_summary.json'
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=2)
        output_paths['json'] = str(json_path)
        logger.info(f'Saved training summary (JSON): {json_path}')

    return output_paths
