"""
GPU configuration optimization tools.

Provides functions to optimize training configuration based on available GPU memory.
"""

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import yaml

logger = logging.getLogger(__name__)

# Import the classes and functions from the original script
# We'll keep the core logic but make it importable

@dataclass
class ModelConfig:
    """Model architecture configuration"""
    base_model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'
    embedding_dim: int = 384  # MiniLM-L6-v2
    num_channels: int = 4  # title, description, excluded, examples
    max_length: int = 512
    lora_r: int = 8
    num_experts: int = 4
    top_k: int = 2
    moe_hidden_dim: int = 1024
    use_fp16: bool = True


@dataclass
class TrainingConfig:
    """Training configuration"""
    batch_size: int = 4
    n_positives: int = 16
    n_negatives: int = 8
    accumulate_grad_batches: int = 16
    gradient_checkpointing: bool = True


@dataclass
class MemoryEstimate:
    """Breakdown of memory usage"""
    model_params_mb: float
    optimizer_state_mb: float
    gradients_mb: float
    activations_mb: float
    batch_data_mb: float
    overhead_mb: float
    total_mb: float
    
    def __str__(self) -> str:
        return (
            f"\nMemory Breakdown:\n"
            f"  Model Parameters:    {self.model_params_mb:>8.1f} MB\n"
            f"  Optimizer State:     {self.optimizer_state_mb:>8.1f} MB\n"
            f"  Gradients:           {self.gradients_mb:>8.1f} MB\n"
            f"  Activations:         {self.activations_mb:>8.1f} MB\n"
            f"  Batch Data:          {self.batch_data_mb:>8.1f} MB\n"
            f"  Overhead:            {self.overhead_mb:>8.1f} MB\n"
            f"  {'─' * 35}\n"
            f"  Total Estimated:     {self.total_mb:>8.1f} MB ({self.total_mb/1024:.2f} GB)"
        )


def estimate_model_parameters(model_cfg: ModelConfig) -> int:
    """Estimate total number of trainable parameters."""
    base_model_params = 22_700_000
    total_base_params = base_model_params * model_cfg.num_channels
    
    lora_params_per_channel = 2 * model_cfg.lora_r * model_cfg.embedding_dim * 50
    total_lora_params = lora_params_per_channel * model_cfg.num_channels
    
    input_dim = model_cfg.embedding_dim * model_cfg.num_channels
    expert_params = (input_dim * model_cfg.moe_hidden_dim + model_cfg.moe_hidden_dim * input_dim)
    total_expert_params = expert_params * model_cfg.num_experts
    
    gating_params = input_dim * model_cfg.num_experts
    projection_params = input_dim * model_cfg.embedding_dim
    
    total_params = (
        total_base_params + 
        total_lora_params + 
        total_expert_params + 
        gating_params + 
        projection_params
    )
    
    return total_params


def estimate_activation_memory(
    model_cfg: ModelConfig,
    train_cfg: TrainingConfig
) -> float:
    """Estimate activation memory per forward pass."""
    batch_size = train_cfg.batch_size
    n_total_samples = batch_size * (1 + 1 + train_cfg.n_negatives)
    
    num_layers = 6  # MiniLM-L6-v2
    attention_activations = model_cfg.max_length * model_cfg.max_length * 12
    hidden_activations = num_layers * model_cfg.max_length * model_cfg.embedding_dim * 4  
    encoder_activations_per_sample = (
        model_cfg.num_channels * (attention_activations + hidden_activations)
    )
    
    moe_input_dim = model_cfg.embedding_dim * model_cfg.num_channels
    moe_activations_per_sample = (
        moe_input_dim * 2 +
        model_cfg.moe_hidden_dim * model_cfg.top_k * 2 +
        model_cfg.embedding_dim * 2
    )
    
    total_activations_per_sample = encoder_activations_per_sample + moe_activations_per_sample
    total_activations = total_activations_per_sample * n_total_samples
    
    if train_cfg.gradient_checkpointing:
        total_activations *= 0.25
    
    bytes_per_element = 2 if model_cfg.use_fp16 else 4
    activation_mb = (total_activations * bytes_per_element) / (1024 ** 2)
    
    return activation_mb


def estimate_batch_data_memory(
    model_cfg: ModelConfig,
    train_cfg: TrainingConfig
) -> float:
    """Estimate memory for storing batch data."""
    batch_size = train_cfg.batch_size
    n_total_samples = batch_size * (1 + 1 + train_cfg.n_negatives)
    
    elements_per_sample = model_cfg.num_channels * model_cfg.max_length * 2
    bytes_per_sample = elements_per_sample * 8  # int64
    
    total_mb = (bytes_per_sample * n_total_samples) / (1024 ** 2)
    
    return total_mb


def estimate_memory_usage(
    model_cfg: ModelConfig,
    train_cfg: TrainingConfig
) -> MemoryEstimate:
    """Estimate total GPU memory usage for training."""
    trainable_params = estimate_model_parameters(model_cfg)
    bytes_per_param = 2 if model_cfg.use_fp16 else 4
    model_params_mb = (trainable_params * bytes_per_param) / (1024 ** 2)
    
    optimizer_state_mb = (trainable_params * 4 * 2) / (1024 ** 2)
    gradients_mb = model_params_mb
    activations_mb = estimate_activation_memory(model_cfg, train_cfg)
    batch_data_mb = estimate_batch_data_memory(model_cfg, train_cfg)
    overhead_mb = 1500  # ~1.5 GB base overhead
    
    total_mb = (
        model_params_mb + 
        optimizer_state_mb + 
        gradients_mb + 
        activations_mb + 
        batch_data_mb + 
        overhead_mb
    )
    
    return MemoryEstimate(
        model_params_mb=model_params_mb,
        optimizer_state_mb=optimizer_state_mb,
        gradients_mb=gradients_mb,
        activations_mb=activations_mb,
        batch_data_mb=batch_data_mb,
        overhead_mb=overhead_mb,
        total_mb=total_mb
    )


def detect_gpu_memory() -> Optional[float]:
    """Detect available GPU memory in GB."""
    if not torch.cuda.is_available():
        logger.error('No CUDA-capable GPU detected')
        return None
    
    device = torch.cuda.current_device()
    total_memory = torch.cuda.get_device_properties(device).total_memory
    total_gb = total_memory / (1024 ** 3)
    
    gpu_name = torch.cuda.get_device_name(device)
    logger.info(f'Detected GPU: {gpu_name}')
    logger.info(f'Total Memory: {total_gb:.2f} GB')
    
    return total_gb


def find_optimal_batch_size(
    available_memory_gb: float,
    model_cfg: ModelConfig,
    n_positives: int,
    n_negatives: int,
    safety_margin: float = 0.85
) -> Tuple[int, MemoryEstimate]:
    """Binary search to find optimal batch size that fits in memory."""
    target_memory_mb = available_memory_gb * 1024 * safety_margin
    
    min_batch = 1
    max_batch = 128
    best_batch = 1
    best_estimate = None
    
    while min_batch <= max_batch:
        mid_batch = (min_batch + max_batch) // 2
        
        train_cfg = TrainingConfig(
            batch_size=mid_batch,
            n_positives=n_positives,
            n_negatives=n_negatives
        )
        
        estimate = estimate_memory_usage(model_cfg, train_cfg)
        
        if estimate.total_mb <= target_memory_mb:
            best_batch = mid_batch
            best_estimate = estimate
            min_batch = mid_batch + 1
        else:
            max_batch = mid_batch - 1
    
    if best_estimate is None:
        train_cfg = TrainingConfig(
            batch_size=1,
            n_positives=n_positives,
            n_negatives=n_negatives
        )
        best_estimate = estimate_memory_usage(model_cfg, train_cfg)
    
    return best_batch, best_estimate


def suggest_configurations(
    available_memory_gb: float,
    model_cfg: ModelConfig,
    target_effective_batch: int = 256
) -> List[Dict]:
    """Suggest multiple training configurations optimized for the available GPU memory."""
    suggestions = []
    
    stage_profiles = [
        {'name': '01_text (Early)', 'n_positives': 32, 'n_negatives': 24},
        {'name': '02-05_text (Later)', 'n_positives': 16, 'n_negatives': 8},
    ]
    
    for profile in stage_profiles:
        n_pos = profile['n_positives']
        n_neg = profile['n_negatives']
        
        batch_size, estimate = find_optimal_batch_size(
            available_memory_gb,
            model_cfg,
            n_pos,
            n_neg
        )
        
        accumulate_grad_batches = max(1, target_effective_batch // batch_size)
        effective_batch = batch_size * accumulate_grad_batches
        
        suggestions.append({
            'stage': profile['name'],
            'batch_size': batch_size,
            'n_positives': n_pos,
            'n_negatives': n_neg,
            'accumulate_grad_batches': accumulate_grad_batches,
            'effective_batch_size': effective_batch,
            'memory_estimate': estimate,
            'memory_utilization': f'{(estimate.total_mb / (available_memory_gb * 1024)) * 100:.1f}%'
        })
    
    return suggestions


def update_config_file(
    config_path: str,
    updates: Dict,
    backup: bool = True
) -> None:
    """Update configuration file with new values."""
    config_path_obj = Path(config_path)
    
    if backup and config_path_obj.exists():
        backup_path = config_path_obj.with_suffix('.yaml.backup')
        shutil.copy(config_path_obj, backup_path)
        logger.info(f'Backup created: {backup_path}')
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    for key_path, value in updates.items():
        keys = key_path.split('.')
        current = config
        for key in keys[:-1]:
            current = current[key]
        current[keys[-1]] = value
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    logger.info(f'Configuration updated: {config_path}')


def update_curriculum_file(
    curriculum_path: str,
    n_positives: int,
    n_negatives: int,
    backup: bool = True
) -> None:
    """Update curriculum stage file with new sample counts."""
    curriculum_path_obj = Path(curriculum_path)
    
    if backup and curriculum_path_obj.exists():
        backup_path = curriculum_path_obj.with_suffix('.yaml.backup')
        shutil.copy(curriculum_path_obj, backup_path)
    
    with open(curriculum_path, 'r') as f:
        config = yaml.safe_load(f)
    
    config['n_positives'] = n_positives
    config['n_negatives'] = n_negatives
    
    with open(curriculum_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    logger.info(f'Curriculum updated: {curriculum_path}')


def optimize_gpu_config(
    gpu_memory_gb: Optional[float] = None,
    auto_detect: bool = False,
    target_effective_batch: int = 256,
    apply: bool = False,
    config_path: str = './conf/config.yaml'
) -> Dict:
    """
    Optimize training configuration for available GPU memory.
    
    Args:
        gpu_memory_gb: GPU memory in GB (if None and auto_detect=False, will error)
        auto_detect: Auto-detect GPU memory
        target_effective_batch: Target effective batch size
        apply: Apply suggested configuration to config files
        config_path: Path to main config file
        
    Returns:
        Dictionary with suggestions and results
    """
    # Determine available memory
    if auto_detect:
        available_memory_gb = detect_gpu_memory()
        if available_memory_gb is None:
            raise RuntimeError('Could not detect GPU memory')
    elif gpu_memory_gb:
        available_memory_gb = gpu_memory_gb
        logger.info(f'Using specified GPU memory: {available_memory_gb:.2f} GB')
    else:
        raise ValueError('Must specify either gpu_memory_gb or auto_detect=True')
    
    # Model configuration
    model_cfg = ModelConfig()
    
    # Generate suggestions
    suggestions = suggest_configurations(
        available_memory_gb,
        model_cfg,
        target_effective_batch
    )
    
    result = {
        'gpu_memory_gb': available_memory_gb,
        'suggestions': suggestions,
        'applied': False
    }
    
    # Apply configuration if requested
    if apply:
        config = suggestions[0]
        
        updates = {
            'data_loader.batch_size': config['batch_size'],
            'training.trainer.accumulate_grad_batches': config['accumulate_grad_batches']
        }
        
        try:
            update_config_file(config_path, updates, backup=True)
            
            # Update curriculum files
            conf_dir = Path(config_path).parent
            curriculum_dir = conf_dir / 'text_curriculum'
            
            # Update stage 01
            stage_01_path = curriculum_dir / '01_text.yaml'
            if stage_01_path.exists():
                update_curriculum_file(
                    str(stage_01_path),
                    n_positives=32,
                    n_negatives=24,
                    backup=True
                )
            
            # Update stages 02-05
            if len(suggestions) > 1:
                later_config = suggestions[1]
                for stage_num in ['02', '03', '04', '05']:
                    stage_path = curriculum_dir / f'{stage_num}_text.yaml'
                    if stage_path.exists():
                        update_curriculum_file(
                            str(stage_path),
                            n_positives=later_config['n_positives'],
                            n_negatives=later_config['n_negatives'],
                            backup=True
                        )
            
            result['applied'] = True
            logger.info('✓ Configuration files updated successfully!')
            
        except Exception as e:
            logger.error(f'✗ Failed to update configuration: {e}')
            raise
    
    return result

