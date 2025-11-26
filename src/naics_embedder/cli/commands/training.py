# -------------------------------------------------------------------------------------------------
# Training Commands
# -------------------------------------------------------------------------------------------------

'''
CLI commands for training NAICS embedding models.

This module provides the ``train`` and ``train-seq`` commands that orchestrate
the text encoder training workflow. Configuration is loaded from YAML files and
can be overridden via command-line arguments.

Commands:
    train: Train a single stage with optional checkpoint resumption.
    train-seq: Run sequential multi-stage training (deprecated, use --legacy).
'''

import logging
import time
from pathlib import Path
from typing import List, Optional

import polars as pl
import pytorch_lightning as pyl
import torch
import typer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from rich.console import Console
from rich.panel import Panel
from typing_extensions import Annotated

from naics_embedder.text_model.dataloader.datamodule import NAICSDataModule
from naics_embedder.text_model.dataloader.tokenization_cache import tokenization_cache
from naics_embedder.text_model.naics_model import NAICSContrastiveModel
from naics_embedder.utils.backend import get_device
from naics_embedder.utils.config import (
    Config,
    TokenizationConfig,
    parse_override_value,
)
from naics_embedder.utils.console import configure_logging
from naics_embedder.utils.training import (
    HardwareInfo,
    TrainingResult,
    collect_training_result,
    create_trainer,
    detect_hardware,
    parse_config_overrides,
    resolve_checkpoint,
    save_training_summary,
)
from naics_embedder.utils.utilities import pick_device
from naics_embedder.utils.validation import validate_training_config
from naics_embedder.utils.warnings import configure_warnings

# Apply centralized warning configuration
configure_warnings()

console = Console()
logger = logging.getLogger(__name__)


# -------------------------------------------------------------------------------------------------
# Embedding Generation
# -------------------------------------------------------------------------------------------------

def generate_embeddings_from_checkpoint(
    checkpoint_path: str,
    config: Config,
    output_path: Optional[str] = None,
    batch_size: int = 32
) -> str:
    """Generate hyperbolic embeddings parquet file from a trained checkpoint.

    Loads a trained model checkpoint, runs inference on all NAICS codes, and
    writes the resulting embeddings to a parquet file compatible with HGCN
    training.

    Args:
        checkpoint_path: Filesystem path to the PyTorch Lightning checkpoint
            that contains the trained contrastive model weights.
        config: Project configuration containing data paths used for token
            caching and parquet loading.
        output_path: Optional path for the embeddings parquet. When omitted,
            ``output/hyperbolic_projection/encodings.parquet`` is used.
        batch_size: Batch size to use during inference to balance throughput
            and memory usage.

    Returns:
        str: Filesystem path to the generated embeddings parquet file.
    """
    logger.info('=' * 80)
    logger.info('GENERATING EMBEDDINGS FROM CHECKPOINT')
    logger.info('=' * 80)
    logger.info(f'Checkpoint: {checkpoint_path}')

    # Determine output path
    if output_path is None:
        # Use default location: ./output/hyperbolic_projection/encodings.parquet
        output_dir = Path('./output/hyperbolic_projection')
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(output_dir / 'encodings.parquet')

    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f'Output: {output_path}')
    
    # Load device
    device = pick_device('auto')  # Auto-detect device
    logger.info(f'Device: {device}')
    
    # Load model from checkpoint
    logger.info('Loading model from checkpoint...')
    model = NAICSContrastiveModel.load_from_checkpoint(
        checkpoint_path,
        map_location=device
    )
    model.eval()
    model.to(device)
    logger.info('Model loaded successfully')
    
    # Load descriptions parquet
    descriptions_path = config.data_loader.streaming.descriptions_parquet
    logger.info(f'Loading NAICS descriptions from: {descriptions_path}')
    
    df = pl.read_parquet(descriptions_path).sort('index')
    logger.info(f'Loaded {df.height:,} NAICS codes')
    
    # Load tokenization cache
    tokenization_cfg = TokenizationConfig(
        descriptions_parquet=descriptions_path,
        tokenizer_name=config.data_loader.tokenization.tokenizer_name,
        max_length=config.data_loader.tokenization.max_length
    )
    
    logger.info('Loading tokenization cache...')
    token_cache = tokenization_cache(tokenization_cfg, use_locking=False)
    logger.info('Tokenization cache loaded')
    
    # Generate embeddings in batches
    logger.info(f'Generating embeddings (batch_size={batch_size})...')
    all_embeddings = []
    all_indices = []
    all_levels = []
    all_codes = []
    
    num_batches = (df.height + batch_size - 1) // batch_size
    
    with torch.no_grad():
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, df.height)
            batch_df = df.slice(start_idx, end_idx - start_idx)
            
            # Prepare batch inputs
            channel_inputs = {
                'title': {'input_ids': [], 'attention_mask': []},
                'description': {'input_ids': [], 'attention_mask': []},
                'excluded': {'input_ids': [], 'attention_mask': []},
                'examples': {'input_ids': [], 'attention_mask': []}
            }
            
            batch_indices = []
            batch_levels = []
            batch_codes = []
            
            for row in batch_df.iter_rows(named=True):
                idx = row['index']
                batch_indices.append(idx)
                batch_levels.append(row['level'])
                batch_codes.append(row['code'])
                
                # Get tokenized inputs from cache
                tokens = token_cache[idx]
                
                for channel in ['title', 'description', 'excluded', 'examples']:
                    channel_inputs[channel]['input_ids'].append(tokens[channel]['input_ids'])
                    channel_inputs[channel]['attention_mask'].append(tokens[channel]['attention_mask'])
            
            # Stack tensors
            for channel in channel_inputs:
                channel_inputs[channel]['input_ids'] = torch.stack(
                    channel_inputs[channel]['input_ids']
                ).to(device)
                channel_inputs[channel]['attention_mask'] = torch.stack(
                    channel_inputs[channel]['attention_mask']
                ).to(device)
            
            # Run inference
            output = model(channel_inputs)
            embeddings = output['embedding']  # Hyperbolic embeddings (batch_size, embedding_dim+1)
            
            # Store embeddings
            all_embeddings.append(embeddings.cpu())
            all_indices.extend(batch_indices)
            all_levels.extend(batch_levels)
            all_codes.extend(batch_codes)
            
            if (batch_idx + 1) % 10 == 0 or batch_idx == num_batches - 1:
                logger.info(f'  Processed {end_idx:,} / {df.height:,} codes ({(end_idx/df.height)*100:.1f}%)')
    
    # Concatenate all embeddings
    logger.info('Concatenating embeddings...')
    all_embeddings_tensor = torch.cat(all_embeddings, dim=0)  # (N, embedding_dim+1)
    embedding_dim = all_embeddings_tensor.shape[1]
    
    logger.info(f'Generated embeddings: shape={all_embeddings_tensor.shape}')
    
    # Convert to numpy
    embeddings_np = all_embeddings_tensor.numpy()
    
    # Create DataFrame with hyp_e* columns
    emb_schema = {f'hyp_e{i}': pl.Float64 for i in range(embedding_dim)}
    emb_df = pl.DataFrame(embeddings_np, schema=emb_schema)
    
    # Combine with metadata
    base_df = pl.DataFrame({
        'index': all_indices,
        'level': all_levels,
        'code': all_codes
    })
    
    result_df = base_df.hstack(emb_df)
    
    # Save to parquet
    logger.info(f'Saving embeddings to: {output_path}')
    result_df.write_parquet(output_path)
    
    logger.info('=' * 80)
    logger.info('EMBEDDING GENERATION COMPLETE')
    logger.info('=' * 80)
    logger.info(f'Embeddings saved: {output_path}')
    logger.info(f'Total codes: {df.height:,}')
    logger.info(f'Embedding dimension: {embedding_dim}')
    
    return output_path


def train(
    config_file: Annotated[
        str,
        typer.Option(
            '--config',
            help='Path to base config YAML file',
        ),
    ] = 'conf/config.yaml',
    ckpt_path: Annotated[
        Optional[str],
        typer.Option(
            '--ckpt-path',
            help='Path to checkpoint file to resume from, or "last" to auto-detect last checkpoint',
        ),
    ] = None,
    skip_validation: Annotated[
        bool,
        typer.Option(
            '--skip-validation',
            help='Skip pre-flight validation of data files and cache',
        ),
    ] = False,
    overrides: Annotated[
        Optional[List[str]],
        typer.Argument(
            help="Config overrides (e.g., 'training.learning_rate=1e-4 data.batch_size=64')"
        ),
    ] = None,
):
    '''
    Train the NAICS text encoder with contrastive learning.
    
    Orchestrates the complete training workflow including configuration loading,
    hardware detection, checkpoint management, and training execution with
    PyTorch Lightning. Supports resumption from checkpoints and runtime
    configuration overrides.
    
    Args:
        config_file: Path to the base YAML configuration file that describes
            data, model, and training settings. Defaults to ``conf/config.yaml``.
        ckpt_path: Optional checkpoint path to resume training. Use ``last`` to
            automatically pick up the latest checkpoint for the configured
            experiment. Specify a full path for cross-experiment resumption.
        skip_validation: Skip pre-flight validation checks for data files and
            tokenization cache. Useful when you know files are valid.
        overrides: Optional list of key-value override strings. Use dot notation
            to specify nested config values like ``training.learning_rate=1e-4``.
    
    Example:
        Train with default configuration::
        
            $ uv run naics-embedder train
        
        Resume from last checkpoint with custom learning rate::
        
            $ uv run naics-embedder train --ckpt-path last training.learning_rate=1e-5
    '''

    configure_logging('train.log')

    console.rule('[bold green]Training NAICS Embedder[/bold green]')
    
    try:

        # Detect hardware using centralized utility
        logger.info('Determining infrastructure...')
        hardware = detect_hardware(log_info=True)
        
        # Log GPU memory if available
        if hardware.gpu_memory:
            logger.info(
                f'GPU Memory: {hardware.gpu_memory["reserved_gb"]:.1f} GB used / '
                f'{hardware.gpu_memory["total_gb"]:.1f} GB total '
                f'({hardware.gpu_memory["utilization_pct"]:.1f}% utilization, '
                f'{hardware.gpu_memory["free_gb"]:.1f} GB free)'
            )

        # Load configuration
        logger.info('Loading configuration...')
        cfg = Config.from_yaml(config_file)
        
        # Apply command-line overrides using centralized parsing
        if overrides:
            logger.info('Applying command-line overrides:')
            override_dict, invalid_overrides = parse_config_overrides(overrides)
            
            for invalid in invalid_overrides:
                console.print(f'[yellow]Warning:[/yellow] Skipping invalid override: {invalid}')
            
            if override_dict:
                logger.info('')
                cfg = cfg.override(override_dict)
        
        # Run pre-flight validation
        if not skip_validation:
            validation_result = validate_training_config(cfg)
            if not validation_result.valid:
                console.print('\n[bold red]Pre-flight validation failed:[/bold red]')
                for error in validation_result.errors:
                    console.print(f'  [red]✗[/red] {error}')
                console.print('\n[dim]Use --skip-validation to bypass these checks[/dim]')
                raise typer.Exit(code=1)
            
            for warning in validation_result.warnings:
                console.print(f'[yellow]⚠[/yellow] {warning}')
        
        # Display configuration summary
        summary_list_1 = [
            f'[bold]Experiment:[/bold] {cfg.experiment_name}',
            f'[bold]Seed:[/bold] {cfg.seed}\n',
            '[cyan]Data:[/cyan]',
            f'  • Batch size: {cfg.data_loader.batch_size}',
            f'  • Num workers: {cfg.data_loader.num_workers}\n',
        ]

        summary_list_3 = [
            '[cyan]Model:[/cyan]',
            f'  • Base: {cfg.model.base_model_name.split("/")[-1]}',
            f'  • LoRA rank: {cfg.model.lora.r}',
            '  • MoE: ',
            f'    - {cfg.model.moe.num_experts} experts\n',
            '[cyan]Training:[/cyan]',
            f'  • Learning rate: {cfg.training.learning_rate}',
            f'  • Max epochs: {cfg.training.trainer.max_epochs}',
            f'  • Accelerator: {hardware.accelerator}',
            f'  • Precision: {hardware.precision}'
        ]
        
        # Add GPU memory info and batch size suggestions
        summary_list_4 = []
        if hardware.gpu_memory:
            summary_list_4.append('\n[cyan]GPU Memory:[/cyan]')
            summary_list_4.append(
                f'  • Used: {hardware.gpu_memory["reserved_gb"]:.1f} GB / '
                f'{hardware.gpu_memory["total_gb"]:.1f} GB '
                f'({hardware.gpu_memory["utilization_pct"]:.1f}% utilization)'
            )
            summary_list_4.append(f'  • Free: {hardware.gpu_memory["free_gb"]:.1f} GB')
            
            # Conservative batch size suggestion
            current_batch_size = cfg.data_loader.batch_size
            if hardware.gpu_memory['free_gb'] > 8.0 and current_batch_size < 12:
                # Suggest 2x-3x current batch size conservatively
                suggested_batch = min(12, current_batch_size * 2)
                if suggested_batch > current_batch_size:
                    summary_list_4.append(
                        '\n[yellow]Batch Size Suggestion:[/yellow]'
                    )
                    summary_list_4.append(
                        f'  • Current: {current_batch_size}'
                    )
                    summary_list_4.append(
                        f'  • Suggested: {suggested_batch} (conservative estimate)'
                    )
                    summary_list_4.append(
                        '  • [dim]Note: gpu_tools.py estimates are optimistic; '
                        'reduce suggested values by ~50%[/dim]'
                    )
                    summary_list_4.append(
                        f'  • [dim]Override with: data.batch_size={suggested_batch}[/dim]'
                    )

        summary = '\n'.join(summary_list_1 + summary_list_3 + summary_list_4)

        console.print(
            Panel(
                summary,
                title='[yellow]Configuration Summary[/yellow]',
                border_style='yellow',
                expand=False
            )
        )
        console.print('')

        # Seed for reproducibility
        logger.info(f'Setting random seed: {cfg.seed}\n')
        pyl.seed_everything(cfg.seed, verbose=False)
        
        # Initialize DataModule
        logger.info('Initializing DataModule...')

        datamodule = NAICSDataModule(
            descriptions_path=cfg.data_loader.streaming.descriptions_parquet,
            triplets_path=cfg.data_loader.streaming.triplets_parquet,
            tokenizer_name=cfg.data_loader.tokenization.tokenizer_name,
            streaming_config=cfg.data_loader.streaming.model_dump(),
            batch_size=cfg.data_loader.batch_size,
            num_workers=cfg.data_loader.num_workers,
            val_split=cfg.data_loader.val_split,
            seed=cfg.seed
        )
        
        # Handle checkpoint resumption using centralized utility
        checkpoint_dir = Path(cfg.dirs.checkpoint_dir) / cfg.experiment_name
        checkpoint_info = resolve_checkpoint(ckpt_path, Path(cfg.dirs.checkpoint_dir), cfg.experiment_name)
        checkpoint_path = checkpoint_info.path
        
        if checkpoint_info.exists:
            console.print(f'[green]✓[/green] Resuming from checkpoint: [cyan]{checkpoint_path}[/cyan]\n')
        elif ckpt_path:
            console.print(f'[yellow]Warning:[/yellow] Checkpoint not found at {ckpt_path}')
            console.print('Starting training from scratch.\n')
        
        # Initialize Model
        logger.info('Initializing Model with evaluation metrics...\n')

        if checkpoint_path:
            # Load model from checkpoint
            model = NAICSContrastiveModel.load_from_checkpoint(
                checkpoint_path,
                # Override learning rate if needed (checkpoint may have different LR)
                learning_rate=cfg.training.learning_rate,
                base_margin=cfg.loss.base_margin,
            )
            logger.info(f'Model loaded from checkpoint: {checkpoint_path}')
        else:
            # Initialize new model
            model = NAICSContrastiveModel(
                base_model_name=cfg.model.base_model_name,
                lora_r=cfg.model.lora.r,
                lora_alpha=cfg.model.lora.alpha,
                lora_dropout=cfg.model.lora.dropout,
                num_experts=cfg.model.moe.num_experts,
                top_k=cfg.model.moe.top_k,
                moe_hidden_dim=cfg.model.moe.hidden_dim,
                temperature=cfg.loss.temperature,
                curvature=cfg.loss.curvature,
                hierarchy_weight=cfg.loss.hierarchy_weight,
                rank_order_weight=cfg.loss.rank_order_weight,
                radius_reg_weight=cfg.loss.radius_reg_weight,
                learning_rate=cfg.training.learning_rate,
                weight_decay=cfg.training.weight_decay,
                warmup_steps=cfg.training.warmup_steps,
                load_balancing_coef=cfg.model.moe.load_balancing_coef,
                distance_matrix_path=cfg.data_loader.streaming.distance_matrix_parquet,
                eval_every_n_epochs=cfg.model.eval_every_n_epochs,
                eval_sample_size=cfg.model.eval_sample_size,
                base_margin=cfg.loss.base_margin,
                tree_distance_alpha=cfg.curriculum.tree_distance_alpha,
                curriculum_phase1_end=cfg.curriculum.phase1_end,
                curriculum_phase2_end=cfg.curriculum.phase2_end,
                curriculum_phase3_end=cfg.curriculum.phase3_end,
                sibling_distance_threshold=cfg.curriculum.sibling_distance_threshold,
                fn_curriculum_start_epoch=cfg.curriculum.fn_curriculum_start_epoch,
                fn_cluster_every_n_epochs=cfg.curriculum.fn_cluster_every_n_epochs,
                fn_num_clusters=cfg.curriculum.fn_num_clusters
            )
        
        # Setup callbacks
        logger.info('Setting up callbacks and checkpointing...\n')
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if checkpoint is from the same stage (for resuming) or different stage (for loading weights)
        resuming_same_stage = checkpoint_info.is_same_stage
        if checkpoint_path:
            if resuming_same_stage:
                logger.info('Checkpoint is from same stage - will resume training from checkpoint')
                console.print('[cyan]Resuming training from checkpoint (will continue from saved epoch)[/cyan]\n')
            else:
                logger.info('Checkpoint is from different stage - will load weights only, start fresh training')
                console.print('[cyan]Loading weights from previous stage checkpoint, starting fresh training[/cyan]\n')
        
        checkpoint_callback = ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename='naics-{epoch:02d}-{val/contrastive_loss:.4f}',
            monitor='val/contrastive_loss',
            mode='min',
            save_top_k=3,
            save_last=True
        )
        
        early_stopping = EarlyStopping(
            monitor='val/contrastive_loss',
            patience=3,  # Reduced from 5 to prevent training beyond optimal point
            mode='min',
            min_delta=0.0001,  # Minimum improvement required
            verbose=True
        )
        
        # TensorBoard logger - ensure directory exists first
        tb_log_dir = Path(cfg.dirs.output_dir) / cfg.experiment_name
        tb_log_dir.mkdir(parents=True, exist_ok=True)
        
        tb_logger = TensorBoardLogger(
            save_dir=cfg.dirs.output_dir,
            name=cfg.experiment_name
        )
        
        # Don't add epoch progress callback - PyTorch Lightning's progress bar
        # already shows epoch info
        
        # Initialize Trainer
        logger.info('Initializing PyTorch Lightning Trainer...\n')
        
        # Use only 1 device as specified in config, even if multiple GPUs are available
        devices_to_use = (
            cfg.training.trainer.devices
            if hasattr(cfg.training.trainer, 'devices')
            else 1
        )
        
        # If using multiple devices, need to handle unused parameters in DDP
        strategy = 'auto'
        if devices_to_use > 1 and hardware.accelerator in ['cuda', 'gpu']:
            from pytorch_lightning.strategies import DDPStrategy
            strategy = DDPStrategy(find_unused_parameters=True)
        
        trainer = pyl.Trainer(
            max_epochs=cfg.training.trainer.max_epochs,
            accelerator=hardware.accelerator,
            devices=devices_to_use,
            strategy=strategy,
            precision=hardware.precision, # type: ignore
            gradient_clip_val=cfg.training.trainer.gradient_clip_val,
            accumulate_grad_batches=cfg.training.trainer.accumulate_grad_batches,
            log_every_n_steps=cfg.training.trainer.log_every_n_steps,
            val_check_interval=cfg.training.trainer.val_check_interval,
            callbacks=[checkpoint_callback, early_stopping],
            logger=tb_logger,
            default_root_dir=cfg.dirs.output_dir
        )
        
        # Start training
        logger.info('Starting model training with evaluation metrics...\n')
        console.print('[bold cyan]Evaluation metrics enabled:[/bold cyan]')
        console.print('  • Cophenetic correlation (hierarchy preservation)')
        console.print('  • NDCG@k (ranking quality: position-aware metric)')
        console.print('  • Embedding statistics (norms, distances)')
        console.print('  • Collapse detection (variance, norm, distance)')
        console.print('  • Distortion metrics (mean, std)\n')
        
        console.print(
            f'[bold yellow]Training for {cfg.training.trainer.max_epochs} epochs...'
            f'[/bold yellow]\n'
        )
        
        # Only pass checkpoint path to trainer.fit() if resuming the same stage
        # If loading from a different stage, we've already loaded the weights above,
        # so we don't pass ckpt_path to start training fresh
        trainer_ckpt_path = checkpoint_path if resuming_same_stage else None
        trainer.fit(model, datamodule, ckpt_path=trainer_ckpt_path)
        
        # Training complete
        logger.info('Training complete!')
        logger.info(f'Best model checkpoint: {checkpoint_callback.best_model_path}')
        
        # Check if early stopping was triggered and get the best loss
        early_stop_triggered = early_stopping.stopped_epoch > 0
        best_loss = early_stopping.best_score if early_stopping.best_score is not None else None
        
        if early_stop_triggered and best_loss is not None:
            logger.info(f'Early stopping triggered at epoch {early_stopping.stopped_epoch} with best loss: {best_loss:.6f}')
        
        console.print(
            f'\n[bold green]✓ Training completed successfully![/bold green]\n'
            f'Best checkpoint: [cyan]{checkpoint_callback.best_model_path}[/cyan]\n'
        )
        
        # Print the loss that decided early stopping as the final metric
        if best_loss is not None:
            label = 'Final evaluation metric (early stopping)' if early_stop_triggered else 'Final evaluation metric'
            console.print(
                f'[bold]{label}:[/bold] '
                f'[cyan]val/contrastive_loss = {best_loss:.6f}[/cyan]\n'
            )
            logger.info(f'{label}: val/contrastive_loss = {best_loss:.6f}')
        
        # Save final config
        config_output_path = checkpoint_dir / 'config.yaml'
        cfg.to_yaml(str(config_output_path))
        console.print(f'Config saved: [cyan]{config_output_path}[/cyan]\n')
        
        # Save training summary artifacts for downstream evaluation and documentation
        training_result = TrainingResult(
            best_checkpoint_path=checkpoint_callback.best_model_path,
            last_checkpoint_path=str(checkpoint_dir / 'last.ckpt'),
            config_path=str(config_output_path),
            best_loss=float(best_loss) if best_loss is not None else None,
            stopped_epoch=early_stopping.stopped_epoch if early_stop_triggered else -1,
            early_stopped=early_stop_triggered,
            metrics={'best_val_loss': float(best_loss) if best_loss is not None else None}
        )
        
        summary_paths = save_training_summary(
            result=training_result,
            config=cfg,
            hardware=hardware,
            output_dir=checkpoint_dir
        )
        console.print(f'Training summary saved: [cyan]{summary_paths.get("yaml", summary_paths.get("json"))}[/cyan]\n')
        
        # Prompt to generate embeddings for HGCN training
        console.print('\n[bold cyan]Generate embeddings for HGCN training?[/bold cyan]')
        generate_embeddings = typer.confirm(
            'Generate embeddings parquet file from this checkpoint?',
            default=False
        )
        
        if generate_embeddings:
            logger.info('Generating embeddings from checkpoint...')
            embeddings_path = generate_embeddings_from_checkpoint(
                checkpoint_path=checkpoint_callback.best_model_path,
                config=cfg,
                output_path=None  # Will use default location
            )
            console.print(
                f'\n[bold green]✓ Embeddings generated successfully![/bold green]\n'
                f'Embeddings saved to: [cyan]{embeddings_path}[/cyan]\n'
                f'This file can be used for HGCN training.\n'
            )
        
    except Exception as e:
        logger.error(f'Training failed: {e}', exc_info=True)
        console.print(f'\n[bold red]✗ Training failed:[/bold red] {e}\n')
        raise typer.Exit(code=1)


def train_sequential(
    num_stages: Annotated[
        int,
        typer.Option(
            '--num-stages',
            '-n',
            help='Number of training stages to run sequentially',
        ),
    ] = 3,
    config_file: Annotated[
        str,
        typer.Option(
            '--config',
            help='Path to base config YAML file',
        ),
    ] = 'conf/config.yaml',
    resume_from_checkpoint: Annotated[
        bool,
        typer.Option(
            '--resume',
            help='Resume from last checkpoint if available',
        ),
    ] = False,
    legacy: Annotated[
        bool,
        typer.Option(
            '--legacy',
            help='Acknowledge use of deprecated sequential training workflow',
        ),
    ] = False,
    overrides: Annotated[
        Optional[List[str]],
        typer.Argument(
            help="Config overrides (e.g., 'training.learning_rate=1e-4')"
        ),
    ] = None,
):
    '''
    Run sequential multi-stage training with automatic checkpoint handoff.
    
    .. deprecated::
        Sequential training is deprecated. Use the standard ``train`` command
        with dynamic curriculum learning instead. Pass ``--legacy`` to
        acknowledge and continue using this deprecated workflow.
    
    This routine iterates over curriculum stages, loading the best checkpoint
    from each stage as initialization for the next. Each stage can have
    different hyperparameters and training objectives.
    
    Args:
        num_stages: Number of training stages to run in sequence.
        config_file: Path to the base configuration file used for all stages.
        resume_from_checkpoint: Whether to resume from the last checkpoint of
            the previous run when available.
        legacy: Flag to acknowledge using the deprecated sequential training
            workflow. Required to proceed with sequential training.
        overrides: Optional list of key-value override strings applied to every
            stage configuration.
    
    Example:
        Run 3-stage sequential training (deprecated)::
        
            $ uv run naics-embedder train-seq --legacy --num-stages 3
    
    See Also:
        Use ``train`` for the recommended single-stage training with dynamic
        curriculum learning.
    '''

    configure_logging('train_sequential.log')

    # Require --legacy flag to acknowledge deprecation
    if not legacy:
        console.print('\n[bold red]Sequential training is deprecated.[/bold red]')
        console.print('\nThe sequential training workflow is maintained for backwards')
        console.print('compatibility but is no longer recommended.')
        console.print('\n[bold]Recommended:[/bold] Use the standard training command instead:')
        console.print('  [cyan]uv run naics-embedder train[/cyan]')
        console.print('\n[bold]To continue with deprecated sequential training:[/bold]')
        console.print('  [cyan]uv run naics-embedder train-seq --legacy[/cyan]')
        console.print('\nFor migration guidance, see: [link]https://lowmason.github.io/naics-embedder/text_training[/link]\n')
        raise typer.Exit(code=1)

    console.rule('[bold cyan]Starting Sequential Training (Legacy)[/bold cyan]')
    console.print('[yellow]⚠ Warning: Using deprecated sequential training workflow.[/yellow]')
    console.print('[dim]Consider migrating to dynamic curriculum training.[/dim]\n')

    console.print(f'[bold]Total stages:[/bold] {num_stages}\n')

    last_checkpoint = None

    # Train each stage
    for i in range(1, num_stages + 1):
        console.rule(f'[bold green]Stage {i}/{num_stages}[/bold green]')

        try:
            # Load config for this stage
            cfg = Config.from_yaml(config_file)
            
            # Apply overrides
            if overrides:
                override_dict = {}
                for override in overrides:
                    if '=' not in override:
                        console.print(
                            f'[yellow]Warning:[/yellow] Skipping invalid override: {override}'
                        )
                        continue
                    key, value_str = override.split('=', 1)
                    value = parse_override_value(value_str)
                    override_dict[key] = value
                cfg = cfg.override(override_dict)
            
            # Create directories
            Path(cfg.dirs.output_dir).mkdir(parents=True, exist_ok=True)
            Path(cfg.dirs.checkpoint_dir).mkdir(parents=True, exist_ok=True)
            
            # Initialize data module
            datamodule = NAICSDataModule(cfg)
            
            # Initialize model (load from checkpoint if available)
            if last_checkpoint:
                console.print(f'[cyan]Loading from previous stage: {last_checkpoint}[/cyan]')
                model = NAICSContrastiveModel.load_from_checkpoint(
                    last_checkpoint,
                    cfg=cfg,
                    strict=False
                )
            else:
                model = NAICSContrastiveModel(cfg)
            
            # Setup callbacks
            checkpoint_callback = ModelCheckpoint(
                dirpath=cfg.dirs.checkpoint_dir,
                filename=f'{curriculum}-{{epoch:02d}}-{{val_loss:.4f}}',
                monitor='val_loss',
                mode='min',
                save_top_k=3,
                save_last=True
            )
            
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=cfg.training.trainer.early_stopping_patience,
                mode='min',
                verbose=True
            )
            
            tb_logger = TensorBoardLogger(
                save_dir=cfg.dirs.output_dir,
                name='tensorboard',
                version=curriculum
            )
            
            # Initialize trainer
            trainer = pyl.Trainer(
                max_epochs=cfg.training.trainer.max_epochs,
                accelerator='gpu' if get_device() == 'cuda' else 'cpu',
                devices=1,
                precision='16-mixed',
                gradient_clip_val=cfg.training.optimizer.gradient_clip_val,
                accumulate_grad_batches=cfg.training.optimizer.accumulate_grad_batches,
                log_every_n_steps=cfg.training.trainer.log_every_n_steps,
                val_check_interval=cfg.training.trainer.val_check_interval,
                callbacks=[checkpoint_callback, early_stopping],
                logger=tb_logger,
                default_root_dir=cfg.dirs.output_dir
            )
            
            # Train this stage
            console.print(f'\n[cyan]Training stage {curriculum}...[/cyan]\n')
            trainer.fit(model, datamodule)
            
            # Store checkpoint for next stage
            last_checkpoint = checkpoint_callback.best_model_path
            
            # Check if early stopping was triggered and get the best loss
            early_stop_triggered = early_stopping.stopped_epoch > 0
            best_loss = early_stopping.best_score if early_stopping.best_score is not None else None
            
            if early_stop_triggered and best_loss is not None:
                logger.info(f'Early stopping triggered at epoch {early_stopping.stopped_epoch} with best loss: {best_loss:.6f}')
            
            console.print(
                f'\n[green]✓[/green] Stage {i} complete. '
                f'Best checkpoint: [cyan]{last_checkpoint}[/cyan]\n'
            )
            
            # Print the loss that decided early stopping as the final metric
            if best_loss is not None:
                label = 'Final evaluation metric (early stopping)' if early_stop_triggered else 'Final evaluation metric'
                console.print(
                    f'[bold]{label}:[/bold] '
                    f'[cyan]val/contrastive_loss = {best_loss:.6f}[/cyan]\n'
                )
                logger.info(f'{label}: val/contrastive_loss = {best_loss:.6f}')
            
            # Brief pause between stages
            if i < num_stages:
                console.print('[dim]Preparing for next stage...[/dim]\n')
                time.sleep(2)

        except Exception as e:
            logger.error(f'Stage {i} failed: {e}', exc_info=True)
            console.print(f'\n[bold red]✗ Stage {i} failed:[/bold red] {e}\n')

            if i < num_stages:
                response = typer.prompt(
                    f'Continue with next stage ({i+1})? [y/N]',
                    default='n'
                )
                if response.lower() != 'y':
                    raise typer.Exit(code=1)
            else:
                raise typer.Exit(code=1)

    console.rule('[bold green]Sequential Training Complete![/bold green]')
    console.print(f'\n[bold]Final checkpoint:[/bold] [cyan]{last_checkpoint}[/cyan]\n')

    # Prompt to generate embeddings for HGCN training
    if last_checkpoint:
        console.print('\n[bold cyan]Generate embeddings for HGCN training?[/bold cyan]')
        generate_embeddings = typer.confirm(
            f'Generate embeddings parquet file from final checkpoint ({Path(last_checkpoint).name})?',
            default=False
        )

        if generate_embeddings:
            # Use the config from the last stage
            final_cfg = Config.from_yaml(config_file)

            logger.info('Generating embeddings from final checkpoint...')
            embeddings_path = generate_embeddings_from_checkpoint(
                checkpoint_path=last_checkpoint,
                config=final_cfg,
                output_path=None  # Will use default location
            )
            console.print(
                f'\n[bold green]✓ Embeddings generated successfully![/bold green]\n'
                f'Embeddings saved to: [cyan]{embeddings_path}[/cyan]\n'
                f'This file can be used for HGCN training.\n'
            )
