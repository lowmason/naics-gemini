# -------------------------------------------------------------------------------------------------
# Training Commands
# -------------------------------------------------------------------------------------------------

import logging
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import polars as pl
import pytorch_lightning as pyl
import torch
import typer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from rich.console import Console
from rich.panel import Panel
from typing_extensions import Annotated

from naics_embedder.data_loader.datamodule import NAICSDataModule
from naics_embedder.data_loader.tokenization_cache import tokenization_cache
from naics_embedder.model.naics_model import NAICSContrastiveModel
from naics_embedder.utils.backend import get_device
from naics_embedder.utils.utilities import pick_device
from naics_embedder.utils.config import ChainConfig, Config, TokenizationConfig, list_available_curricula, parse_override_value
from naics_embedder.utils.console import configure_logging

# -------------------------------------------------------------------------------------------------
# Suppress warnings
# -------------------------------------------------------------------------------------------------

warnings.filterwarnings(
    'ignore',
    message='.*Precision.*is not supported by the model summary.*',
    category=UserWarning,
    module='pytorch_lightning.utilities.model_summary.model_summary'
)

warnings.filterwarnings(
    'ignore',
    message='.*Found .* module.*in eval mode.*',
    category=UserWarning,
    module='pytorch_lightning'
)

warnings.filterwarnings(
    'ignore',
    message='.*does not have many workers.*',
    category=UserWarning,
    module='pytorch_lightning'
)

warnings.filterwarnings(
    'ignore',
    message='.*Checkpoint directory.*exists and is not empty.*',
    category=UserWarning,
    module='pytorch_lightning'
)

warnings.filterwarnings(
    'ignore',
    message='.*Trying to infer the.*batch_size.*',
    category=UserWarning,
    module='pytorch_lightning'
)

console = Console()
logger = logging.getLogger(__name__)


# -------------------------------------------------------------------------------------------------
# Embedding Generation
# -------------------------------------------------------------------------------------------------

def generate_embeddings_from_checkpoint(
    checkpoint_path: str,
    config: Config,
    output_path: Optional[str] = None,
    curriculum_name: Optional[str] = None,
    batch_size: int = 32
) -> str:
    """
    Generate hyperbolic embeddings parquet file from a trained checkpoint.
    
    This function loads a trained model checkpoint, runs inference on all NAICS codes,
    and saves the hyperbolic embeddings to a parquet file in the format expected by HGCN training.
    
    Args:
        checkpoint_path: Path to the PyTorch Lightning checkpoint file
        config: Config object with model and data paths
        output_path: Optional output path for embeddings parquet. If None, uses default location.
        curriculum_name: Optional curriculum name for output directory naming
        batch_size: Batch size for inference
        
    Returns:
        Path to the generated embeddings parquet file
    """
    logger.info('=' * 80)
    logger.info('GENERATING EMBEDDINGS FROM CHECKPOINT')
    logger.info('=' * 80)
    logger.info(f'Checkpoint: {checkpoint_path}')
    
    # Determine output path
    if output_path is None:
        # Use default location: ./output/hyperbolic_projection/encodings.parquet
        output_dir = Path('./output/hyperbolic_projection')
        if curriculum_name:
            output_dir = output_dir / curriculum_name
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
    curriculum: Annotated[
        str,
        typer.Option(
            '--curriculum',
            '-c',
            help="Curriculum config name (e.g., '01_text', '02_text', '01_graph', '02_graph')",
        ),
    ] = 'default',
    curriculum_type: Annotated[
        str,
        typer.Option(
            '--curriculum-type',
            help="Curriculum type: 'text' or 'graph' (default: 'text')",
        ),
    ] = 'text',
    config_file: Annotated[
        str,
        typer.Option(
            '--config',
            help='Path to base config YAML file',
        ),
    ] = 'conf/config.yaml',
    list_curricula: Annotated[
        bool,
        typer.Option(
            '--list-curricula',
            help='List available curricula and exit',
        ),
    ] = False,
    ckpt_path: Annotated[
        Optional[str],
        typer.Option(
            '--ckpt-path',
            help='Path to checkpoint file to resume from, or "last" to auto-detect last checkpoint',
        ),
    ] = None,
    overrides: Annotated[
        Optional[List[str]],
        typer.Argument(
            help="Config overrides (e.g., 'training.learning_rate=1e-4 data.batch_size=64')"
        ),
    ] = None,
):
    
    # List curricula and exit
    if list_curricula:
        curricula = list_available_curricula()
        console.print('\n[bold]Available Curricula:[/bold]')
        for curr in curricula:
            console.print(f'  • {curr}')
        console.print('')
        return
    
    configure_logging('train.log')
    
    console.rule(
        f"[bold green]Training NAICS: Curriculum '[cyan]{curriculum}[/cyan]'[/bold green]"
    )
    
    try:

        # Check device
        logger.info('Determining infrastructure...')
        accelerator, precision, num_devices = get_device(log_info=True)
        
        # Check GPU memory status if CUDA is available
        gpu_memory_info = None
        if accelerator in ['cuda', 'gpu']:
            try:
                import torch
                if torch.cuda.is_available():
                    device = torch.cuda.current_device()
                    total_memory = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)  # GB
                    reserved_memory = torch.cuda.memory_reserved(device) / (1024 ** 3)  # GB
                    allocated_memory = torch.cuda.memory_allocated(device) / (1024 ** 3)  # GB
                    free_memory = total_memory - reserved_memory
                    
                    gpu_memory_info = {
                        'total_gb': total_memory,
                        'reserved_gb': reserved_memory,
                        'allocated_gb': allocated_memory,
                        'free_gb': free_memory,
                        'utilization_pct': (reserved_memory / total_memory) * 100 if total_memory > 0 else 0
                    }
                    
                    logger.info(
                        f'GPU Memory: {reserved_memory:.1f} GB used / {total_memory:.1f} GB total '
                        f'({gpu_memory_info["utilization_pct"]:.1f}% utilization, '
                        f'{free_memory:.1f} GB free)'
                    )
            except Exception as e:
                logger.debug(f'Could not get GPU memory info: {e}')

        # Load configuration
        logger.info('Loading configuration...')
        if curriculum_type == 'graph':
            # For graph training, use GraphConfig and hgcn module
            from naics_embedder.utils.config import GraphConfig
            from naics_embedder.model.hgcn import main as hgcn_main
            cfg = GraphConfig.from_yaml(config_file, curriculum_name=curriculum, curriculum_type='graph')
            # Call HGCN training directly
            hgcn_main(config_file=config_file, curriculum_stages=[curriculum] if curriculum != 'default' else None)
            return
        else:
            cfg = Config.from_yaml(config_file, curriculum_name=curriculum, curriculum_type='text')
        
        # Apply command-line overrides
        if overrides:
            override_dict = {}
            logger.info('Applying command-line overrides:')
            for override in overrides:
                if '=' not in override:
                    console.print(
                        f'[yellow]Warning:[/yellow] Skipping invalid override: {override}'
                    )
                    continue
                
                key, value_str = override.split('=', 1)
                value = parse_override_value(value_str)
                override_dict[key] = value
                logger.info(f'  • {key} = {value} ({type(value).__name__})')

            logger.info('')
            
            cfg = cfg.override(override_dict)
        
        # Display configuration summary w/ curriculum
        summary_list_1 = [
            f'[bold]Experiment:[/bold] {cfg.experiment_name}',
            f'[bold]Curriculum:[/bold] {cfg.curriculum.name}',
            f'[bold]Seed:[/bold] {cfg.seed}\n',
            '[cyan]Data:[/cyan]',
            f'  • Batch size: {cfg.data_loader.batch_size}',
            f'  • Num workers: {cfg.data_loader.num_workers}\n',
            '[cyan]Curriculum:[/cyan]',
        ]

        summary_list_2 = []
        for k, v in cfg.curriculum.model_dump().items():
            if k != 'name' and v is not None:
                summary_list_2.append(f'  • {k}: {v}')

        summary_list_3 = [
            '\n[cyan]Model:[/cyan]',
            f'  • Base: {cfg.model.base_model_name.split("/")[-1]}',
            f'  • LoRA rank: {cfg.model.lora.r}',
            '  • MoE: ',
            f'    - {cfg.model.moe.num_experts} experts\n',
            '[cyan]Training:[/cyan]',
            f'  • Learning rate: {cfg.training.learning_rate}',
            f'  • Max epochs: {cfg.training.trainer.max_epochs}',
            f'  • Accelerator: {accelerator}',
            f'  • Precision: {precision}'
        ]
        
        # Add GPU memory info and batch size suggestions
        summary_list_4 = []
        if gpu_memory_info:
            summary_list_4.append('\n[cyan]GPU Memory:[/cyan]')
            summary_list_4.append(
                f'  • Used: {gpu_memory_info["reserved_gb"]:.1f} GB / '
                f'{gpu_memory_info["total_gb"]:.1f} GB '
                f'({gpu_memory_info["utilization_pct"]:.1f}% utilization)'
            )
            summary_list_4.append(f'  • Free: {gpu_memory_info["free_gb"]:.1f} GB')
            
            # Conservative batch size suggestion
            current_batch_size = cfg.data_loader.batch_size
            if gpu_memory_info["free_gb"] > 8.0 and current_batch_size < 12:
                # Suggest 2x-3x current batch size conservatively
                suggested_batch = min(12, current_batch_size * 2)
                if suggested_batch > current_batch_size:
                    summary_list_4.append(
                        f'\n[yellow]Batch Size Suggestion:[/yellow]'
                    )
                    summary_list_4.append(
                        f'  • Current: {current_batch_size}'
                    )
                    summary_list_4.append(
                        f'  • Suggested: {suggested_batch} (conservative estimate)'
                    )
                    summary_list_4.append(
                        f'  • [dim]Note: gpu_tools.py estimates are optimistic; '
                        f'reduce suggested values by ~50%[/dim]'
                    )
                    summary_list_4.append(
                        f'  • [dim]Override with: data.batch_size={suggested_batch}[/dim]'
                    )

        summary = '\n'.join(summary_list_1 + summary_list_2 + summary_list_3 + summary_list_4)

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
            streaming_config=cfg.curriculum.model_dump(),
            batch_size=cfg.data_loader.batch_size,
            num_workers=cfg.data_loader.num_workers,
            val_split=cfg.data_loader.val_split,
            seed=cfg.seed
        )
        
        # Handle checkpoint resumption
        checkpoint_path = None
        if ckpt_path:
            # Check if user wants to auto-detect last checkpoint (supports both 'last' and 'last.ckpt')
            ckpt_path_lower = ckpt_path.lower()
            if ckpt_path_lower == 'last' or ckpt_path_lower == 'last.ckpt':
                # Auto-detect last checkpoint
                checkpoint_dir = Path(cfg.dirs.checkpoint_dir) / cfg.experiment_name
                last_ckpt = checkpoint_dir / 'last.ckpt'
                if last_ckpt.exists():
                    checkpoint_path = str(last_ckpt)
                    logger.info(f'Auto-detected last checkpoint: {checkpoint_path}')
                    console.print(f'[green]✓[/green] Resuming from last checkpoint: [cyan]{checkpoint_path}[/cyan]\n')
                else:
                    console.print(f'[yellow]Warning:[/yellow] Last checkpoint not found at {last_ckpt}')
                    console.print('Starting training from scratch.\n')
            else:
                # Use provided checkpoint path
                checkpoint_path_obj = Path(ckpt_path)
                if checkpoint_path_obj.exists():
                    checkpoint_path = str(checkpoint_path_obj.resolve())
                    logger.info(f'Using checkpoint: {checkpoint_path}')
                    console.print(f'[green]✓[/green] Resuming from checkpoint: [cyan]{checkpoint_path}[/cyan]\n')
                else:
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
                eval_sample_size=cfg.model.eval_sample_size
            )
        
        # Setup callbacks
        logger.info('Setting up callbacks and checkpointing...\n')
        checkpoint_dir = Path(cfg.dirs.checkpoint_dir) / cfg.experiment_name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if checkpoint is from the same stage (for resuming) or different stage (for loading weights)
        resuming_same_stage = False
        if checkpoint_path:
            checkpoint_path_obj = Path(checkpoint_path)
            # Check if checkpoint is in the same checkpoint directory (same stage)
            try:
                checkpoint_abs = checkpoint_path_obj.resolve()
                current_dir_abs = checkpoint_dir.resolve()
                # Check if checkpoint's parent directory matches the current stage's checkpoint directory
                resuming_same_stage = checkpoint_abs.parent == current_dir_abs
            except Exception:
                # If we can't determine, assume it's a different stage (safer)
                resuming_same_stage = False
            
            if resuming_same_stage:
                logger.info(f'Checkpoint is from same stage - will resume training from checkpoint')
                console.print('[cyan]Resuming training from checkpoint (will continue from saved epoch)[/cyan]\n')
            else:
                logger.info(f'Checkpoint is from different stage - will load weights only, start fresh training')
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
        if devices_to_use > 1 and accelerator in ['cuda', 'gpu']:
            from pytorch_lightning.strategies import DDPStrategy
            strategy = DDPStrategy(find_unused_parameters=True)
        
        trainer = pyl.Trainer(
            max_epochs=cfg.training.trainer.max_epochs,
            accelerator=accelerator,
            devices=devices_to_use,
            strategy=strategy,
            precision=precision, # type: ignore
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
            label = "Final evaluation metric (early stopping)" if early_stop_triggered else "Final evaluation metric"
            console.print(
                f'[bold]{label}:[/bold] '
                f'[cyan]val/contrastive_loss = {best_loss:.6f}[/cyan]\n'
            )
            logger.info(f'{label}: val/contrastive_loss = {best_loss:.6f}')
        
        # Save final config
        config_output_path = checkpoint_dir / 'config.yaml'
        cfg.to_yaml(str(config_output_path))
        console.print(f'Config saved: [cyan]{config_output_path}[/cyan]\n')
        
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
                output_path=None,  # Will use default location
                curriculum_name=curriculum
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
    curricula: Annotated[
        List[str],
        typer.Option(
            '--curricula',
            '-c',
            help="List of curriculum stages to run sequentially (e.g., '01_text 02_text 03_text')",
        ),
    ] = None,
    curriculum_type: Annotated[
        str,
        typer.Option(
            '--curriculum-type',
            help="Curriculum type: 'text' or 'graph' (default: 'text')",
        ),
    ] = 'text',
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
    overrides: Annotated[
        Optional[List[str]],
        typer.Argument(
            help="Config overrides (e.g., 'training.learning_rate=1e-4')"
        ),
    ] = None,
):
    """
    Run sequential curriculum training with automatic checkpoint handoff.
    
    Trains through multiple curriculum stages, automatically loading the best checkpoint
    from each stage as the initialization for the next stage.
    """
    
    # Your full train_sequential implementation here
    # (Lines 651-981 from your original cli.py)
    
    configure_logging('train_sequential.log')
    
    console.rule('[bold cyan]Starting Sequential Curriculum Training[/bold cyan]')
    
    # Use default curricula if none provided
    if curricula is None:
        curricula = ['01_text', '02_text', '03_text']
        console.print(f'[dim]Using default curricula: {curricula}[/dim]\n')
    
    console.print(f'[bold]Training sequence:[/bold] {" → ".join(curricula)}')
    console.print(f'[bold]Total stages:[/bold] {len(curricula)}\n')
    
    last_checkpoint = None
    
    # Train each curriculum stage
    for i, curriculum in enumerate(curricula, 1):
        console.rule(f'[bold green]Stage {i}/{len(curricula)}: {curriculum}[/bold green]')
        
        try:
            # Load config for this stage
            cfg = Config.from_yaml(
                config_file,
                curriculum_name=curriculum,
                curriculum_type=curriculum_type
            )
            
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
                label = "Final evaluation metric (early stopping)" if early_stop_triggered else "Final evaluation metric"
                console.print(
                    f'[bold]{label}:[/bold] '
                    f'[cyan]val/contrastive_loss = {best_loss:.6f}[/cyan]\n'
                )
                logger.info(f'{label}: val/contrastive_loss = {best_loss:.6f}')
            
            # Brief pause between stages
            if i < len(curricula):
                console.print('[dim]Preparing for next stage...[/dim]\n')
                time.sleep(2)
            
        except Exception as e:
            logger.error(f'Stage {curriculum} failed: {e}', exc_info=True)
            console.print(f'\n[bold red]✗ Stage {curriculum} failed:[/bold red] {e}\n')
            
            if i < len(curricula):
                response = typer.prompt(
                    f'Continue with next stage ({curricula[i]})? [y/N]',
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
            final_curriculum = curricula[-1] if curricula else 'default'
            final_cfg = Config.from_yaml(
                config_file,
                curriculum_name=final_curriculum,
                curriculum_type=curriculum_type
            )
            
            logger.info('Generating embeddings from final checkpoint...')
            embeddings_path = generate_embeddings_from_checkpoint(
                checkpoint_path=last_checkpoint,
                config=final_cfg,
                output_path=None,  # Will use default location
                curriculum_name=final_curriculum
            )
            console.print(
                f'\n[bold green]✓ Embeddings generated successfully![/bold green]\n'
                f'Embeddings saved to: [cyan]{embeddings_path}[/cyan]\n'
                f'This file can be used for HGCN training.\n'
            )
