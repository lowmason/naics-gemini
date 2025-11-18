# -------------------------------------------------------------------------------------------------
# Imports
# -------------------------------------------------------------------------------------------------

import logging
import os
import time
import warnings
from pathlib import Path
from typing import List, Optional

import pytorch_lightning as pyl
import typer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from rich.console import Console
from rich.panel import Panel
from typing_extensions import Annotated

from naics_embedder.data_generation.compute_distances import calculate_pairwise_distances
from naics_embedder.data_generation.compute_relations import calculate_pairwise_relations
from naics_embedder.data_generation.create_triplets import generate_training_triplets
from naics_embedder.data_generation.download_data import download_preprocess_data
from naics_embedder.data_loader.datamodule import NAICSDataModule
from naics_embedder.model.naics_model import NAICSContrastiveModel
from naics_embedder.utils.backend import get_device
from naics_embedder.utils.config import ChainConfig, Config, list_available_curricula, parse_override_value
from naics_embedder.utils.console import configure_logging
from naics_embedder.tools.config_tools import show_current_config
from naics_embedder.tools.gpu_tools import optimize_gpu_config, detect_gpu_memory
from naics_embedder.tools.metrics_tools import visualize_metrics, investigate_hierarchy

# Set CUDA memory allocator configuration to reduce fragmentation
os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'

console = Console()
logger = logging.getLogger(__name__)


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


# -------------------------------------------------------------------------------------------------
# Setup Typer App
# -------------------------------------------------------------------------------------------------

app = typer.Typer(
    help=Panel.fit(
        '[bold cyan]NAICS Embedder[/bold cyan]\n\nText-enhanced Hyperbolic NAICS Embedding System',
        border_style='cyan',
        padding=(1, 2),
    ) #type: ignore
)
data_app = typer.Typer(help='Manage and generate project datasets.')
app.add_typer(data_app, name='data')

tools_app = typer.Typer(help='Utility tools for configuration, GPU optimization, and metrics analysis.')
app.add_typer(tools_app, name='tools')


# -------------------------------------------------------------------------------------------------
# Data generation commands
# -------------------------------------------------------------------------------------------------

@data_app.command('preprocess')
def run_preprocess():

    '''
    Download and preprocess all raw NAICS data files.
    
    Generates: data/naics_descriptions.parquet
    '''
    
    configure_logging('data_preprocess.log')

    console.rule('[bold green]Stage 1: Preprocessing[/bold green]')

    download_preprocess_data()

    console.print('\n[bold]Preprocessing complete.[/bold]\n')


@data_app.command('relations')
def run_relations():

    '''
    Compute pairwise graph relationships between all NAICS codes.
    
    Requires: data/naics_descriptions.parquet
    Generates: data/naics_relations.parquet
    '''

    configure_logging('data_relations.log')
    
    console.rule('[bold green]Stage 2: Computing Relations[/bold green]')

    calculate_pairwise_relations()

    console.print('\n[bold]Relation computation complete.[/bold]\n')


@data_app.command('distances')
def run_distances():

    '''
    Compute pairwise graph distances between all NAICS codes.
    
    Requires: data/naics_descriptions.parquet
    Generates: data/naics_distances.parquet
    '''

    configure_logging('data_distances.log')
    
    console.rule('[bold green]Stage 2: Computing Distances[/bold green]')

    calculate_pairwise_distances()

    console.print('\n[bold]Distance computation complete.[/bold]\n')


@data_app.command('triplets')
def run_triplets():

    '''
    Generate (anchor, positive, negative) training triplets.
    
    Requires: data/naics_descriptions.parquet, data/naics_distances.parquet
    Generates: data/naics_training_pairs.parquet
    '''

    configure_logging('data_triplets.log')

    console.rule('[bold green]Stage 3: Generating Triplets[/bold green]')

    generate_training_triplets()

    console.print('\n[bold]Triplet generation complete.[/bold]\n')


@data_app.command('all')
def run_all_data_gen():

    '''
    Run the full data generation pipeline: preprocess, distances, and triplets.
    '''

    configure_logging('data_all.log' )

    console.rule('[bold green]Starting Full Data Pipeline[/bold green]')
    
    run_preprocess()
    run_relations()
    run_distances()
    run_triplets()
    
    console.rule('[bold green]Full Data Pipeline Complete![/bold green]')


# -------------------------------------------------------------------------------------------------
# Model training: single curricula
# -------------------------------------------------------------------------------------------------

@app.command('train')
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
        
    except Exception as e:
        logger.error(f'Training failed: {e}', exc_info=True)
        console.print(f'\n[bold red]✗ Training failed:[/bold red] {e}\n')
        raise typer.Exit(code=1)


# -------------------------------------------------------------------------------------------------
# Model training: curriculum
# -------------------------------------------------------------------------------------------------

@app.command('train-curriculum')
def train_sequential(
    curricula: Annotated[
        Optional[List[str]],
        typer.Option(
            '--curricula',
            '-c',
            help="List of curriculum configs (e.g., '01_text,02_text,03_text' or '01_graph,02_graph'). Ignored if --chain is provided.",
        ),
    ] = None,
    chain: Annotated[
        Optional[str],
        typer.Option(
            '--chain',
            help='Chain configuration file (e.g., "chain_text" or "chain_graph"). Overrides --curricula.',
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
    ] = True,
    ckpt_path: Annotated[
        Optional[str],
        typer.Option(
            '--ckpt-path',
            help='Path to checkpoint file to resume from at start, or "last" to auto-detect last checkpoint',
        ),
    ] = None,
):
    
    configure_logging('train_sequential.log')
    
    # Handle graph training
    if curriculum_type == 'graph':
        from naics_embedder.utils.config import ChainConfig as GraphChainConfig
        from naics_embedder.model.hgcn import main as hgcn_main
        
        # Load chain configuration if provided
        if chain:
            chain_config = GraphChainConfig.from_yaml(chain, curriculum_type='graph')
            curricula = chain_config.get_stage_names()
            console.rule(f'[bold green]Graph Curriculum Training - {chain_config.chain_name}[/bold green]')
            console.print(f'[bold]Chain:[/bold] {chain_config.chain_name}')
            console.print(f'[bold]Stages to run:[/bold] {", ".join(curricula)}\n')
            hgcn_main(config_file=config_file, chain_file=chain, curriculum_stages=None)
        else:
            # Use provided curricula or default
            if curricula is None:
                curricula = ['01_graph', '02_graph', '03_graph', '04_graph', '05_graph', '06_graph']
            console.rule('[bold green]Graph Curriculum Training[/bold green]')
            console.print(f'[bold]Stages to run:[/bold] {", ".join(curricula)}\n')
            hgcn_main(config_file=config_file, chain_file=None, curriculum_stages=curricula)
        return
    
    # Text training (original logic)
    # Load chain configuration if provided
    chain_config = None
    if chain:
        chain_path = Path('conf/text_curriculum') / f'{chain}.yaml'
        if not chain_path.exists():
            console.print(f'[bold red]Error:[/bold red] Chain config not found: {chain_path}')
            raise typer.Exit(code=1)
        
        chain_config = ChainConfig.from_yaml(str(chain_path), curriculum_type='text')
        curricula = chain_config.get_stage_names()
        console.rule(f'[bold green]Sequential Curriculum Training - {chain_config.chain_name}[/bold green]')
        console.print(f'[bold]Chain:[/bold] {chain_config.chain_name}')
        console.print(f'[bold]Stages to run:[/bold] {", ".join(curricula)}\n')
    else:
        # Use default curricula if not provided
        if curricula is None:
            curricula = ['01_text', '02_text', '03_text', '04_text', '05_text']
        console.rule('[bold green]Sequential Curriculum Training[/bold green]')
        console.print(f'[bold]Stages to run:[/bold] {", ".join(curricula)}\n')
    
    last_checkpoint = None
    start_stage_index = 0  # Index in curricula list to start from
    
    # Handle initial checkpoint if provided
    if ckpt_path:
        # Check if user wants to auto-detect last checkpoint (supports both 'last' and 'last.ckpt')
        ckpt_path_lower = ckpt_path.lower()
        if ckpt_path_lower == 'last' or ckpt_path_lower == 'last.ckpt':
            # Try to find last checkpoint from sequential training
            # We need to determine the checkpoint directory structure
            if chain_config:
                stages = '-'.join(s.split('_')[0] for s in chain_config.get_stage_names())
            elif curricula:
                stages = '-'.join(s.split('_')[0] for s in curricula)
            else:
                stages = '01-02-03-04-05'  # Default
            
            # Try to find the last checkpoint from the last stage
            # We'll check the first curriculum's config to get the checkpoint dir
            temp_cfg = Config.from_yaml(config_file, curriculum_name=curricula[0] if curricula else '01_text')
            sequential_dir = Path(f'{temp_cfg.dirs.checkpoint_dir}/sequential_{stages}')
            
            # Find the last stage that has a checkpoint
            for curriculum in reversed(curricula if curricula else ['01_text', '02_text', '03_text', '04_text', '05_text']):
                stage_checkpoint_dir = sequential_dir / curriculum
                last_ckpt = stage_checkpoint_dir / 'last.ckpt'
                if last_ckpt.exists():
                    last_checkpoint = str(last_ckpt)
                    # Find which stage this checkpoint belongs to
                    if curricula:
                        try:
                            start_stage_index = curricula.index(curriculum)
                        except ValueError:
                            start_stage_index = 0
                    logger.info(f'Auto-detected last checkpoint: {last_checkpoint}')
                    logger.info(f'Checkpoint belongs to stage: {curriculum} (index {start_stage_index})')
                    console.print(f'[green]✓[/green] Resuming from last checkpoint: [cyan]{last_checkpoint}[/cyan]')
                    console.print(f'[cyan]Will resume from stage: {curriculum} (stage {start_stage_index + 1}/{len(curricula)})[/cyan]\n')
                    break
            
            if not last_checkpoint:
                console.print(f'[yellow]Warning:[/yellow] Last checkpoint not found in sequential training directory')
                console.print('Starting training from scratch.\n')
        else:
            # Use provided checkpoint path
            checkpoint_path_obj = Path(ckpt_path)
            if checkpoint_path_obj.exists():
                last_checkpoint = str(checkpoint_path_obj.resolve())
                # Extract stage name from checkpoint path
                # Path format: .../sequential_01-02-03/02_text/last.ckpt
                path_parts = Path(last_checkpoint).parts
                for part in path_parts:
                    if part.endswith('_text'):
                        stage_name = part
                        if curricula:
                            try:
                                start_stage_index = curricula.index(stage_name)
                            except ValueError:
                                start_stage_index = 0
                        logger.info(f'Checkpoint belongs to stage: {stage_name} (index {start_stage_index})')
                        console.print(f'[green]✓[/green] Resuming from checkpoint: [cyan]{last_checkpoint}[/cyan]')
                        console.print(f'[cyan]Will resume from stage: {stage_name} (stage {start_stage_index + 1}/{len(curricula)})[/cyan]\n')
                        break
                else:
                    logger.info(f'Using initial checkpoint: {last_checkpoint}')
                    console.print(f'[green]✓[/green] Resuming from checkpoint: [cyan]{last_checkpoint}[/cyan]\n')
            else:
                console.print(f'[yellow]Warning:[/yellow] Checkpoint not found at {ckpt_path}')
                console.print('Starting training from scratch.\n')
    
    # Start from the stage that contains the checkpoint
    for i, curriculum in enumerate(curricula[start_stage_index:], start_stage_index + 1):
        console.rule(
            f'[bold cyan]Stage {i}/{len(curricula)}: {curriculum}[/bold cyan]'
        )
        
        try:

            # Load configuration for this curriculum
            cfg = Config.from_yaml(config_file, curriculum_name=curriculum, curriculum_type='text')
            
            # Apply chain-specific overrides if chain config is provided
            if chain_config:
                stage_overrides = chain_config.get_stage_overrides(curriculum)
                if stage_overrides:
                    logger.info(f'Applying chain overrides for {curriculum}: {stage_overrides}')
                    cfg = cfg.override(stage_overrides)
            
            # Setup checkpoint directory for this stage
            stages = '-'.join(s.split('_')[0] for s in curricula)
            checkpoint_dir = Path(f'{cfg.dirs.checkpoint_dir}/sequential_{stages}/{curriculum}')
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            # Check if this stage already has a checkpoint
            existing_checkpoints = list(checkpoint_dir.glob(f'{curriculum}-*.ckpt'))
            last_ckpt = checkpoint_dir / 'last.ckpt'
            
            # Check if we're resuming from a checkpoint that belongs to this stage
            resuming_this_stage = False
            if ckpt_path and last_checkpoint:
                # Check if the checkpoint path matches this stage's checkpoint
                if str(last_ckpt.resolve()) == Path(last_checkpoint).resolve():
                    resuming_this_stage = True
                    logger.info(f'Resuming training within stage {curriculum} from checkpoint')
            
            # Note: We don't skip stages here because the loop already starts from start_stage_index
            # If we're resuming this stage, we'll handle it below by passing ckpt_path to trainer.fit()
            
            # Check device
            accelerator, precision, num_devices = get_device(log_info=(i == start_stage_index + 1))
            
            # Seed for reproducibility
            pyl.seed_everything(cfg.seed + i, verbose=False)
            
            # Initialize DataModule with curriculum-specific config
            datamodule = NAICSDataModule(
                descriptions_path=cfg.data_loader.streaming.descriptions_parquet,
                triplets_path=cfg.data_loader.streaming.triplets_parquet,
                tokenizer_name=cfg.data_loader.tokenization.tokenizer_name,
                streaming_config=cfg.curriculum.model_dump(),
                batch_size=cfg.data_loader.batch_size,
                num_workers=cfg.data_loader.num_workers,
                val_split=cfg.data_loader.val_split,
                seed=cfg.seed + i
            )
            
            # Initialize or load model
            # If we're resuming within this stage, trainer.fit() will load from checkpoint
            # Otherwise, if we have a checkpoint from a previous stage, load it for continuation
            if resuming_this_stage:
                # Don't load manually - trainer.fit() will load from checkpoint with epoch info
                logger.info(f'Will resume from checkpoint in trainer.fit() - initializing model structure')
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
                    learning_rate=cfg.training.learning_rate,
                    weight_decay=cfg.training.weight_decay,
                    warmup_steps=cfg.training.warmup_steps,
                    load_balancing_coef=cfg.model.moe.load_balancing_coef,
                    distance_matrix_path=cfg.data_loader.streaming.distance_matrix_parquet,
                    eval_every_n_epochs=cfg.model.eval_every_n_epochs,
                    eval_sample_size=cfg.model.eval_sample_size
                )
                console.print('[cyan]Model structure initialized - will load weights from checkpoint[/cyan]\n')
            elif last_checkpoint and resume_from_checkpoint:
                # Loading from previous stage checkpoint for next stage
                logger.info(f'Loading model from previous stage checkpoint: {last_checkpoint}')
                model = NAICSContrastiveModel.load_from_checkpoint(
                    last_checkpoint,
                    # Override learning rate for new stage if needed
                    learning_rate=cfg.training.learning_rate,
                )
                console.print('[green]✓[/green] Resumed from previous stage checkpoint\n')

            else:
                logger.info('Initializing new model...')
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
                    learning_rate=cfg.training.learning_rate,
                    weight_decay=cfg.training.weight_decay,
                    warmup_steps=cfg.training.warmup_steps,
                    load_balancing_coef=cfg.model.moe.load_balancing_coef,
                    distance_matrix_path=cfg.data_loader.streaming.distance_matrix_parquet,
                    eval_every_n_epochs=cfg.model.eval_every_n_epochs,
                    eval_sample_size=cfg.model.eval_sample_size
                )
                if i == 1:
                    console.print('[green]✓[/green] Initialized new model\n')

                else:
                    console.print(
                        '[yellow]![/yellow] No checkpoint found, '
                        'using random initialization\n'
                    )
            
            # Setup callbacks with stage-specific checkpoint directory
            checkpoint_callback = ModelCheckpoint(
                dirpath=checkpoint_dir,
                filename=f'{curriculum}-{{epoch:02d}}-{{val/contrastive_loss:.4f}}',
                monitor='val/contrastive_loss',
                mode='min',
                save_top_k=1,
                save_last=True
            )
            
            # Early stopping with stage-appropriate patience
            early_stopping = EarlyStopping(
                monitor='val/contrastive_loss',
                patience=2 if i < len(curricula) else 3,  # Reduced to prevent overfitting
                mode='min',
                min_delta=0.0001,  # Minimum improvement required
                verbose=True
            )
            
            # TensorBoard logger with stage info
            # Create the log directory first to avoid FileNotFoundError
            tb_log_dir = Path(f'{cfg.dirs.output_dir}/sequential_{stages}/{curriculum}')
            tb_log_dir.mkdir(parents=True, exist_ok=True)
            
            tb_logger = TensorBoardLogger(
                save_dir=cfg.dirs.output_dir,
                name=f'sequential_{stages}',
                version=curriculum
            )
            
            # Initialize Trainer
            trainer = pyl.Trainer(
                max_epochs=cfg.training.trainer.max_epochs,
                accelerator=accelerator,
                devices=num_devices,
                precision=precision, # type: ignore
                gradient_clip_val=cfg.training.trainer.gradient_clip_val,
                accumulate_grad_batches=cfg.training.trainer.accumulate_grad_batches,
                log_every_n_steps=cfg.training.trainer.log_every_n_steps,
                val_check_interval=cfg.training.trainer.val_check_interval,
                callbacks=[checkpoint_callback, early_stopping],
                logger=tb_logger,
                default_root_dir=cfg.dirs.output_dir
            )
            
            # Train this stage
            logger.info(f'Starting training for stage: {curriculum}\n')
            
            # Check if we should resume training within this stage
            stage_checkpoint_path = None
            if resume_from_checkpoint and resuming_this_stage and last_checkpoint:
                # We're resuming this specific stage from a checkpoint
                # Use the checkpoint path directly (it contains epoch information)
                stage_checkpoint_path = last_checkpoint
                logger.info(f'Resuming training within stage {curriculum} from: {stage_checkpoint_path}')
                console.print(f'[cyan]Resuming stage {curriculum} from checkpoint (will continue from saved epoch)[/cyan]\n')
            
            trainer.fit(model, datamodule, ckpt_path=stage_checkpoint_path)
            
            # Store checkpoint path for next stage
            last_checkpoint = checkpoint_callback.best_model_path
            
            # Check if early stopping was triggered and get the best loss
            early_stop_triggered = early_stopping.stopped_epoch > 0
            best_loss = early_stopping.best_score if early_stopping.best_score is not None else None
            
            if early_stop_triggered and best_loss is not None:
                logger.info(f'Early stopping triggered at epoch {early_stopping.stopped_epoch} with best loss: {best_loss:.6f}')
            
            console.print(
                f'[green]✓[/green] Stage {i} complete. '
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
            
            # Optional: Run evaluation on test set between stages
            if i < len(curricula):
                console.print('[dim]Preparing for next stage...[/dim]\n')
                time.sleep(2)  # Brief pause between stages
            
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


# -------------------------------------------------------------------------------------------------
# Tools commands
# -------------------------------------------------------------------------------------------------

@tools_app.command('config')
def show_config(
    config_file: Annotated[
        str,
        typer.Option(
            '--config',
            help='Path to base config YAML file',
        ),
    ] = 'conf/config.yaml',
):
    """
    Display current training and curriculum configuration.
    """
    show_current_config(config_file)


@tools_app.command('gpu')
def optimize_gpu(
    gpu_memory: Annotated[
        Optional[float],
        typer.Option(
            '--gpu-memory',
            help='GPU memory in GB (e.g., 24 for RTX 6000, 80 for A100). Use --auto to detect automatically.',
        ),
    ] = None,
    auto: Annotated[
        bool,
        typer.Option(
            '--auto',
            help='Auto-detect GPU memory',
        ),
    ] = False,
    target_effective_batch: Annotated[
        int,
        typer.Option(
            '--target-effective-batch',
            help='Target effective batch size (batch_size * accumulate_grad_batches)',
        ),
    ] = 256,
    apply: Annotated[
        bool,
        typer.Option(
            '--apply',
            help='Apply suggested configuration to config files',
        ),
    ] = False,
    config_file: Annotated[
        str,
        typer.Option(
            '--config',
            help='Path to base config YAML file',
        ),
    ] = 'conf/config.yaml',
):
    """
    Optimize training configuration for available GPU memory.
    
    Suggests optimal batch_size and accumulate_grad_batches based on your GPU.
    """
    configure_logging('gpu_config.log')
    
    if not auto and gpu_memory is None:
        console.print('[bold red]Error:[/bold red] Must specify either --gpu-memory or --auto')
        raise typer.Exit(code=1)
    
    try:
        result = optimize_gpu_config(
            gpu_memory_gb=gpu_memory,
            auto_detect=auto,
            target_effective_batch=target_effective_batch,
            apply=apply,
            config_path=config_file
        )
        
        console.print('\n[bold green]GPU Configuration Optimization[/bold green]\n')
        console.print(f'GPU Memory: {result["gpu_memory_gb"]:.1f} GB\n')
        
        for i, config in enumerate(result['suggestions'], 1):
            console.print(f'[bold]Configuration {i}:[/bold] {config["stage"]}')
            console.print(f'  • batch_size: {config["batch_size"]}')
            console.print(f'  • n_positives: {config["n_positives"]}')
            console.print(f'  • n_negatives: {config["n_negatives"]}')
            console.print(f'  • accumulate_grad_batches: {config["accumulate_grad_batches"]}')
            console.print(f'  • Effective batch size: {config["effective_batch_size"]}')
            console.print(f'  • Memory utilization: {config["memory_utilization"]}')
            console.print(str(config['memory_estimate']))
            console.print()
        
        if result['applied']:
            console.print('[bold green]✓ Configuration files updated successfully![/bold green]')
            console.print('  Backup files created with .backup extension\n')
        elif not apply:
            console.print('[yellow]Tip:[/yellow] Use --apply to automatically update config files\n')
            
    except Exception as e:
        console.print(f'[bold red]Error:[/bold red] {e}')
        raise typer.Exit(code=1)


@tools_app.command('visualize')
def visualize(
    stage: Annotated[
        str,
        typer.Option(
            '--stage',
            '-s',
            help="Stage name to filter (e.g., '02_text')",
        ),
    ] = '02_text',
    log_file: Annotated[
        Optional[str],
        typer.Option(
            '--log-file',
            help='Path to log file (default: logs/train_sequential.log)',
        ),
    ] = None,
    output_dir: Annotated[
        Optional[str],
        typer.Option(
            '--output-dir',
            help='Output directory for plots (default: outputs/visualizations/)',
        ),
    ] = None,
):
    """
    Visualize training metrics from log files.
    
    Creates comprehensive visualizations and analysis of training metrics including:
    - Hyperbolic radius over time
    - Hierarchy preservation correlations
    - Embedding diversity metrics
    """
    try:
        log_path = Path(log_file) if log_file else None
        output_path = Path(output_dir) if output_dir else None
        
        result = visualize_metrics(
            stage=stage,
            log_file=log_path,
            output_dir=output_path
        )
        
        if result.get('output_file'):
            console.print(f'\n[bold green]✓[/bold green] Visualization saved to: [cyan]{result["output_file"]}[/cyan]\n')
        
    except Exception as e:
        console.print(f'[bold red]Error:[/bold red] {e}')
        raise typer.Exit(code=1)


@tools_app.command('investigate')
def investigate(
    distance_matrix: Annotated[
        Optional[str],
        typer.Option(
            '--distance-matrix',
            help='Path to ground truth distance matrix (default: data/naics_distance_matrix.parquet)',
        ),
    ] = None,
    config_file: Annotated[
        Optional[str],
        typer.Option(
            '--config',
            help='Path to config file (default: conf/config.yaml)',
        ),
    ] = None,
):
    """
    Investigate why hierarchy preservation correlations might be low.
    
    Analyzes ground truth distances, evaluation configuration, and provides
    recommendations for improving hierarchy preservation metrics.
    """
    try:
        dist_path = Path(distance_matrix) if distance_matrix else None
        config_path = Path(config_file) if config_file else None
        
        result = investigate_hierarchy(
            distance_matrix_path=dist_path,
            config_path=config_path
        )
        
        console.print('\n[bold green]Investigation complete![/bold green]\n')
        
    except Exception as e:
        console.print(f'[bold red]Error:[/bold red] {e}')
        raise typer.Exit(code=1)