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
from pytorch_lightning.callbacks import Callback, EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from rich.console import Console
from rich.panel import Panel
from typing_extensions import Annotated

from naics_gemini.data_generation.compute_distances import calculate_pairwise_distances
from naics_gemini.data_generation.compute_relations import calculate_pairwise_relations
from naics_gemini.data_generation.create_triplets import generate_training_triplets
from naics_gemini.data_generation.download_data import download_preprocess_data
from naics_gemini.data_loader.datamodule import NAICSDataModule
from naics_gemini.model.naics_model import NAICSContrastiveModel
from naics_gemini.utils.backend import get_device
from naics_gemini.utils.config import Config, list_available_curricula, parse_override_value
from naics_gemini.utils.console import configure_logging

# Set CUDA memory allocator configuration to reduce fragmentation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

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
        '[bold cyan]NAICS Gemini[/bold cyan]\n\nContrastive Learning for NAICS Code Embeddings.',
        border_style='cyan',
        padding=(1, 2),
    ) #type: ignore
)
data_app = typer.Typer(help='Manage and generate project datasets.')
app.add_typer(data_app, name='data')


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
            help="Curriculum config name (e.g., '01_stage', '02_stage')",
        ),
    ] = 'default',
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

        # Load configuration
        logger.info('Loading configuration...')
        cfg = Config.from_yaml(config_file, curriculum_name=curriculum)
        
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
            f'    - {"enabled" if cfg.model.moe.enabled else "disabled"} ',
            f'    - {cfg.model.moe.num_experts} experts\n',
            '[cyan]Training:[/cyan]',
            f'  • Learning rate: {cfg.training.learning_rate}',
            f'  • Max epochs: {cfg.training.trainer.max_epochs}',
            f'  • Accelerator: {accelerator}',
            f'  • Precision: {precision}'
        ]

        summary = '\n'.join(summary_list_1 + summary_list_2 + summary_list_3)

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
        
        # Construct distances path
        distances_path = cfg.data_loader.streaming.distances_parquet
        
        # Initialize Model
        logger.info('Initializing Model with evaluation metrics...\n')

        model = NAICSContrastiveModel(
            base_model_name=cfg.model.base_model_name,
            lora_r=cfg.model.lora.r,
            lora_alpha=cfg.model.lora.alpha,
            lora_dropout=cfg.model.lora.dropout,
            use_moe=cfg.model.moe.enabled,
            num_experts=cfg.model.moe.num_experts,
            top_k=cfg.model.moe.top_k,
            moe_hidden_dim=cfg.model.moe.hidden_dim,
            temperature=cfg.loss.temperature,
            curvature=cfg.loss.curvature,
            learning_rate=cfg.training.learning_rate,
            weight_decay=cfg.training.weight_decay,
            warmup_steps=cfg.training.warmup_steps,
            load_balancing_coef=cfg.model.moe.load_balancing_coef,
            # Evaluation settings
            distances_path=distances_path,
            eval_every_n_epochs=cfg.model.eval_every_n_epochs,
            eval_sample_size=cfg.model.eval_sample_size
        )
        
        # Setup callbacks
        logger.info('Setting up callbacks and checkpointing...\n')
        checkpoint_dir = Path(cfg.dirs.checkpoint_dir) / cfg.experiment_name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
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
            patience=5,
            mode='min'
        )
        
        # TensorBoard logger - ensure directory exists first
        tb_log_dir = Path(cfg.dirs.output_dir) / cfg.experiment_name
        tb_log_dir.mkdir(parents=True, exist_ok=True)
        
        tb_logger = TensorBoardLogger(
            save_dir=cfg.dirs.output_dir,
            name=cfg.experiment_name
        )
        
        # Don't add epoch progress callback - PyTorch Lightning's progress bar already shows epoch info
        
        # Initialize Trainer
        logger.info('Initializing PyTorch Lightning Trainer...\n')
        
        # Use only 1 device as specified in config, even if multiple GPUs are available
        devices_to_use = cfg.training.trainer.devices if hasattr(cfg.training.trainer, 'devices') else 1
        
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
        console.print('  • Spearman correlation (rank preservation)')
        console.print('  • Embedding statistics (norms, distances)')
        console.print('  • Collapse detection (variance, norm, distance)')
        console.print('  • Distortion metrics (mean, std)\n')
        
        console.print(f'[bold yellow]Training for {cfg.training.trainer.max_epochs} epochs...[/bold yellow]\n')
        
        trainer.fit(model, datamodule)
        
        # Training complete
        logger.info('Training complete!')
        logger.info(f'Best model checkpoint: {checkpoint_callback.best_model_path}')
        
        console.print(
            f'\n[bold green]✓ Training completed successfully![/bold green]\n'
            f'Best checkpoint: [cyan]{checkpoint_callback.best_model_path}[/cyan]\n'
        )
        
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
        List[str],
        typer.Option(
            '--curricula',
            '-c',
            help="List of curriculum configs (e.g., '01_stage,02_stage,03_stage')",
        ),
    ] = ['01_stage', '02_stage', '03_stage', '04_stage', '05_stage'],
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
):
    
    configure_logging('train_sequential.log')
    
    console.rule('[bold green]Sequential Curriculum Training[/bold green]')
    console.print(f'[bold]Stages to run:[/bold] {", ".join(curricula)}\n')
    
    last_checkpoint = None
    
    for i, curriculum in enumerate(curricula, 1):
        console.rule(
            f'[bold cyan]Stage {i}/{len(curricula)}: {curriculum}[/bold cyan]'
        )
        
        try:

            # Load configuration for this curriculum
            cfg = Config.from_yaml(config_file, curriculum_name=curriculum)
            
            # Setup checkpoint directory for this stage
            stages = '-'.join(s.split('_')[0] for s in curricula)
            checkpoint_dir = Path(f'{cfg.dirs.checkpoint_dir}/sequential_{stages}/{curriculum}')
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            # Check if this stage already has a checkpoint
            existing_checkpoints = list(checkpoint_dir.glob(f'{curriculum}-*.ckpt'))
            last_ckpt = checkpoint_dir / 'last.ckpt'
            
            if resume_from_checkpoint and existing_checkpoints and last_ckpt.exists():
                # Stage already complete - use its checkpoint and skip training
                last_checkpoint = str(last_ckpt)
                console.print(f'[green]✓[/green] Stage already complete: {curriculum}')
                console.print(f'[cyan]Using checkpoint:[/cyan] {last_checkpoint}\n')
                logger.info(f'Skipping stage {curriculum} - already trained')
                continue
            
            # Check device
            accelerator, precision, num_devices = get_device(log_info=(i == 1))
            
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
            if last_checkpoint and resume_from_checkpoint:
                logger.info(f'Loading model from checkpoint: {last_checkpoint}')
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
                    use_moe=cfg.model.moe.enabled,
                    num_experts=cfg.model.moe.num_experts,
                    top_k=cfg.model.moe.top_k,
                    moe_hidden_dim=cfg.model.moe.hidden_dim,
                    temperature=cfg.loss.temperature,
                    curvature=cfg.loss.curvature,
                    learning_rate=cfg.training.learning_rate,
                    weight_decay=cfg.training.weight_decay,
                    warmup_steps=cfg.training.warmup_steps,
                    load_balancing_coef=cfg.model.moe.load_balancing_coef,
                    distances_path=cfg.data_loader.streaming.distances_parquet,
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
                patience=3 if i < len(curricula) else 5,
                mode='min'
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
            trainer.fit(model, datamodule)
            
            # Store checkpoint path for next stage
            last_checkpoint = checkpoint_callback.best_model_path
            
            console.print(
                f'[green]✓[/green] Stage {i} complete. '
                f'Best checkpoint: [cyan]{last_checkpoint}[/cyan]\n'
            )
            
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