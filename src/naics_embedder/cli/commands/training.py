# -------------------------------------------------------------------------------------------------
# Training Commands
# -------------------------------------------------------------------------------------------------

import logging
import time
from pathlib import Path
from typing import List, Optional

import pytorch_lightning as pyl
import typer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from rich.console import Console
from typing_extensions import Annotated

from naics_embedder.data_loader.datamodule import NAICSDataModule
from naics_embedder.model.naics_model import NAICSContrastiveModel
from naics_embedder.utils.backend import get_device
from naics_embedder.utils.config import ChainConfig, Config, list_available_curricula, parse_override_value
from naics_embedder.utils.console import configure_logging

console = Console()
logger = logging.getLogger(__name__)


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
    """
    Train a model with a specific curriculum configuration.
    
    This command trains a model using a single curriculum stage. For multi-stage
    sequential training with automatic checkpoint handoff, use 'train-seq' instead.
    """
    
    # Your full train implementation here
    # (Lines 200-650 from your original cli.py)
    
    configure_logging('train.log')
    
    console.rule('[bold cyan]Starting Single Curriculum Training[/bold cyan]')
    
    # Handle --list-curricula flag
    if list_curricula:
        available = list_available_curricula(curriculum_type)
        console.print('\n[bold]Available curricula:[/bold]\n')
        for curr in available:
            console.print(f'  • {curr}')
        console.print()
        raise typer.Exit()
    
    # Load config with curriculum
    try:
        cfg = Config.from_yaml(
            config_file,
            curriculum=curriculum,
            curriculum_type=curriculum_type
        )
        
        # Apply overrides if provided
        if overrides:
            override_dict = {}
            for override in overrides:
                key, value = override.split('=', 1)
                override_dict[key] = parse_override_value(value)
            cfg = cfg.merge_overrides(override_dict)
            
    except Exception as e:
        console.print(f'[bold red]Error loading configuration:[/bold red] {e}')
        raise typer.Exit(code=1)
    
    # Display configuration
    console.print(f'\n[bold]Curriculum:[/bold] {curriculum}')
    console.print(f'[bold]Config:[/bold] {config_file}\n')
    
    # Create output directories
    Path(cfg.dirs.output_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.dirs.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize data module
    console.print('[cyan]Initializing data module...[/cyan]')
    datamodule = NAICSDataModule(cfg)
    
    # Initialize model
    console.print('[cyan]Initializing model...[/cyan]')
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
    
    # Setup logger
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
    
    # Train
    console.print(f'\n[bold green]Starting training...[/bold green]\n')
    trainer.fit(model, datamodule, ckpt_path=ckpt_path)
    
    console.print(f'\n[bold green]✓ Training complete![/bold green]')
    console.print(f'Best checkpoint: [cyan]{checkpoint_callback.best_model_path}[/cyan]\n')


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
                curriculum=curriculum,
                curriculum_type=curriculum_type
            )
            
            # Apply overrides
            if overrides:
                override_dict = {}
                for override in overrides:
                    key, value = override.split('=', 1)
                    override_dict[key] = parse_override_value(value)
                cfg = cfg.merge_overrides(override_dict)
            
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
            
            console.print(
                f'\n[green]✓[/green] Stage {i} complete. '
                f'Best checkpoint: [cyan]{last_checkpoint}[/cyan]\n'
            )
            
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
