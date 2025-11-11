# -------------------------------------------------------------------------------------------------
# Imports
# -------------------------------------------------------------------------------------------------

import logging
from pathlib import Path
from typing import List, Optional

import pytorch_lightning as pyl
import typer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
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
from naics_gemini.utils.config import Config, list_available_curricula, parse_override_value
from naics_gemini.utils.console import configure_logging

console = Console()
logger = logging.getLogger(__name__)


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
# Model training commands
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

        # Load configuration
        logger.info('Loading configuration...')
        cfg = Config.from_yaml(config_file, curriculum_name=curriculum)
        
        # Apply command-line overrides
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
                logger.info(f'Override: {key} = {value} ({type(value).__name__})')
            
            cfg = cfg.override(override_dict)
        
        # Display configuration summary
        console.print(
            Panel(
                f'[bold]Experiment:[/bold] {cfg.experiment_name}\n'
                f'[bold]Curriculum:[/bold] {cfg.curriculum.name}\n'
                f'[bold]Seed:[/bold] {cfg.seed}\n\n'
                f'[cyan]Data:[/cyan]\n'
                f'  • Batch size: {cfg.data.batch_size}\n'
                f'  • Num workers: {cfg.data.num_workers}\n\n'
                f'[cyan]Model:[/cyan]\n'
                f'  • Base: {cfg.model.base_model_name.split("/")[-1]}\n'
                f'  • LoRA rank: {cfg.model.lora.r}\n'
                f'  • MoE: {"enabled" if cfg.model.moe.enabled else "disabled"} '
                f'({cfg.model.moe.num_experts} experts)\n\n'
                f'[cyan]Training:[/cyan]\n'
                f'  • Learning rate: {cfg.training.learning_rate}\n'
                f'  • Max epochs: {cfg.training.trainer.max_epochs}\n'
                f'  • Accelerator: {cfg.training.trainer.accelerator}\n'
                f'  • Precision: {cfg.training.trainer.precision}',
                title='[yellow]Configuration Summary[/yellow]',
                border_style='yellow',
                expand=False
            )
        )
        
        # Seed for reproducibility
        logger.info(f'Setting random seed: {cfg.seed}')
        pyl.seed_everything(cfg.seed)
        
        # Initialize DataModule
        logger.info('Initializing DataModule...')
        datamodule = NAICSDataModule(
            descriptions_path=cfg.data.descriptions_path,
            triplets_path=cfg.data.triplets_path,
            tokenizer_name=cfg.data.tokenizer_name,
            curriculum_config={
                'anchor_level': cfg.curriculum.anchor_level,
                'excluded': cfg.curriculum.excluded,
                'unrelated': cfg.curriculum.unrelated,
                'relation_margins': cfg.curriculum.relation_margins,
                'distance_margins': cfg.curriculum.distance_margins,
                'positive_level': cfg.curriculum.positive_level,
                'positive_relation': cfg.curriculum.positive_relation,
                'positive_distance': cfg.curriculum.positive_distance,
                'n_positives': cfg.curriculum.n_positives,
                'negative_level': cfg.curriculum.negative_level,
                'negative_relation': cfg.curriculum.negative_relation,
                'negative_distance': cfg.curriculum.negative_distance,
                'n_negatives': cfg.curriculum.n_negatives
            },
            batch_size=cfg.data.batch_size,
            num_workers=cfg.data.num_workers,
            val_split=cfg.data.val_split,
            seed=cfg.seed
        )
        
        # Construct distances path
        distances_path = str(Path(cfg.paths.data_dir) / 'naics_distances.parquet')
        
        # Initialize Model
        logger.info('Initializing Model with evaluation metrics...')
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
        logger.info('Setting up callbacks and checkpointing...')
        checkpoint_dir = Path(cfg.paths.checkpoint_dir) / cfg.experiment_name
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
        
        tb_logger = TensorBoardLogger(
            save_dir=cfg.paths.output_dir,
            name=cfg.experiment_name
        )
        
        # Initialize Trainer
        logger.info('Initializing PyTorch Lightning Trainer...')
        trainer = pyl.Trainer(
            max_epochs=cfg.training.trainer.max_epochs,
            accelerator=cfg.training.trainer.accelerator,
            devices=cfg.training.trainer.devices,
            precision=cfg.training.trainer.precision,
            gradient_clip_val=cfg.training.trainer.gradient_clip_val,
            accumulate_grad_batches=cfg.training.trainer.accumulate_grad_batches,
            log_every_n_steps=cfg.training.trainer.log_every_n_steps,
            val_check_interval=cfg.training.trainer.val_check_interval,
            callbacks=[checkpoint_callback, early_stopping],
            logger=tb_logger,
            default_root_dir=cfg.paths.output_dir
        )
        
        # Start training
        logger.info('Starting model training with evaluation metrics...')
        console.print('\n[bold cyan]📊 Evaluation metrics enabled:[/bold cyan]')
        console.print('  • Cophenetic correlation (hierarchy preservation)')
        console.print('  • Spearman correlation (rank preservation)')
        console.print('  • Embedding statistics (norms, distances)')
        console.print('  • Collapse detection (variance, norm, distance)')
        console.print('  • Distortion metrics (mean, std)\n')
        
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