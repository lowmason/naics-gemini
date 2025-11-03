# -------------------------------------------------------------------------------------------------
# Imports
# -------------------------------------------------------------------------------------------------

import logging
from pathlib import Path
from typing import List, Optional

import pytorch_lightning as pl
import typer
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from rich.console import Console
from rich.panel import Panel
from typing_extensions import Annotated

from naics_gemini.data_loader.datamodule import NAICSDataModule
from naics_gemini.data_generation.compute_distances import calculate_pairwise_distances
from naics_gemini.data_generation.create_triplets import generate_training_triplets
from naics_gemini.data_generation.download_data import download_preprocess_data
from naics_gemini.model.naics_model import NAICSContrastiveModel
from naics_gemini.utils.rich_setup import configure_logging

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

console = Console()
logger = logging.getLogger(__name__)


# -------------------------------------------------------------------------------------------------
# Data generation commands
# -------------------------------------------------------------------------------------------------

@data_app.command('preprocess')
def run_preprocess():

    '''
    Download and preprocess all raw NAICS data files.
    
    Generates: data/naics_descriptions.parquet
    '''
    
    configure_logging()

    console.rule('[bold green]Stage 1: Preprocessing[/bold green]')

    download_preprocess_data()

    console.print('\n[bold]Preprocessing complete.[/bold]\n')


@data_app.command('distances')
def run_distances():

    '''
    Compute pairwise graph distances between all NAICS codes.
    
    Requires: data/naics_descriptions.parquet
    Generates: data/naics_distances.parquet
    '''

    configure_logging()
    
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

    configure_logging()

    console.rule('[bold green]Stage 3: Generating Triplets[/bold green]')

    generate_training_triplets()

    console.print('\n[bold]Triplet generation complete.[/bold]\n')


@data_app.command('all')
def run_all_data_gen():

    '''
    Run the full data generation pipeline: preprocess, distances, and triplets.
    '''

    configure_logging()

    console.rule('[bold green]Starting Full Data Pipeline[/bold green]')
    
    run_preprocess()
    run_distances()
    run_triplets()
    
    console.rule('[bold green]Full Data Pipeline Complete![/bold green]')


# --- Model Training Command ---

@app.command('train')
def train(
    curriculum: Annotated[
        str,
        typer.Option(
            '--curriculum',
            '-c',
            help="Curriculum config name (e.g., '01_stage_easy').",
        ),
    ] = '01_stage_easy',
    overrides: Annotated[
        Optional[List[str]],
        typer.Argument(
            help="Hydra-style overrides, e.g., 'training.trainer.max_epochs=10'"
        ),
    ] = None,
):
    
    '''
    Train the NAICS-Gemini model using a specified curriculum.
    '''
    
    configure_logging()
    
    console.rule(
        f"[bold green]Starting Training: Curriculum '[cyan]{curriculum}[/cyan]'[/bold green]"
    )

    try:
        
        # 1. Initialize Hydra
        GlobalHydra.instance().clear()
        initialize(config_path='../conf', job_name='naics_gemini_train')

        # 2. Compose Config
        cfg_overrides = [f'curriculum={curriculum}'] + (overrides or [])
        cfg = compose(config_name='config', overrides=cfg_overrides)

        console.print(
            Panel(
                OmegaConf.to_yaml(cfg),
                title='[yellow]Computed Configuration[/yellow]',
                border_style='yellow',
                expand=True,
            )
        )
        console.rule()

        logger.info('Seeding random generators...')
        pl.seed_everything(cfg.seed)
        
        logger.info('Initializing DataModule...')
        datamodule = NAICSDataModule(
            descriptions_path=cfg.data.descriptions_path,
            triplets_path=cfg.data.triplets_path,
            tokenizer_name=cfg.data.tokenizer_name,
            curriculum_config={
                'positive_levels': cfg.curriculum.positive_levels,
                'positive_distance_min': cfg.curriculum.positive_distance_min,
                'positive_distance_max': cfg.curriculum.positive_distance_max,
                'max_positives': cfg.curriculum.max_positives,
                'difficulty_buckets': cfg.curriculum.difficulty_buckets,
                'bucket_percentages': cfg.curriculum.bucket_percentages,
                'k_negatives': cfg.curriculum.k_negatives
            },
            batch_size=cfg.data.batch_size,
            num_workers=cfg.data.num_workers,
            val_split=cfg.data.val_split,
            seed=cfg.seed
        )
        
        logger.info('Initializing Model...')
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
            load_balancing_coef=cfg.model.moe.load_balancing_coef
        )
        
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
        
        logger.info('Initializing Trainer...')
        trainer = pl.Trainer(
            **cfg.training.trainer,
            callbacks=[checkpoint_callback, early_stopping],
            logger=tb_logger,
            default_root_dir=cfg.paths.output_dir
        )
        
        logger.info('Starting model training...')
        trainer.fit(model, datamodule)
        
        logger.info('Training complete.')
        logger.info(f'Best model checkpoint: {checkpoint_callback.best_model_path}')
        
        console.print(
            f'\n[bold green]Training for curriculum '
            f"'[cyan]{curriculum}[/cyan]' completed successfully.[/bold]"
        )
        console.print(f'Best checkpoint: {checkpoint_callback.best_model_path}')

    except Exception as e:
        logger.error(f'An error occurred during training: {e}', exc_info=True)
        raise typer.Exit(code=1)


if __name__ == '__main__':
    app()
