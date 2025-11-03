# -------------------------------------------------------------------------------------------------
# Imports
# -------------------------------------------------------------------------------------------------

import logging
from typing import List, Optional

import typer
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
from rich.console import Console
from rich.panel import Panel
from typing_extensions import Annotated

from naics_gemini.data_generation.compute_distances import calculate_pairwise_distances
from naics_gemini.data_generation.create_triplets import generate_training_triplets
from naics_gemini.data_generation.download_data import download_preprocess_data
from naics_gemini.utils.console import configure_logging

# --- Model & Data Imports (to be created later) ---
# from naics_gemini.data.datamodule import NAICSDataModule
# from naics_gemini.model.naics_model import NAICSModel


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
        GlobalHydra.instance().clear()  # Clear global state
        initialize(config_path='../conf', job_name='naics_gemini_train')

        # 2. Compose Config
        # Base config is "config.yaml"
        # We merge the specified curriculum (e.g., "curriculum/01_stage_easy.yaml")
        # And finally apply any command-line overrides
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

        # --- STUB: This is where we will load the real modules ---
        logger.info('--- STUB: Initializing components ---')
        logger.warning(
            "This is a STUB. The 'train' command is wired up, "
            "but the Model and DataModule classes are not yet implemented."
        )
        
        # --- (Future Implementation) ---
        # logger.info("Initializing DataModule...")
        # datamodule = NAICSDataModule(cfg.data, cfg.curriculum)
        
        # logger.info("Initializing Model...")
        # model = NAICSModel(cfg.model, cfg.loss, cfg.training.schedule)
        
        # logger.info("Initializing Trainer...")
        # # You can add callbacks here, e.g., from cfg.training.callbacks
        # trainer = pl.Trainer(**cfg.training.trainer)
        
        # logger.info("Starting model training...")
        # trainer.fit(model, datamodule)
        
        # logger.info("Training complete.")
        # logger.info(f"Best model checkpoint: {trainer.checkpoint_callback.best_model_path}")
        # --- End Future Implementation ---

        console.print(
            f'\n[bold green]Training command stub for curriculum '
            f"'[cyan]{curriculum}[/cyan]' executed successfully.[/bold]"
        )
        console.print('Next steps: Implement NAICSDataModule and NAICSModel.')

    except Exception as e:
        logger.error(f'An error occurred during training: {e}', exc_info=True)
        raise typer.Exit(code=1)


if __name__ == '__main__':
    app()
