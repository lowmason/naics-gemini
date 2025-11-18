# -------------------------------------------------------------------------------------------------
# Data Generation Commands
# -------------------------------------------------------------------------------------------------

import typer
from rich.console import Console

from naics_embedder.data_generation.compute_distances import calculate_pairwise_distances
from naics_embedder.data_generation.compute_relations import calculate_pairwise_relations
from naics_embedder.data_generation.create_triplets import generate_training_triplets
from naics_embedder.data_generation.download_data import download_preprocess_data
from naics_embedder.utils.console import configure_logging

console = Console()

# Create sub-app for data commands
app = typer.Typer(help='Manage and generate project datasets.')


@app.command('preprocess')
def preprocess():
    '''
    Download and preprocess all raw NAICS data files.
    
    Generates: data/naics_descriptions.parquet
    '''
    
    configure_logging('data_preprocess.log')

    console.rule('[bold green]Stage 1: Preprocessing[/bold green]')

    download_preprocess_data()

    console.print('\n[bold]Preprocessing complete.[/bold]\n')


@app.command('relations')
def relations():
    '''
    Compute pairwise graph relationships between all NAICS codes.
    
    Requires: data/naics_descriptions.parquet
    Generates: data/naics_relations.parquet
    '''

    configure_logging('data_relations.log')
    
    console.rule('[bold green]Stage 2: Computing Relations[/bold green]')

    calculate_pairwise_relations()

    console.print('\n[bold]Relation computation complete.[/bold]\n')


@app.command('distances')
def distances():
    '''
    Compute pairwise graph distances between all NAICS codes.
    
    Requires: data/naics_descriptions.parquet
    Generates: data/naics_distances.parquet
    '''

    configure_logging('data_distances.log')
    
    console.rule('[bold green]Stage 2: Computing Distances[/bold green]')

    calculate_pairwise_distances()

    console.print('\n[bold]Distance computation complete.[/bold]\n')


@app.command('triplets')
def triplets():
    '''
    Generate (anchor, positive, negative) training triplets.
    
    Requires: data/naics_descriptions.parquet, data/naics_distances.parquet
    Generates: data/naics_training_pairs.parquet
    '''

    configure_logging('data_triplets.log')

    console.rule('[bold green]Stage 3: Generating Triplets[/bold green]')

    generate_training_triplets()

    console.print('\n[bold]Triplet generation complete.[/bold]\n')


@app.command('all')
def all_data():
    '''
    Run the full data generation pipeline: preprocess, distances, and triplets.
    '''

    configure_logging('data_all.log')

    console.rule('[bold green]Starting Full Data Pipeline[/bold green]')
    
    preprocess()
    relations()
    distances()
    triplets()
    
    console.rule('[bold green]Full Data Pipeline Complete![/bold green]')
