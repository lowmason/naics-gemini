# -------------------------------------------------------------------------------------------------
# Data Generation Commands
# -------------------------------------------------------------------------------------------------
'''
CLI commands for NAICS data generation and preprocessing.

This module provides the ``data`` command group that orchestrates the data
preparation pipeline. Commands should be run in order or via ``data all``.

Pipeline Stages:
    1. preprocess: Download and clean raw NAICS data files
    2. relations: Compute pairwise graph relationships
    3. distances: Compute pairwise graph distances
    4. triplets: Generate training triplets for contrastive learning

Commands:
    preprocess: Download raw NAICS files and produce descriptions parquet.
    relations: Build relationship annotations between all NAICS codes.
    distances: Compute tree distances between all NAICS codes.
    triplets: Generate (anchor, positive, negative) training triplets.
    all: Run the complete data generation pipeline.
'''

import typer
from rich.console import Console

from naics_embedder.data.compute_distances import calculate_pairwise_distances
from naics_embedder.data.compute_relations import calculate_pairwise_relations
from naics_embedder.data.create_triplets import generate_training_triplets
from naics_embedder.data.download_data import download_preprocess_data
from naics_embedder.utils.console import configure_logging

# -------------------------------------------------------------------------------------------------
# Data generation sub-commands
# -------------------------------------------------------------------------------------------------

console = Console()

app = typer.Typer(
    help='Data generation and preprocessing commands for NAICS taxonomy.', no_args_is_help=True
)

# -------------------------------------------------------------------------------------------------
# Download and preprocess data
# -------------------------------------------------------------------------------------------------

@app.command('preprocess')
def preprocess():
    '''
    Download and preprocess all raw NAICS data files.

    Downloads the official 2022 NAICS taxonomy files from the U.S. Census
    Bureau and processes them into a unified descriptions parquet file.

    The output file contains columns for code, title, description, examples,
    and exclusions for each NAICS code at all hierarchy levels (2-6 digit).

    Output:
        ``data/naics_descriptions.parquet`` - Unified NAICS taxonomy data.

    Example:
        Download and preprocess NAICS data::

            $ uv run naics-embedder data preprocess
    '''

    configure_logging('data_preprocess.log')

    console.rule('[bold green]Stage 1: Preprocessing[/bold green]')

    download_preprocess_data()

    console.print('\n[bold]Preprocessing complete.[/bold]\n')

# -------------------------------------------------------------------------------------------------
# Compute pairwise graph relationships
# -------------------------------------------------------------------------------------------------

@app.command('relations')
def relations():
    '''
    Compute pairwise graph relationships between all NAICS codes.

    Analyzes the NAICS hierarchy to determine relationship types between
    every pair of codes (child, sibling, cousin, etc.). These relationships
    are used for curriculum-based sampling during training.

    Requires:
        ``data/naics_descriptions.parquet`` - From the preprocess stage.

    Output:
        ``data/naics_relations.parquet`` - Pairwise relationship annotations.
        ``data/naics_relation_matrix.parquet`` - Sparse matrix representation.

    Example:
        Compute all pairwise relationships::

            $ uv run naics-embedder data relations
    '''

    configure_logging('data_relations.log')

    console.rule('[bold green]Stage 2: Computing Relations[/bold green]')

    calculate_pairwise_relations()

    console.print('\n[bold]Relation computation complete.[/bold]\n')

# -------------------------------------------------------------------------------------------------
# Compute pairwise graph distances
# -------------------------------------------------------------------------------------------------

@app.command('distances')
def distances():
    '''
    Compute pairwise graph distances between all NAICS codes.

    Calculates tree distances in the NAICS hierarchy for every pair of codes.
    Distance is computed as the sum of edges traversed to reach the lowest
    common ancestor. Used for hierarchy preservation loss and evaluation.

    Requires:
        ``data/naics_descriptions.parquet`` - From the preprocess stage.

    Output:
        ``data/naics_distances.parquet`` - Pairwise distance annotations.
        ``data/naics_distance_matrix.parquet`` - Sparse matrix representation.

    Example:
        Compute all pairwise distances::

            $ uv run naics-embedder data distances
    '''

    configure_logging('data_distances.log')

    console.rule('[bold green]Stage 2: Computing Distances[/bold green]')

    calculate_pairwise_distances()

    console.print('\n[bold]Distance computation complete.[/bold]\n')

# -------------------------------------------------------------------------------------------------
# Generate training triplets
# -------------------------------------------------------------------------------------------------

@app.command('triplets')
def triplets():
    '''
    Generate (anchor, positive, negative) training triplets.

    Creates training triplets for contrastive learning by sampling anchors
    from the NAICS taxonomy and pairing them with positive samples (related
    codes) and negative samples (distant codes).

    Triplet generation uses the distance and relation annotations to ensure
    meaningful contrastive pairs that respect the hierarchical structure.

    Requires:
        ``data/naics_descriptions.parquet`` - From the preprocess stage.
        ``data/naics_distances.parquet`` - From the distances stage.
        ``data/naics_relations.parquet`` - From the relations stage.

    Output:
        ``data/naics_training_pairs/`` - Directory of parquet files with triplets.

    Example:
        Generate training triplets::

            $ uv run naics-embedder data triplets
    '''

    configure_logging('data_triplets.log')

    console.rule('[bold green]Stage 3: Generating Triplets[/bold green]')

    generate_training_triplets()

    console.print('\n[bold]Triplet generation complete.[/bold]\n')

# -------------------------------------------------------------------------------------------------
# Run full data generation pipeline
# -------------------------------------------------------------------------------------------------

@app.command('all')
def all_data():
    '''
    Run the complete data generation pipeline.

    Executes all data preparation stages in order: preprocess, relations,
    distances, and triplets. This is the recommended way to prepare data
    for training from scratch.

    Output:
        All data files required for training will be created in ``data/``.

    Example:
        Run the full pipeline::

            $ uv run naics-embedder data all

    Note:
        This command may take 10-30 minutes depending on your system.
        Progress is logged to ``logs/data_all.log``.
    '''

    configure_logging('data_all.log')

    console.rule('[bold green]Starting Full Data Pipeline[/bold green]')

    preprocess()
    relations()
    distances()
    triplets()

    console.rule('[bold green]Full Data Pipeline Complete![/bold green]')
