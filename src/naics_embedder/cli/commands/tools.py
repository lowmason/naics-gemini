# -------------------------------------------------------------------------------------------------
# Tools Commands
# -------------------------------------------------------------------------------------------------
'''
CLI utility commands for configuration, GPU optimization, and metrics analysis.

This module provides the ``tools`` command group with utilities for inspecting
configuration, visualizing training metrics, and investigating model behavior.

Commands:
    config: Display current training configuration.
    visualize: Generate visualizations from training log files.
    investigate: Analyze hierarchy preservation metrics.
'''

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from typing_extensions import Annotated

from naics_embedder.tools.config_tools import show_current_config
from naics_embedder.tools.metrics_tools import investigate_hierarchy, visualize_metrics
from naics_embedder.utils.console import configure_logging

# -------------------------------------------------------------------------------------------------
# Tools Commands
# -------------------------------------------------------------------------------------------------

console = Console()

app = typer.Typer(
    help='Utility tools for configuration, metrics analysis, and debugging.', no_args_is_help=True
)

# -------------------------------------------------------------------------------------------------
# View configuration
# -------------------------------------------------------------------------------------------------

@app.command('config')
def config(
    config_file: Annotated[
        str,
        typer.Option(
            '--config',
            help='Path to base config YAML file',
        ),
    ] = 'conf/config.yaml',
):
    '''
    Display the current training and curriculum configuration.

    Loads the specified configuration file and displays a formatted summary
    of all settings including data paths, model architecture, training
    hyperparameters, and loss function weights.

    Args:
        config_file: Path to the YAML configuration file to display.
            Defaults to ``conf/config.yaml``.

    Example:
        Display default configuration::

            $ uv run naics-embedder tools config

        Display custom configuration::

            $ uv run naics-embedder tools config --config conf/custom.yaml
    '''

    configure_logging('tools_config.log')

    show_current_config(config_file)

# -------------------------------------------------------------------------------------------------
# Visualize metrics
# -------------------------------------------------------------------------------------------------

@app.command('visualize')
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
    '''
    Visualize training metrics from log files.

    Parses training log files and generates visualizations showing the
    progression of key metrics including contrastive loss, hierarchy
    correlation, embedding statistics, and learning rate schedules.

    Output visualizations are saved as PNG files in the specified output
    directory.

    Args:
        stage: Stage identifier used to filter metrics. Use this to focus
            on a specific training stage like ``02_text``.
        log_file: Path to the training log file to parse. When omitted,
            defaults to ``logs/train_sequential.log``.
        output_dir: Directory for saving visualization files. When omitted,
            defaults to ``outputs/visualizations/``.

    Example:
        Visualize metrics from default log::

            $ uv run naics-embedder tools visualize --stage 02_text

        Visualize custom log file::

            $ uv run naics-embedder tools visualize --log-file logs/train.log
    '''

    configure_logging('tools_visualize.log')

    try:
        log_path = Path(log_file) if log_file else None
        output_path = Path(output_dir) if output_dir else None

        result = visualize_metrics(stage=stage, log_file=log_path, output_dir=output_path)

        if result.get('output_file'):
            console.print(
                '\n[bold green]✓[/bold green] Visualization saved to: '
                f'[cyan]{result["output_file"]}[/cyan]\n'
            )

    except Exception as e:
        console.print(f'[bold red]Error:[/bold red] {e}')
        raise typer.Exit(code=1)

# -------------------------------------------------------------------------------------------------
# Investigate hierarchy preservation metrics
# -------------------------------------------------------------------------------------------------

@app.command('investigate')
def investigate(
    distance_matrix: Annotated[
        Optional[str],
        typer.Option(
            '--distance-matrix',
            help='Path to ground truth distance matrix',
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
    '''
    Analyze why hierarchy preservation correlations might be low.
    
    Investigates potential causes for poor hierarchy preservation metrics
    by analyzing the ground truth distance matrix, evaluation configuration,
    and providing diagnostic recommendations.
    
    Use this command when training produces unexpectedly low hierarchy
    correlation metrics to identify configuration or data issues.
    
    Args:
        distance_matrix: Path to the ground truth distance matrix parquet.
            When omitted, uses the path from the configuration file.
        config_file: Path to the configuration file. When omitted, uses
            the default ``conf/config.yaml``.
    
    Example:
        Investigate hierarchy metrics::
        
            $ uv run naics-embedder tools investigate
        
        Use custom distance matrix::
        
            $ uv run naics-embedder tools investigate \\
                --distance-matrix data/custom_distances.parquet
    '''

    configure_logging('tools_investigate.log')

    try:
        dist_path = Path(distance_matrix) if distance_matrix else None
        config_path = Path(config_file) if config_file else None

        result = investigate_hierarchy(distance_matrix_path=dist_path, config_path=config_path)
        for key, value in result.items():
            console.print(f'[bold green]{key}:[/bold green] {value}')

        console.print('\n[bold green]Investigation complete![/bold green]\n')

    except Exception as e:
        console.print(f'[bold red]Error:[/bold red] {e}')
        raise typer.Exit(code=1)
