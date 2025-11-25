# -------------------------------------------------------------------------------------------------
# Imports
# -------------------------------------------------------------------------------------------------

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

app = typer.Typer(help='Utility tools for configuration and metrics analysis.')


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
    Display current training and curriculum configuration.
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
    
    Creates comprehensive visualizations and analysis of training metrics including:
    - Hyperbolic radius over time
    - Hierarchy preservation correlations
    - Embedding diversity metrics
    '''
    
    configure_logging('tools_visualize.log')
        
    try:
        log_path = Path(log_file) if log_file else None
        output_path = Path(output_dir) if output_dir else None
        
        result = visualize_metrics(
            stage=stage,
            log_file=log_path,
            output_dir=output_path
        )
        
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
    Investigate why hierarchy preservation correlations might be low.
    
    Analyzes ground truth distances, evaluation configuration, and provides
    recommendations for improving hierarchy preservation metrics.
    '''
    
    configure_logging('tools_investigate.log')

    try:
        dist_path = Path(distance_matrix) if distance_matrix else None
        config_path = Path(config_file) if config_file else None
        
        result = investigate_hierarchy(
            distance_matrix_path=dist_path,
            config_path=config_path
        )
        for key, value in result.items():
            console.print(f'[bold green]{key}:[/bold green] {value}')
            
        console.print('\n[bold green]Investigation complete![/bold green]\n')
        
    except Exception as e:
        console.print(f'[bold red]Error:[/bold red] {e}')
        raise typer.Exit(code=1)
