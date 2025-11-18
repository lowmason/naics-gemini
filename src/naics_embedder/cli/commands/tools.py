# -------------------------------------------------------------------------------------------------
# Tools Commands
# -------------------------------------------------------------------------------------------------

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from typing_extensions import Annotated

from naics_embedder.tools.config_tools import show_current_config
from naics_embedder.tools.gpu_tools import detect_gpu_memory, optimize_gpu_config
from naics_embedder.tools.metrics_tools import investigate_hierarchy, visualize_metrics
from naics_embedder.utils.console import configure_logging

console = Console()

# Create sub-app for tools commands
app = typer.Typer(help='Utility tools for configuration, GPU optimization, and metrics analysis.')


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
    """
    Display current training and curriculum configuration.
    """
    show_current_config(config_file)


@app.command('gpu')
def gpu(
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


@app.command('investigate')
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
