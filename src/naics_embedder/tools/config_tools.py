'''
Configuration display tools.

Display current training and curriculum configuration.
'''

import yaml
from pathlib import Path
from typing import Optional

from rich.console import Console, Group
from rich.panel import Panel


console = Console()


def show_current_config(config_path: str = './conf/config.yaml'):
    '''
    Display current training and curriculum configuration.
    
    Args:
        config_path: Path to main configuration file
    '''
    
    config_path_obj = Path(config_path)
    if not config_path_obj.exists():
        console.print(f'[bold red]Error:[/bold red] Config file not found: {config_path}')
        return
    
    # Load configurations
    config = load_config(config_path)
    
    batch_size = config['data_loader']['batch_size']
    accumulate = config['training']['trainer']['accumulate_grad_batches']
    num_workers = config['data_loader']['num_workers']
    learning_rate = config['training']['learning_rate']
    weight_decay = config['training']['weight_decay']
    warmup_steps = config['training']['warmup_steps']
    precision = config['training']['trainer']['precision']
    max_epochs = config['training']['trainer']['max_epochs']
    
    current_config = [
        '\n[blue]Main Configuration (conf/config.yaml):[/blue]\n',
        f'[cyan]Effective batch size:[/cyan] {batch_size * accumulate}',
        f'  • [bold]batch_size:[/bold] {batch_size}',
        f'  • [bold]accumulate_grad_batches:[/bold] {accumulate}\n',
        '[cyan]Data loader:[/cyan]',
        f'  • [bold]num_workers:[/bold] {num_workers}\n',
        '[cyan]Training:[/cyan]',
        f'  • [bold]learning_rate:[/bold] {learning_rate}',
        f'  • [bold]weight_decay:[/bold] {weight_decay}',
        f'  • [bold]warmup_steps:[/bold] {warmup_steps}',
        f'  • [bold]precision:[/bold] {precision}',
        f'  • [bold]max_epochs:[/bold] {max_epochs}\n',
    ]    
    
    console.print(
        Panel(
            '\n'.join(current_config),
            title='[yellow]Current Training Configuration[/yellow]',
            border_style='yellow',
            expand=True
        )
    )


def load_config(config_path: str = './conf/config.yaml'):
    '''Load main configuration file.'''
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

