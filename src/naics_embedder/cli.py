'''
Top-level Typer application that wires together project subcommands.

The module configures the global Typer instance and attaches command groups for
data preparation, tooling, and training. Warning filters are applied via the
centralized ``naics_embedder.utils.warnings`` module.
'''

import logging
import os

import typer
from rich.console import Console
from rich.panel import Panel

from naics_embedder.cli.commands import data, tools, training
from naics_embedder.utils.warnings import configure_warnings

os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'

# Apply centralized warning configuration
configure_warnings()

console = Console()
logger = logging.getLogger(__name__)


app = typer.Typer(
    help=Panel.fit(
        '[bold cyan]NAICS Embedder[/bold cyan]\n\nText-enhanced Hyperbolic NAICS Embedding System',
        border_style='cyan',
        padding=(1, 2),
    )  # type: ignore
)

app.add_typer(data.app, name='data')
app.add_typer(tools.app, name='tools')

app.command('train')(training.train)
app.command('train-seq')(training.train_sequential)