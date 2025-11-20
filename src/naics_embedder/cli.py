# -------------------------------------------------------------------------------------------------
# Main CLI Entry Point
# -------------------------------------------------------------------------------------------------

import logging
import os
import warnings

import typer
from rich.console import Console
from rich.panel import Panel

# Import command groups from separate modules
from naics_embedder.cli.commands import data, tools, training

# Set CUDA memory allocator configuration to reduce fragmentation
os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'

console = Console()
logger = logging.getLogger(__name__)


# -------------------------------------------------------------------------------------------------
# Suppress warnings
# -------------------------------------------------------------------------------------------------

warnings.filterwarnings(
    'ignore',
    message='.*Precision.*is not supported by the model summary.*',
    category=UserWarning,
    module='pytorch_lightning.utilities.model_summary.model_summary'
)

warnings.filterwarnings(
    'ignore',
    message='.*Found .* module.*in eval mode.*',
    category=UserWarning,
    module='pytorch_lightning'
)

warnings.filterwarnings(
    'ignore',
    message='.*does not have many workers.*',
    category=UserWarning,
    module='pytorch_lightning'
)

warnings.filterwarnings(
    'ignore',
    message='.*Checkpoint directory.*exists and is not empty.*',
    category=UserWarning,
    module='pytorch_lightning'
)

warnings.filterwarnings(
    'ignore',
    message='.*Trying to infer the.*batch_size.*',
    category=UserWarning,
    module='pytorch_lightning'
)


# -------------------------------------------------------------------------------------------------
# Setup Main Typer App
# -------------------------------------------------------------------------------------------------

app = typer.Typer(
    help=Panel.fit(
        '[bold cyan]NAICS Embedder[/bold cyan]\n\nText-enhanced Hyperbolic NAICS Embedding System',
        border_style='cyan',
        padding=(1, 2),
    )  # type: ignore
)

# Add command groups (sub-apps)
app.add_typer(data.app, name='data')
app.add_typer(tools.app, name='tools')

# Add training commands directly to main app
app.command('train')(training.train)
app.command('train-seq')(training.train_sequential)