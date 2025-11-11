# -------------------------------------------------------------------------------------------------
# Imports
# -------------------------------------------------------------------------------------------------

import logging
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler

# -------------------------------------------------------------------------------------------------
# Configure logging
# -------------------------------------------------------------------------------------------------

class ConsoleFormatter(logging.Formatter):

    def __init__(self, timefmt='[%H:%M:%S]'):
        super().__init__()
        self.timefmt = timefmt
        self._last_time_str = None

    def format(self, record: logging.LogRecord) -> str:
        time_str = datetime.fromtimestamp(record.created).strftime(self.timefmt)
        message = record.getMessage()

        if time_str != self._last_time_str:
            self._last_time_str = time_str
            return f'{time_str}\n{message}'
        
        else:
            return message


def configure_logging(
    log_file: str,
    log_dir: str = './logs',
    level: str = 'INFO'
):

    # Create log directory if it doesn't exist
    if not Path(log_dir).exists():
        Path(log_dir).mkdir(parents=True, exist_ok=True)

    # Rich console handler
    console = Console(markup=False)
    console_handler = RichHandler(
        console=console,
        rich_tracebacks=True,
        tracebacks_suppress=[
            'typer',
            'click',
            'hydra',
            'pytorch_lightning',
            'torch'
        ],
        show_path=False,
        show_time=False,
        show_level=False,
        markup=True
    )
    console_handler.setFormatter(ConsoleFormatter())
    
    # Rich file handler
    file_handler = logging.FileHandler(f'{log_dir}/{log_file}', encoding='utf-8')
    file_formatter = logging.Formatter(
        fmt='%(asctime)s | %(levelname)s | %(name)s | %(filename)s:%(lineno)d] | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)

    # BasicConfig with both handlers
    logging.basicConfig(
        level=level,
        handlers=[console_handler, file_handler]
    )

    # Quiet down noisy libs
    for noisy in ['hydra', 'pytorch_lightning', 'transformers', 'httpx']:
        logging.getLogger(noisy).setLevel(logging.WARNING)

    # Explicitly re-enable your own package
    logging.getLogger('naics_gemini').setLevel(level)
