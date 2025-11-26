# -------------------------------------------------------------------------------------------------
# Imports
# -------------------------------------------------------------------------------------------------

import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import polars as pl
from polars import selectors as cs
from rich.console import Console
from rich.logging import RichHandler

# -------------------------------------------------------------------------------------------------
# Configure logging
# -------------------------------------------------------------------------------------------------

class ConsoleFormatter(logging.Formatter):
    '''Formatter that prints time only if more than `time_interval` seconds have elapsed.'''

    def __init__(self, timefmt='[%H:%M:%S]', time_interval: float = 600.0):
        super().__init__()
        self.timefmt = timefmt
        self.time_interval = time_interval
        self._last_time = None
        self._last_time_str = None

    def format(self, record: logging.LogRecord) -> str:
        current_time = datetime.fromtimestamp(record.created)
        time_str = current_time.strftime(self.timefmt)
        message = record.getMessage()

        # Determine whether to print timestamp
        if (
            self._last_time is None
            or (current_time - self._last_time).total_seconds() >= self.time_interval
        ):
            self._last_time = current_time
            self._last_time_str = time_str
            return f'{time_str}\n{message}'
        else:
            return message

def configure_logging(log_file: str, log_dir: str = './logs', level: str = 'INFO'):
    # Create log directory if it doesn't exist
    if not Path(log_dir).exists():
        Path(log_dir).mkdir(parents=True, exist_ok=True)

    # Rich console handler
    console = Console(markup=False)
    console_handler = RichHandler(
        console=console,
        rich_tracebacks=True,
        tracebacks_suppress=['typer', 'click', 'hydra', 'pytorch_lightning', 'torch'],
        show_path=False,
        show_time=False,
        show_level=False,
        markup=True,
    )
    console_handler.setFormatter(ConsoleFormatter())

    # Rich file handler
    file_handler = logging.FileHandler(f'{log_dir}/{log_file}', encoding='utf-8')
    file_formatter = logging.Formatter(
        fmt='%(asctime)s | %(levelname)s | %(name)s | %(filename)s:%(lineno)d] | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    file_handler.setFormatter(file_formatter)

    # BasicConfig with both handlers
    logging.basicConfig(level=level, handlers=[console_handler, file_handler])

    # Quiet down noisy libs
    for noisy in ['hydra', 'pytorch_lightning', 'transformers', 'httpx']:
        logging.getLogger(noisy).setLevel(logging.WARNING)

    # Explicitly re-enable your own package
    logging.getLogger('naics_embedder').setLevel(level)

# -------------------------------------------------------------------------------------------------
# Print styled table
# -------------------------------------------------------------------------------------------------

def log_table(
    df: pl.DataFrame,
    title: str,
    headers: Optional[List[str]] = None,
    logger: Optional[logging.Logger] = None,
    output: Optional[str] = None,
) -> None:
    if headers:
        cols, span_1, span_2 = [], [], []
        for h, c in zip(headers, df.columns):
            if ':' in h:
                s_part, h_part = h.split(':')

                cols.append((h_part, c))
                span_1.append(s_part)
                span_2.append(c)
            else:
                cols.append((h, c))

        cols_labels_dict = {c: h for h, c in cols}

        if len(span_2) >= 2:
            span_label = list(set(span_1))[0]
            span_cols = span_2
        else:
            span_label = None
            span_cols = None

    else:
        cols_labels_dict, span_label, span_cols = {c: c for c in df.columns}, None, None

    if span_label is not None and span_cols is not None:
        table = (
            df.style.tab_header(title=title).cols_label(cols_labels_dict)  # type: ignore
            .tab_spanner(span_label,
                         cs.by_name(span_cols)).fmt_integer('cnt').fmt_number('pct', decimals=4)
        )

    else:
        table = (
            df.style.tab_header(title=title).cols_label(cols_labels_dict)  # type: ignore
            .fmt_integer('cnt').fmt_number('pct', decimals=4)
        )

    with pl.Config() as cfg:
        cfg.set_tbl_rows(df.height)
        cfg.set_thousands_separator(',')
        cfg.set_float_precision(4)

        if logger:
            logger.info(f'\n{str(df)}')

        if output:
            table.save(output)
