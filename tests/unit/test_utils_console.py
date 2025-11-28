import logging

import polars as pl
import pytest

from naics_embedder.utils.console import configure_logging, log_table

@pytest.mark.unit
def test_configure_logging_creates_log_file(tmp_path, monkeypatch):
    # Reset root handlers so configure_logging can attach new ones
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    log_dir = tmp_path / 'logs'
    configure_logging('test.log', log_dir=str(log_dir))

    assert (log_dir / 'test.log').exists()

@pytest.mark.unit
def test_log_table_writes_output(tmp_path):
    df = pl.DataFrame({'relation_id': [1], 'relation': ['child'], 'cnt': [5], 'pct': [50.0]})
    output = tmp_path / 'table.png'

    log_table(
        df=df,
        title='Relation Stats',
        headers=['Relation ID:relation_id', 'Relation:relation', 'cnt', 'pct'],
        output=str(output),
    )

    assert output.exists()
