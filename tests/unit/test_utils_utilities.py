from pathlib import Path

import httpx
import pytest

from naics_embedder.utils.config import DirConfig
from naics_embedder.utils.utilities import (
    download_with_retry,
    make_directories,
    map_relationships,
    setup_directory,
)

@pytest.mark.unit
def test_make_directories_creates_all(tmp_path):
    cfg = DirConfig(
        checkpoint_dir=str(tmp_path / 'ckpts'),
        conf_dir=str(tmp_path / 'conf'),
        data_dir=str(tmp_path / 'data'),
        docs_dir=str(tmp_path / 'docs'),
        log_dir=str(tmp_path / 'logs'),
        output_dir=str(tmp_path / 'outputs'),
    )

    make_directories(cfg)

    for path in cfg.model_dump().values():
        assert Path(path).exists()

@pytest.mark.unit
def test_map_relationships_returns_both_mappings():
    forward = map_relationships('child')
    reverse = map_relationships(1)

    assert forward['child'] == 1
    assert reverse[1] == 'child'

    with pytest.raises(ValueError):
        map_relationships(1.5)

@pytest.mark.unit
def test_download_with_retry_succeeds_after_retry(monkeypatch):
    attempts = {'count': 0}

    class DummyResponse:

        def __init__(self, content):
            self.content = content

        def raise_for_status(self):
            return None

    def fake_get(url, timeout):
        attempts['count'] += 1
        if attempts['count'] == 1:
            raise httpx.HTTPError('boom')
        return DummyResponse(b'data')

    monkeypatch.setattr('naics_embedder.utils.utilities.httpx.get', fake_get)
    monkeypatch.setattr('naics_embedder.utils.utilities.time.sleep', lambda *_: None)

    data = download_with_retry('https://example.com', max_retries=1)

    assert data == b'data'
    assert attempts['count'] == 2

@pytest.mark.unit
def test_download_with_retry_raises_after_exhaustion(monkeypatch):

    def boom(*_args, **_kwargs):
        raise httpx.HTTPError('boom')

    monkeypatch.setattr('naics_embedder.utils.utilities.httpx.get', boom)
    monkeypatch.setattr('naics_embedder.utils.utilities.time.sleep', lambda *_: None)

    with pytest.raises(httpx.HTTPError):
        download_with_retry('https://example.com', max_retries=0)

@pytest.mark.unit
def test_setup_directory_creates_path(tmp_path):
    target = tmp_path / 'new_dir'

    path = setup_directory(str(target))

    assert path.exists()
