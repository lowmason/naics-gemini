import polars as pl
import pytest
import torch

from naics_embedder.utils.config import Config, TokenizationConfig
from naics_embedder.utils.validation import (
    ValidationError,
    require_valid_config,
    validate_data_paths,
    validate_descriptions_schema,
    validate_distances_schema,
    validate_tokenization_cache,
    validate_training_config,
)

def _touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('test')
    return str(path)

@pytest.mark.unit
def test_validate_data_paths_all_present(tmp_path):
    cfg = Config()
    streaming = cfg.data_loader.streaming
    streaming.descriptions_parquet = _touch(tmp_path / 'descriptions.parquet')
    streaming.distances_parquet = _touch(tmp_path / 'distances.parquet')
    streaming.distance_matrix_parquet = _touch(tmp_path / 'distance_matrix.parquet')
    streaming.relations_parquet = _touch(tmp_path / 'relations.parquet')
    triplets_dir = tmp_path / 'triplets'
    triplets_dir.mkdir()
    (triplets_dir / 'batch.parquet').write_text('rows')
    streaming.triplets_parquet = str(triplets_dir)

    result = validate_data_paths(cfg)
    assert result.valid

@pytest.mark.unit
def test_validate_data_paths_missing_file(tmp_path):
    cfg = Config()
    streaming = cfg.data_loader.streaming
    streaming.descriptions_parquet = str(tmp_path / 'missing.parquet')

    result = validate_data_paths(cfg)
    assert result.valid is False
    assert any('Descriptions file not found' in err for err in result.errors)

@pytest.mark.unit
def test_validate_descriptions_schema_success(tmp_path):
    path = tmp_path / 'descriptions.parquet'
    pl.DataFrame(
        {
            'index': [0],
            'code': ['11'],
            'level': [2],
            'title': ['Manufacturing'],
            'description': ['Test'],
        }
    ).write_parquet(path)

    cfg = Config()
    cfg.data_loader.streaming.descriptions_parquet = str(path)
    result = validate_descriptions_schema(cfg)
    assert result.valid

@pytest.mark.unit
def test_validate_distances_schema_missing_column(tmp_path):
    path = tmp_path / 'distances.parquet'
    pl.DataFrame({'idx_i': [0], 'idx_j': [1]}).write_parquet(path)
    cfg = Config()
    cfg.data_loader.streaming.distances_parquet = str(path)

    result = validate_distances_schema(cfg)
    assert result.valid is False
    assert any('missing columns' in err for err in result.errors)

@pytest.mark.unit
def test_validate_tokenization_cache_missing_returns_warning(tmp_path):
    cfg = Config()
    token_cfg = TokenizationConfig(
        descriptions_parquet=cfg.data_loader.streaming.descriptions_parquet,
        tokenizer_name=cfg.data_loader.tokenization.tokenizer_name,
        max_length=cfg.data_loader.tokenization.max_length,
        output_path=str(tmp_path / 'cache.pt'),
    )

    result = validate_tokenization_cache(cfg, tokenization_cfg=token_cfg)
    assert result.valid
    assert result.warnings

@pytest.mark.unit
def test_validate_tokenization_cache_structure_error(tmp_path):
    cache_path = tmp_path / 'cache.pt'
    torch.save({0: {'code': '11', 'title': {}, 'description': {}}}, cache_path)

    cfg = Config()
    token_cfg = TokenizationConfig(
        descriptions_parquet=cfg.data_loader.streaming.descriptions_parquet,
        tokenizer_name=cfg.data_loader.tokenization.tokenizer_name,
        max_length=cfg.data_loader.tokenization.max_length,
        output_path=str(cache_path),
    )

    result = validate_tokenization_cache(cfg, tokenization_cfg=token_cfg)
    assert result.valid is False
    assert any('wrong structure' in err for err in result.errors)

@pytest.mark.unit
def test_validate_training_config_reports_errors(tmp_path):
    cfg = Config()
    cfg.data_loader.streaming.descriptions_parquet = str(tmp_path / 'missing.parquet')

    result = validate_training_config(cfg)
    assert result.valid is False
    assert result.errors

@pytest.mark.unit
def test_require_valid_config_raises_validation_error(tmp_path):
    cfg = Config()
    cfg.data_loader.streaming.descriptions_parquet = str(tmp_path / 'missing.parquet')
    with pytest.raises(ValidationError):
        require_valid_config(cfg)
