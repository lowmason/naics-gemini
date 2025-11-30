'''
Unit tests for HGCN streaming dataset functions.

Tests cover:
- Cache key generation and path handling
- Streaming generator output structure
- Deterministic behavior with same seed
'''

import pytest

from naics_embedder.graph_model.dataloader import hgcn_streaming_dataset as streaming
from naics_embedder.utils.config import StreamingConfig

def _sample_cached_triplets():
    '''Create cached triplets resembling load_streaming_triplets output (new flat format).'''

    def _neg(negative_idx):
        return {
            'negative_idx': negative_idx,
            'negative_code': f'N{negative_idx}',
            'relation_margin': 0.1 * negative_idx,
            'distance_margin': 0.05 * negative_idx,
        }

    return [
        {
            'anchor_idx': 1,
            'anchor_code': '11',
            'positive_idx': 101,
            'positive_code': '110',
            'positive_level': 3,
            'stratum_id': 0,
            'stratum_wgt': 0.5,
            'negatives': [_neg(201), _neg(202)],
        },
        {
            'anchor_idx': 2,
            'anchor_code': '12',
            'positive_idx': 102,
            'positive_code': '120',
            'positive_level': 3,
            'stratum_id': 0,
            'stratum_wgt': 0.5,
            'negatives': [_neg(301)],
        },
    ]

@pytest.mark.unit
def test_streaming_triplet_access(monkeypatch: pytest.MonkeyPatch):
    '''Streaming generator should provide deterministic, indexable triplets.'''
    cached = _sample_cached_triplets()
    cfg = StreamingConfig(n_negatives=2, seed=7)

    load_calls = []

    def fake_loader(cfg_arg, *, worker_id='Main', allow_cache_save=True, log_stats=True):
        load_calls.append(worker_id)
        return cached

    monkeypatch.setattr(streaming, 'load_streaming_triplets', fake_loader)

    first_pass = list(streaming.create_streaming_generator(cfg))
    second_pass = list(streaming.create_streaming_generator(cfg))

    assert load_calls == ['Main', 'Main']
    assert first_pass == second_pass
    assert first_pass[1]['anchor_idx'] == cached[1]['anchor_idx']
    assert set(first_pass[0]['negatives'][0].keys()) == {
        'negative_idx',
        'negative_code',
        'relation_margin',
        'distance_margin',
    }

@pytest.mark.unit
def test_cache_key_deterministic():
    '''Same config should produce same cache key.'''
    cfg1 = StreamingConfig(seed=42, n_negatives=5)
    cfg2 = StreamingConfig(seed=42, n_negatives=5)

    key1 = streaming._get_cache_key(cfg1)
    key2 = streaming._get_cache_key(cfg2)

    assert key1 == key2

@pytest.mark.unit
def test_cache_key_changes_with_seed():
    '''Different seeds should produce different cache keys.'''
    cfg1 = StreamingConfig(seed=42, n_negatives=5)
    cfg2 = StreamingConfig(seed=43, n_negatives=5)

    key1 = streaming._get_cache_key(cfg1)
    key2 = streaming._get_cache_key(cfg2)

    assert key1 != key2

@pytest.mark.unit
def test_cache_key_changes_with_n_negatives():
    '''Different n_negatives should produce different cache keys.'''
    cfg1 = StreamingConfig(seed=42, n_negatives=5)
    cfg2 = StreamingConfig(seed=42, n_negatives=10)

    key1 = streaming._get_cache_key(cfg1)
    key2 = streaming._get_cache_key(cfg2)

    assert key1 != key2

@pytest.mark.unit
def test_cache_path_in_streaming_cache_dir():
    '''Cache path should be in streaming_cache directory.'''
    cfg = StreamingConfig(descriptions_parquet='./data/naics_descriptions.parquet')

    path = streaming._get_final_cache_path(cfg)

    assert 'streaming_cache' in str(path)
    assert path.name.startswith('streaming_final_')

@pytest.mark.unit
def test_generator_yields_expected_fields(monkeypatch: pytest.MonkeyPatch):
    '''Generator should yield dicts with expected fields.'''
    cached = _sample_cached_triplets()
    cfg = StreamingConfig(n_negatives=2, seed=7)

    def fake_loader(cfg_arg, *, worker_id='Main', allow_cache_save=True, log_stats=True):
        return cached

    monkeypatch.setattr(streaming, 'load_streaming_triplets', fake_loader)

    results = list(streaming.create_streaming_generator(cfg))

    assert len(results) == 2
    for item in results:
        assert 'anchor_idx' in item
        assert 'anchor_code' in item
        assert 'positive_idx' in item
        assert 'positive_code' in item
        assert 'negatives' in item
        assert 'positive_level' in item
        assert 'stratum_id' in item
        assert 'stratum_wgt' in item

@pytest.mark.unit
def test_generator_negatives_structure(monkeypatch: pytest.MonkeyPatch):
    '''Negatives should have expected structure.'''
    cached = _sample_cached_triplets()
    cfg = StreamingConfig(n_negatives=2, seed=7)

    def fake_loader(cfg_arg, *, worker_id='Main', allow_cache_save=True, log_stats=True):
        return cached

    monkeypatch.setattr(streaming, 'load_streaming_triplets', fake_loader)

    results = list(streaming.create_streaming_generator(cfg))

    for item in results:
        for neg in item['negatives']:
            assert 'negative_idx' in neg
            assert 'negative_code' in neg
            assert 'relation_margin' in neg
            assert 'distance_margin' in neg
