'''
Unit tests for streaming dataset functions.

Tests cover:
- Phase 1 sampling utilities (_load_distance_matrix, _load_excluded_codes)
- Sampling weight computation (_compute_phase1_weights)
- SANS static sampling (_sample_negatives_sans_static)
- Cache path generation (_get_final_cache_path)
'''

import polars as pl
import pytest

from naics_embedder.data.positive_sampling import (
    build_anchor_list,
    build_taxonomy,
    create_positive_sampler,
)
from naics_embedder.text_model.dataloader.streaming_dataset import (
    _compute_phase1_weights,
    _get_final_cache_path,
    _load_distance_matrix,
    _load_excluded_codes,
    _sample_negatives_phase1,
    _sample_negatives_sans_static,
)
from naics_embedder.utils.config import SansStaticConfig, StreamingConfig

# -------------------------------------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------------------------------------

@pytest.fixture
def mock_codes_parquet(tmp_path):
    '''Create minimal 6-level NAICS codes for testing.'''
    data = {
        'code': [
            '31',
            '311',
            '3111',
            '31111',
            '311111',
            '311112',
            '32',
            '321',
            '3211',
            '32111',
            '321111',
        ],
        'level': [2, 3, 4, 5, 6, 6, 2, 3, 4, 5, 6],
        'index':
        list(range(11)),
    }
    df = pl.DataFrame(data)
    path = tmp_path / 'codes.parquet'
    df.write_parquet(path)
    return str(path)

@pytest.fixture
def mock_relations_parquet(tmp_path):
    '''Create minimal relations data for sibling sampling.'''
    data = {
        'code_i': ['311111', '311112'],
        'code_j': ['311112', '311111'],
        'relation_id': [2, 2],  # siblings
    }
    df = pl.DataFrame(data)
    path = tmp_path / 'relations.parquet'
    df.write_parquet(path)
    return str(path)

@pytest.fixture
def mock_distance_matrix(tmp_path):
    '''Create mock distance matrix with expected column format.'''
    data = {
        'idx_0-code_111': [0.0, 4.0, 6.0],
        'idx_1-code_222': [4.0, 0.0, 2.0],
        'idx_2-code_333': [6.0, 2.0, 0.0],
    }
    df = pl.DataFrame(data)
    path = tmp_path / 'distance_matrix.parquet'
    df.write_parquet(path)
    return str(path)

@pytest.fixture
def mock_descriptions_with_exclusions(tmp_path):
    '''Create descriptions with excluded_codes column.'''
    data = {
        'code': ['111', '222', '333'],
        'excluded_codes': [['222', '999'], None, ['111']],  # 999 doesn't exist
    }
    df = pl.DataFrame(data)
    path = tmp_path / 'descriptions.parquet'
    df.write_parquet(path)
    return str(path)

# -------------------------------------------------------------------------------------------------
# Taxonomy Utilities Tests (from positive_sampling module)
# -------------------------------------------------------------------------------------------------

def test_taxonomy_returns_six_digit_codes(mock_codes_parquet):
    '''build_taxonomy should extract all 6-digit codes with ancestor columns.'''
    result = build_taxonomy(mock_codes_parquet)

    assert 'code_6' in result.columns
    assert 'code_5' in result.columns
    assert 'code_4' in result.columns
    assert 'code_3' in result.columns
    assert 'code_2' in result.columns
    # Should have 3 six-digit codes: 311111, 311112, 321111
    assert result.height == 3

def test_taxonomy_normalizes_manufacturing_sectors(mock_codes_parquet):
    '''Sectors 31, 32, 33 should normalize to '31' in code_2.'''
    result = build_taxonomy(mock_codes_parquet)

    code_2_values = result.get_column('code_2').unique().to_list()
    # Both 31xxx and 32xxx codes should have code_2 = '31' (normalized)
    assert '31' in code_2_values
    # 32 should be normalized to 31
    assert '32' not in code_2_values

def test_taxonomy_preserves_code_hierarchy(mock_codes_parquet):
    '''Each row should have consistent code hierarchy.'''
    result = build_taxonomy(mock_codes_parquet)

    for row in result.iter_rows(named=True):
        code_6 = row['code_6']
        code_5 = row['code_5']
        code_4 = row['code_4']
        code_3 = row['code_3']

        # Each code should be prefix of the next
        assert code_6.startswith(code_5) or code_6[2:].startswith(code_5[2:])
        assert code_5.startswith(code_4) or code_5[2:].startswith(code_4[2:])
        assert code_4.startswith(code_3) or code_4[2:].startswith(code_3[2:])

def test_anchor_list_extracts_all_codes(mock_codes_parquet):
    '''build_anchor_list should return all codes with level >= 2.'''
    result = build_anchor_list(mock_codes_parquet)

    assert 'level' in result.columns
    assert 'anchor' in result.columns
    # Should have all 11 codes
    assert result.height == 11

# -------------------------------------------------------------------------------------------------
# Matrix Loading Tests
# -------------------------------------------------------------------------------------------------

def test_load_distance_matrix_structure(mock_distance_matrix):
    '''Distance matrix should correctly parse column format.'''
    code_to_idx = {'111': 0, '222': 1, '333': 2}
    idx_to_code = {0: '111', 1: '222', 2: '333'}

    result = _load_distance_matrix(mock_distance_matrix, code_to_idx, idx_to_code)

    # Should create lookup for code pairs
    assert ('111', '222') in result
    assert result[('111', '222')] == 4.0
    assert result[('111', '333')] == 6.0

def test_load_distance_matrix_diagonal_is_zero(mock_distance_matrix):
    '''Diagonal entries (self-distance) should be zero.'''
    code_to_idx = {'111': 0, '222': 1, '333': 2}
    idx_to_code = {0: '111', 1: '222', 2: '333'}

    result = _load_distance_matrix(mock_distance_matrix, code_to_idx, idx_to_code)

    assert result[('111', '111')] == 0.0
    assert result[('222', '222')] == 0.0

def test_load_excluded_codes_filters_unknown(mock_descriptions_with_exclusions):
    '''Unknown excluded codes should be filtered out.'''
    code_to_idx = {'111': 0, '222': 1, '333': 2}

    result = _load_excluded_codes(mock_descriptions_with_exclusions, code_to_idx)

    # Only valid exclusions should remain
    assert result['111'] == {'222'}
    assert '999' not in result.get('111', set())

def test_load_excluded_codes_handles_null(mock_descriptions_with_exclusions):
    '''Codes with null excluded_codes should not appear in result.'''

    result = _load_excluded_codes(mock_descriptions_with_exclusions)

    # '222' has null excluded_codes, should not be in result
    assert '222' not in result

def test_load_excluded_codes_bidirectional(mock_descriptions_with_exclusions):
    '''Multiple codes can have exclusions.'''
    code_to_idx = {'111': 0, '222': 1, '333': 2}

    result = _load_excluded_codes(mock_descriptions_with_exclusions, code_to_idx)

    assert '111' in result
    assert '333' in result
    assert result['333'] == {'111'}

# -------------------------------------------------------------------------------------------------
# Sampling Weight Computation Tests
# -------------------------------------------------------------------------------------------------

def test_sibling_masking_sets_weight_to_zero():
    '''Distance == 2 (siblings) should have weight 0.'''
    anchor = '111'
    candidates = [
        {
            'negative_code': '112',
            'negative_idx': 0
        },  # sibling
        {
            'negative_code': '221',
            'negative_idx': 1
        },  # non-sibling
    ]
    distance_lookup = {
        (anchor, '112'): 2.0,  # sibling distance
        (anchor, '221'): 4.0,
    }

    weights, _ = _compute_phase1_weights(
        anchor_code=anchor,
        anchor_idx=0,
        candidate_negatives=candidates,
        distance_lookup=distance_lookup,
        excluded_map={},
        code_to_idx={
            '111': 0,
            '112': 1,
            '221': 2
        },
    )

    assert weights[0] == 0.0  # sibling masked
    assert weights[1] > 0.0  # non-sibling has weight

def test_alpha_parameter_affects_distribution():
    '''Higher alpha should concentrate weights on closer distances.'''
    anchor = '111'
    candidates = [
        {
            'negative_code': '222',
            'negative_idx': 0
        },
        {
            'negative_code': '333',
            'negative_idx': 1
        },
    ]
    distance_lookup = {
        (anchor, '222'): 4.0,
        (anchor, '333'): 8.0,
    }

    weights_low_alpha, _ = _compute_phase1_weights(
        anchor_code=anchor,
        anchor_idx=0,
        candidate_negatives=candidates,
        distance_lookup=distance_lookup,
        excluded_map={},
        code_to_idx={},
        alpha=1.0,
    )

    weights_high_alpha, _ = _compute_phase1_weights(
        anchor_code=anchor,
        anchor_idx=0,
        candidate_negatives=candidates,
        distance_lookup=distance_lookup,
        excluded_map={},
        code_to_idx={},
        alpha=3.0,
    )

    # Higher alpha should make ratio of weights more extreme
    ratio_low = weights_low_alpha[0] / weights_low_alpha[1]
    ratio_high = weights_high_alpha[0] / weights_high_alpha[1]
    assert ratio_high > ratio_low

def test_exclusion_weight_prioritizes_excluded_codes():
    '''Excluded codes should receive high constant weight.'''
    anchor = '111'
    candidates = [
        {
            'negative_code': '222',
            'negative_idx': 0
        },
        {
            'negative_code': '333',
            'negative_idx': 1
        },  # excluded
    ]
    distance_lookup = {
        (anchor, '222'): 4.0,
        (anchor, '333'): 4.0,  # same distance
    }
    excluded_map = {anchor: {'333'}}

    weights, mask = _compute_phase1_weights(
        anchor_code=anchor,
        anchor_idx=0,
        candidate_negatives=candidates,
        distance_lookup=distance_lookup,
        excluded_map=excluded_map,
        code_to_idx={
            '111': 0,
            '222': 1,
            '333': 2
        },
        exclusion_weight=100.0,
    )

    # Excluded code should have much higher weight
    assert weights[1] == 100.0
    assert weights[1] > weights[0]
    assert mask[1] == True  # noqa: E712 - numpy bool comparison
    assert mask[0] == False  # noqa: E712 - numpy bool comparison

def test_missing_distance_uses_default():
    '''Missing distance lookup should use default large distance.'''
    anchor = '111'
    candidates = [
        {
            'negative_code': '222',
            'negative_idx': 0
        },
        {
            'negative_code': '999',
            'negative_idx': 1
        },  # not in lookup
    ]
    distance_lookup = {
        (anchor, '222'): 4.0,
        # '999' not in lookup
    }

    weights, _ = _compute_phase1_weights(
        anchor_code=anchor,
        anchor_idx=0,
        candidate_negatives=candidates,
        distance_lookup=distance_lookup,
        excluded_map={},
        code_to_idx={},
    )

    # Both should have positive weights
    assert weights[0] > 0.0
    assert weights[1] > 0.0
    # Missing distance uses default 12.0, so weight should be lower
    assert weights[0] > weights[1]

def test_fallback_to_uniform_when_all_zero():
    '''When all weights are zero, should use uniform sampling.'''
    anchor = '111'
    candidates = [
        {
            'negative_code': '112',
            'negative_idx': 0
        },
        {
            'negative_code': '113',
            'negative_idx': 1
        },
    ]
    # All siblings -> all weights zero
    distance_lookup = {
        (anchor, '112'): 2.0,
        (anchor, '113'): 2.0,
    }

    sampled = _sample_negatives_phase1(
        anchor_code=anchor,
        anchor_idx=0,
        candidate_negatives=candidates,
        n_negatives=1,
        distance_lookup=distance_lookup,
        excluded_map={},
        code_to_idx={},
        seed=42,
    )

    # Should still return samples (uniform fallback)
    assert len(sampled) == 1

def test_sample_respects_n_negatives_limit():
    '''Should not sample more than n_negatives.'''
    anchor = '111'
    candidates = [
        {
            'negative_code': '222',
            'negative_idx': 0
        },
        {
            'negative_code': '333',
            'negative_idx': 1
        },
        {
            'negative_code': '444',
            'negative_idx': 2
        },
        {
            'negative_code': '555',
            'negative_idx': 3
        },
    ]
    distance_lookup = {(anchor, c['negative_code']): 4.0 for c in candidates}

    sampled = _sample_negatives_phase1(
        anchor_code=anchor,
        anchor_idx=0,
        candidate_negatives=candidates,
        n_negatives=2,
        distance_lookup=distance_lookup,
        excluded_map={},
        code_to_idx={},
        seed=42,
    )

    assert len(sampled) == 2

def test_sample_handles_fewer_candidates_than_requested():
    '''When candidates < n_negatives, return all candidates.'''
    anchor = '111'
    candidates = [
        {
            'negative_code': '222',
            'negative_idx': 0
        },
    ]
    distance_lookup = {(anchor, '222'): 4.0}

    sampled = _sample_negatives_phase1(
        anchor_code=anchor,
        anchor_idx=0,
        candidate_negatives=candidates,
        n_negatives=5,
        distance_lookup=distance_lookup,
        excluded_map={},
        code_to_idx={},
        seed=42,
    )

    assert len(sampled) == 1

# -------------------------------------------------------------------------------------------------
# SANS Static Sampling Tests
# -------------------------------------------------------------------------------------------------

def test_sans_empty_near_bucket():
    '''When near bucket is empty, all weight goes to far bucket.'''
    anchor = '111'
    candidates = [
        {
            'negative_code': '888',
            'negative_idx': 0
        },
        {
            'negative_code': '999',
            'negative_idx': 1
        },
    ]
    # All far (distance > 4)
    distance_lookup = {
        (anchor, '888'): 10.0,
        (anchor, '999'): 12.0,
    }

    sans_cfg = SansStaticConfig(
        near_distance_threshold=4.0,
        near_bucket_weight=0.8,
        far_bucket_weight=0.2,
    )

    sampled, metadata = _sample_negatives_sans_static(
        anchor_code=anchor,
        candidate_negatives=candidates,
        n_negatives=2,
        distance_lookup=distance_lookup,
        sans_cfg=sans_cfg,
        seed=42,
    )

    assert metadata['candidates_near'] == 0
    assert metadata['candidates_far'] == 2
    assert metadata['sampled_near'] == 0
    assert metadata['sampled_far'] == 2

def test_sans_empty_far_bucket():
    '''When far bucket is empty, all weight goes to near bucket.'''
    anchor = '111'
    candidates = [
        {
            'negative_code': '222',
            'negative_idx': 0
        },
        {
            'negative_code': '333',
            'negative_idx': 1
        },
    ]
    # All near (distance <= 4)
    distance_lookup = {
        (anchor, '222'): 2.0,
        (anchor, '333'): 3.0,
    }

    sans_cfg = SansStaticConfig(
        near_distance_threshold=4.0,
        near_bucket_weight=0.8,
        far_bucket_weight=0.2,
    )

    sampled, metadata = _sample_negatives_sans_static(
        anchor_code=anchor,
        candidate_negatives=candidates,
        n_negatives=2,
        distance_lookup=distance_lookup,
        sans_cfg=sans_cfg,
        seed=42,
    )

    assert metadata['candidates_near'] == 2
    assert metadata['candidates_far'] == 0
    assert metadata['sampled_near'] == 2
    assert metadata['sampled_far'] == 0

def test_sans_symmetric_distance_lookup():
    '''Should check reversed key when forward lookup fails.'''
    anchor = '111'
    candidates = [
        {
            'negative_code': '222',
            'negative_idx': 0
        },
    ]
    # Only reversed key exists
    distance_lookup = {
        ('222', anchor): 3.0,  # Near (< 4)
    }

    sans_cfg = SansStaticConfig(near_distance_threshold=4.0)

    sampled, metadata = _sample_negatives_sans_static(
        anchor_code=anchor,
        candidate_negatives=candidates,
        n_negatives=1,
        distance_lookup=distance_lookup,
        sans_cfg=sans_cfg,
        seed=42,
    )

    assert metadata['candidates_near'] == 1

def test_sans_metadata_weight_tracking():
    '''Metadata should track effective weights correctly.'''
    anchor = '111'
    candidates = [
        {
            'negative_code': '222',
            'negative_idx': 0
        },  # near
        {
            'negative_code': '333',
            'negative_idx': 1
        },  # far
        {
            'negative_code': '444',
            'negative_idx': 2
        },  # far
    ]
    distance_lookup = {
        (anchor, '222'): 2.0,
        (anchor, '333'): 8.0,
        (anchor, '444'): 10.0,
    }

    sans_cfg = SansStaticConfig(
        near_distance_threshold=4.0,
        near_bucket_weight=0.6,
        far_bucket_weight=0.4,
    )

    _, metadata = _sample_negatives_sans_static(
        anchor_code=anchor,
        candidate_negatives=candidates,
        n_negatives=2,
        distance_lookup=distance_lookup,
        sans_cfg=sans_cfg,
        seed=42,
    )

    assert metadata['candidates_near'] == 1
    assert metadata['candidates_far'] == 2
    # Effective weights should sum to ~1.0
    total_weight = metadata['effective_near_weight'] + metadata['effective_far_weight']
    assert abs(total_weight - 1.0) < 1e-6

def test_sans_empty_candidates():
    '''Empty candidates should return empty list.'''
    sans_cfg = SansStaticConfig()

    sampled, metadata = _sample_negatives_sans_static(
        anchor_code='111',
        candidate_negatives=[],
        n_negatives=5,
        distance_lookup={},
        sans_cfg=sans_cfg,
        seed=42,
    )

    assert sampled == []
    assert metadata['candidates_near'] == 0
    assert metadata['candidates_far'] == 0

# -------------------------------------------------------------------------------------------------
# Cache Path Tests
# -------------------------------------------------------------------------------------------------

def test_cache_key_deterministic():
    '''Same config should always produce same cache key.'''
    cfg1 = StreamingConfig(seed=42, n_negatives=5)
    cfg2 = StreamingConfig(seed=42, n_negatives=5)

    path1 = _get_final_cache_path(cfg1)
    path2 = _get_final_cache_path(cfg2)

    assert path1 == path2

def test_cache_key_changes_with_config():
    '''Different configs should produce different cache keys.'''
    cfg1 = StreamingConfig(seed=42)
    cfg2 = StreamingConfig(seed=43)

    path1 = _get_final_cache_path(cfg1)
    path2 = _get_final_cache_path(cfg2)

    assert path1 != path2

def test_cache_path_in_streaming_cache_dir():
    '''Cache path should be in streaming_cache directory.'''
    cfg = StreamingConfig(descriptions_parquet='./data/naics_descriptions.parquet')

    path = _get_final_cache_path(cfg)

    assert 'streaming_cache' in str(path)
    assert path.name.startswith('streaming_final_')

# -------------------------------------------------------------------------------------------------
# Positive Sampler Tests
# -------------------------------------------------------------------------------------------------

class TestPositiveSampler:
    '''Tests for PositiveSampler class.'''

    def test_sampler_initialization(self, mock_codes_parquet, mock_relations_parquet, monkeypatch):
        '''PositiveSampler should initialize with anchor list.'''
        # Mock get_indices_codes
        code_to_idx = {
            '31': 0,
            '311': 1,
            '3111': 2,
            '31111': 3,
            '311111': 4,
            '311112': 5,
            '32': 6,
            '321': 7,
            '3211': 8,
            '32111': 9,
            '321111': 10,
        }
        idx_to_code = {v: k for k, v in code_to_idx.items()}

        def mock_get_indices_codes(key):
            if key == 'code_to_idx':
                return code_to_idx
            elif key == 'idx_to_code':
                return idx_to_code
            return None

        monkeypatch.setattr(
            'naics_embedder.data.positive_sampling.get_indices_codes', mock_get_indices_codes
        )

        sampler = create_positive_sampler(
            descriptions_parquet=mock_codes_parquet,
            relations_parquet=mock_relations_parquet,
            max_per_stratum=4,
            seed=42,
        )

        assert len(sampler.anchors) > 0

    def test_sample_positives_returns_list(
        self, mock_codes_parquet, mock_relations_parquet, monkeypatch
    ):
        '''sample_positives should return list of positive dicts.'''
        code_to_idx = {
            '31': 0,
            '311': 1,
            '3111': 2,
            '31111': 3,
            '311111': 4,
            '311112': 5,
            '32': 6,
            '321': 7,
            '3211': 8,
            '32111': 9,
            '321111': 10,
        }
        idx_to_code = {v: k for k, v in code_to_idx.items()}

        def mock_get_indices_codes(key):
            if key == 'code_to_idx':
                return code_to_idx
            elif key == 'idx_to_code':
                return idx_to_code
            return None

        monkeypatch.setattr(
            'naics_embedder.data.positive_sampling.get_indices_codes', mock_get_indices_codes
        )

        sampler = create_positive_sampler(
            descriptions_parquet=mock_codes_parquet,
            relations_parquet=mock_relations_parquet,
            max_per_stratum=4,
            seed=42,
        )

        # Sample positives for first anchor
        if sampler.anchors:
            positives = sampler.sample_positives(sampler.anchors[0])
            assert isinstance(positives, list)
            for p in positives:
                assert 'positive_idx' in p
                assert 'positive_code' in p
                assert 'stratum_id' in p
                assert 'stratum_wgt' in p
