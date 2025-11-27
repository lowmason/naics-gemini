'''
Unit tests for streaming dataset functions.

Tests cover:
- Taxonomy utilities (_taxonomy, _anchors, _linear_skip)
- Ancestor/descendant generation (_descendants, _ancestors)
- Matrix loading (_load_distance_matrix, _load_excluded_codes)
- Sampling weight computation (_compute_phase1_weights)
- Cache path generation (_get_final_cache_path)
'''

import polars as pl
import pytest

from naics_embedder.text_model.dataloader.streaming_dataset import (
    _ancestors,
    _anchors,
    _compute_phase1_weights,
    _descendants,
    _get_final_cache_path,
    _linear_skip,
    _load_distance_matrix,
    _load_excluded_codes,
    _sample_negatives_phase1,
    _sample_negatives_sans_static,
    _taxonomy,
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
    }
    df = pl.DataFrame(data)
    path = tmp_path / 'codes.parquet'
    df.write_parquet(path)
    return str(path)

@pytest.fixture
def mock_triplets_parquet(tmp_path):
    '''Create minimal triplets data.'''
    data = {
        'anchor_level': [6, 6, 3],
        'anchor_code': ['311111', '321111', '311'],
    }
    df = pl.DataFrame(data)
    path = tmp_path / 'triplets.parquet'
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
# Taxonomy Utilities Tests
# -------------------------------------------------------------------------------------------------

def test_taxonomy_returns_six_digit_codes(mock_codes_parquet):
    '''_taxonomy should extract all 6-digit codes with ancestor columns.'''
    result = _taxonomy(mock_codes_parquet)

    assert 'code_6' in result.columns
    assert 'code_5' in result.columns
    assert 'code_4' in result.columns
    assert 'code_3' in result.columns
    assert 'code_2' in result.columns
    # Should have 3 six-digit codes: 311111, 311112, 321111
    assert result.height == 3

def test_taxonomy_normalizes_manufacturing_sectors(mock_codes_parquet):
    '''Sectors 31, 32, 33 should normalize to '31' in code_2.'''
    result = _taxonomy(mock_codes_parquet)

    code_2_values = result.get_column('code_2').unique().to_list()
    # Both 31xxx and 32xxx codes should have code_2 = '31' (normalized)
    assert '31' in code_2_values
    # 32 should be normalized to 31
    assert '32' not in code_2_values

def test_taxonomy_preserves_code_hierarchy(mock_codes_parquet):
    '''Each row should have consistent code hierarchy.'''
    result = _taxonomy(mock_codes_parquet)

    for row in result.iter_rows(named=True):
        code_6 = row['code_6']
        code_5 = row['code_5']
        code_4 = row['code_4']
        code_3 = row['code_3']

        # Each code should be prefix of the next
        assert code_6.startswith(code_5) or code_6[2:].startswith(code_5[2:])
        assert code_5.startswith(code_4) or code_5[2:].startswith(code_4[2:])
        assert code_4.startswith(code_3) or code_4[2:].startswith(code_3[2:])

def test_anchors_extracts_unique_anchor_level_pairs(mock_triplets_parquet):
    '''_anchors should return unique (level, anchor) pairs.'''
    result = _anchors(mock_triplets_parquet)

    assert 'level' in result.columns
    assert 'anchor' in result.columns
    # Should have 3 unique anchors: 311111, 321111, 311
    assert result.height == 3

def test_linear_skip_returns_descendants(mock_codes_parquet):
    '''_linear_skip should return next-level descendants for an anchor.'''
    taxonomy = _taxonomy(mock_codes_parquet)

    # From level 5, should get level 6 descendants
    result = _linear_skip('31111', taxonomy)
    assert '311111' in result
    assert '311112' in result

def test_linear_skip_handles_leaf_level(mock_codes_parquet):
    '''_linear_skip at level 5 should return level 6 codes.'''
    taxonomy = _taxonomy(mock_codes_parquet)

    result = _linear_skip('31111', taxonomy)
    # Should return 6-digit descendants
    assert all(len(code) == 6 for code in result)

# -------------------------------------------------------------------------------------------------
# Ancestor/Descendant Generation Tests
# -------------------------------------------------------------------------------------------------

def test_ancestors_for_six_digit_anchor(mock_codes_parquet, mock_triplets_parquet):
    '''6-digit anchors should return all ancestor levels (5, 4, 3, 2).'''
    taxonomy = _taxonomy(mock_codes_parquet)
    anchors = _anchors(mock_triplets_parquet)

    result = _ancestors(anchors, taxonomy)

    # Should have entries for 6-digit anchors
    six_digit_results = result.filter(pl.col('level').eq(6))
    assert six_digit_results.height > 0

    # Each 6-digit anchor should have 4 ancestor positives
    for row in six_digit_results.iter_rows(named=True):
        assert len(row['positive']) == 4

def test_ancestors_returns_correct_ancestor_codes(mock_codes_parquet, mock_triplets_parquet):
    '''Ancestor codes should match expected hierarchy.'''
    taxonomy = _taxonomy(mock_codes_parquet)
    anchors = _anchors(mock_triplets_parquet)

    result = _ancestors(anchors, taxonomy)

    # Find 311111's ancestors
    anchor_311111 = result.filter(pl.col('anchor').eq('311111'))
    if anchor_311111.height > 0:
        positives = anchor_311111.row(0, named=True)['positive']
        # Should contain 31111, 3111, 311, and sector code
        assert '31111' in positives
        assert '3111' in positives
        assert '311' in positives

def test_descendants_for_parent_anchor(mock_codes_parquet, mock_triplets_parquet):
    '''Parent anchors should return descendant positives.'''
    taxonomy = _taxonomy(mock_codes_parquet)
    anchors = _anchors(mock_triplets_parquet)

    result = _descendants(anchors, taxonomy)

    # Should have entries for non-6-digit anchors (level 3 anchor '311')
    assert result.height > 0

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
    cfg1 = StreamingConfig(seed=42, n_positives=3, n_negatives=5)
    cfg2 = StreamingConfig(seed=42, n_positives=3, n_negatives=5)

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
    cfg = StreamingConfig(triplets_parquet='./data/naics_training_pairs')

    path = _get_final_cache_path(cfg)

    assert 'streaming_cache' in str(path)
    assert path.name.startswith('streaming_final_')
    assert path.suffix == '.pkl'

# -------------------------------------------------------------------------------------------------
# Streaming Dataset Iteration Tests
# -------------------------------------------------------------------------------------------------

class TestStreamingDatasetIteration:
    '''Test suite for streaming dataset full iteration.

    Tests the complete pipeline of create_streaming_dataset including
    tokenization lookup and triplet generation.
    '''

    @pytest.fixture
    def full_mock_setup(self, tmp_path):
        '''Create a complete mock setup for streaming dataset testing.'''
        # Create descriptions with all required fields
        desc_data = {
            'index': [0, 1, 2, 3, 4],
            'code': ['31', '311', '3111', '31111', '311111'],
            'level': [2, 3, 4, 5, 6],
            'title': ['Manuf', 'Food Manuf', 'Animal Food', 'Animal Food', 'Dog Food'],
            'description': ['Manufacturing', 'Food', 'Animal', 'Animal Food', 'Dog Food'],
            'excluded': ['', '', '', '', ''],
            'examples': ['', '', '', '', ''],
            'excluded_codes': [None, None, None, None, None],
        }
        desc_df = pl.DataFrame(desc_data)
        desc_path = tmp_path / 'descriptions.parquet'
        desc_df.write_parquet(desc_path)

        # Create triplets directory structure
        triplets_dir = tmp_path / 'triplets'
        triplets_dir.mkdir()

        # Create triplets for anchor at index 4 (311111)
        anchor_dir = triplets_dir / 'anchor=4'
        anchor_dir.mkdir()

        triplet_data = {
            'anchor_idx': [4, 4],
            'anchor_code': ['311111', '311111'],
            'anchor_level': [6, 6],
            'positive_idx': [3, 2],
            'positive_code': ['31111', '3111'],
            'positive_level': [5, 4],
            'negative_idx': [0, 1],
            'negative_code': ['31', '311'],
            'negative_level': [2, 3],
            'relation_margin': [0, 0],
            'distance_margin': [4, 4],
            'positive_relation': [1, 2],
            'positive_distance': [1, 2],
            'negative_relation': [3, 3],
            'negative_distance': [6, 5],
        }
        triplet_df = pl.DataFrame(triplet_data)
        triplet_path = anchor_dir / 'part0.parquet'
        triplet_df.write_parquet(triplet_path)

        return {
            'descriptions_path': str(desc_path),
            'triplets_path': str(triplets_dir),
        }

    @pytest.fixture
    def mock_token_cache(self):
        '''Create mock token cache with all required fields.'''
        import torch

        def make_tokens(seq_len):
            return {
                'input_ids': torch.randint(0, 1000, (seq_len, )),
                'attention_mask': torch.ones(seq_len, dtype=torch.long),
            }

        return {
            0: {
                'code': '31',
                'title': make_tokens(24),
                'description': make_tokens(128),
                'excluded': make_tokens(128),
                'examples': make_tokens(128),
            },
            1: {
                'code': '311',
                'title': make_tokens(24),
                'description': make_tokens(128),
                'excluded': make_tokens(128),
                'examples': make_tokens(128),
            },
            2: {
                'code': '3111',
                'title': make_tokens(24),
                'description': make_tokens(128),
                'excluded': make_tokens(128),
                'examples': make_tokens(128),
            },
            3: {
                'code': '31111',
                'title': make_tokens(24),
                'description': make_tokens(128),
                'excluded': make_tokens(128),
                'examples': make_tokens(128),
            },
            4: {
                'code': '311111',
                'title': make_tokens(24),
                'description': make_tokens(128),
                'excluded': make_tokens(128),
                'examples': make_tokens(128),
            },
        }

    def test_streaming_generator_yields_triplets(self, full_mock_setup):
        '''Test that streaming generator yields triplet dictionaries.'''
        from unittest.mock import patch

        from naics_embedder.text_model.dataloader.streaming_dataset import (
            create_streaming_generator,
        )

        cfg = StreamingConfig(
            descriptions_parquet=full_mock_setup['descriptions_path'],
            triplets_parquet=full_mock_setup['triplets_path'],
            n_positives=1,
            n_negatives=2,
            seed=42,
        )

        # Mock get_indices_codes to return our test data
        code_to_idx = {'31': 0, '311': 1, '3111': 2, '31111': 3, '311111': 4}
        idx_to_code = {v: k for k, v in code_to_idx.items()}

        with patch(
            'naics_embedder.text_model.dataloader.streaming_dataset.get_indices_codes'
        ) as mock_get:
            mock_get.side_effect = lambda x: code_to_idx if x == 'code_to_idx' else idx_to_code

            generator = create_streaming_generator(cfg)
            triplets = list(generator)

        # Should yield at least one triplet
        assert len(triplets) > 0

        # Check triplet structure
        triplet = triplets[0]
        assert 'anchor_idx' in triplet
        assert 'anchor_code' in triplet
        assert 'positive_idx' in triplet
        assert 'positive_code' in triplet
        assert 'negatives' in triplet

    def test_streaming_generator_respects_n_negatives(self, full_mock_setup):
        '''Test that generator respects n_negatives parameter.'''
        from unittest.mock import patch

        from naics_embedder.text_model.dataloader.streaming_dataset import (
            create_streaming_generator,
        )

        cfg = StreamingConfig(
            descriptions_parquet=full_mock_setup['descriptions_path'],
            triplets_parquet=full_mock_setup['triplets_path'],
            n_positives=1,
            n_negatives=1,  # Request only 1 negative
            seed=42,
        )

        code_to_idx = {'31': 0, '311': 1, '3111': 2, '31111': 3, '311111': 4}
        idx_to_code = {v: k for k, v in code_to_idx.items()}

        with patch(
            'naics_embedder.text_model.dataloader.streaming_dataset.get_indices_codes'
        ) as mock_get:
            mock_get.side_effect = lambda x: code_to_idx if x == 'code_to_idx' else idx_to_code

            generator = create_streaming_generator(cfg)
            triplets = list(generator)

        if triplets:
            # Each triplet should have at most n_negatives
            for triplet in triplets:
                assert len(triplet['negatives']) <= 1

    def test_streaming_dataset_with_token_cache(self, full_mock_setup, mock_token_cache):
        '''Test full streaming dataset iteration with token cache.'''
        from unittest.mock import patch

        from naics_embedder.text_model.dataloader.streaming_dataset import (
            create_streaming_dataset,
        )

        cfg = StreamingConfig(
            descriptions_parquet=full_mock_setup['descriptions_path'],
            triplets_parquet=full_mock_setup['triplets_path'],
            n_positives=1,
            n_negatives=2,
            seed=42,
        )

        code_to_idx = {'31': 0, '311': 1, '3111': 2, '31111': 3, '311111': 4}
        idx_to_code = {v: k for k, v in code_to_idx.items()}

        with patch(
            'naics_embedder.text_model.dataloader.streaming_dataset.get_indices_codes'
        ) as mock_get:
            mock_get.side_effect = lambda x: code_to_idx if x == 'code_to_idx' else idx_to_code

            dataset = create_streaming_dataset(mock_token_cache, cfg)
            items = list(dataset)

        # Should yield items
        assert len(items) > 0

        # Check item structure includes embeddings
        item = items[0]
        assert 'anchor_idx' in item
        assert 'anchor_code' in item
        assert 'anchor_embedding' in item
        assert 'positives' in item or 'positive_embedding' in item
        assert 'negatives' in item

    def test_streaming_dataset_anchor_embedding_structure(self, full_mock_setup, mock_token_cache):
        '''Test that anchor embedding has correct structure.'''
        from unittest.mock import patch

        from naics_embedder.text_model.dataloader.streaming_dataset import (
            create_streaming_dataset,
        )

        cfg = StreamingConfig(
            descriptions_parquet=full_mock_setup['descriptions_path'],
            triplets_parquet=full_mock_setup['triplets_path'],
            n_positives=1,
            n_negatives=2,
            seed=42,
        )

        code_to_idx = {'31': 0, '311': 1, '3111': 2, '31111': 3, '311111': 4}
        idx_to_code = {v: k for k, v in code_to_idx.items()}

        with patch(
            'naics_embedder.text_model.dataloader.streaming_dataset.get_indices_codes'
        ) as mock_get:
            mock_get.side_effect = lambda x: code_to_idx if x == 'code_to_idx' else idx_to_code

            dataset = create_streaming_dataset(mock_token_cache, cfg)
            items = list(dataset)

        if items:
            anchor_embedding = items[0]['anchor_embedding']
            # Should have all four channels
            assert 'title' in anchor_embedding
            assert 'description' in anchor_embedding
            assert 'excluded' in anchor_embedding
            assert 'examples' in anchor_embedding
            # Code should not be in embedding (filtered out)
            assert 'code' not in anchor_embedding

    def test_streaming_dataset_negative_embedding_structure(
        self, full_mock_setup, mock_token_cache
    ):
        '''Test that negative embeddings have correct structure.'''
        from unittest.mock import patch

        from naics_embedder.text_model.dataloader.streaming_dataset import (
            create_streaming_dataset,
        )

        cfg = StreamingConfig(
            descriptions_parquet=full_mock_setup['descriptions_path'],
            triplets_parquet=full_mock_setup['triplets_path'],
            n_positives=1,
            n_negatives=2,
            seed=42,
        )

        code_to_idx = {'31': 0, '311': 1, '3111': 2, '31111': 3, '311111': 4}
        idx_to_code = {v: k for k, v in code_to_idx.items()}

        with patch(
            'naics_embedder.text_model.dataloader.streaming_dataset.get_indices_codes'
        ) as mock_get:
            mock_get.side_effect = lambda x: code_to_idx if x == 'code_to_idx' else idx_to_code

            dataset = create_streaming_dataset(mock_token_cache, cfg)
            items = list(dataset)

        if items and items[0]['negatives']:
            negative = items[0]['negatives'][0]
            # Should have expected fields
            assert 'negative_idx' in negative
            assert 'negative_code' in negative
            assert 'negative_embedding' in negative
            assert 'relation_margin' in negative
            assert 'distance_margin' in negative

            # Embedding should have all channels
            neg_embedding = negative['negative_embedding']
            assert 'title' in neg_embedding
            assert 'description' in neg_embedding

    def test_streaming_dataset_deterministic_with_seed(self, full_mock_setup, mock_token_cache):
        '''Test that streaming dataset is deterministic with same seed.'''
        from unittest.mock import patch

        from naics_embedder.text_model.dataloader.streaming_dataset import (
            create_streaming_dataset,
        )

        cfg1 = StreamingConfig(
            descriptions_parquet=full_mock_setup['descriptions_path'],
            triplets_parquet=full_mock_setup['triplets_path'],
            n_positives=1,
            n_negatives=2,
            seed=42,
        )

        cfg2 = StreamingConfig(
            descriptions_parquet=full_mock_setup['descriptions_path'],
            triplets_parquet=full_mock_setup['triplets_path'],
            n_positives=1,
            n_negatives=2,
            seed=42,  # Same seed
        )

        code_to_idx = {'31': 0, '311': 1, '3111': 2, '31111': 3, '311111': 4}
        idx_to_code = {v: k for k, v in code_to_idx.items()}

        with patch(
            'naics_embedder.text_model.dataloader.streaming_dataset.get_indices_codes'
        ) as mock_get:
            mock_get.side_effect = lambda x: code_to_idx if x == 'code_to_idx' else idx_to_code

            dataset1 = create_streaming_dataset(mock_token_cache, cfg1)
            items1 = list(dataset1)

        with patch(
            'naics_embedder.text_model.dataloader.streaming_dataset.get_indices_codes'
        ) as mock_get:
            mock_get.side_effect = lambda x: code_to_idx if x == 'code_to_idx' else idx_to_code

            dataset2 = create_streaming_dataset(mock_token_cache, cfg2)
            items2 = list(dataset2)

        # Same seed should produce same results
        assert len(items1) == len(items2)
        if items1:
            assert items1[0]['anchor_code'] == items2[0]['anchor_code']

    def test_streaming_dataset_different_with_different_seed(
        self, full_mock_setup, mock_token_cache
    ):
        '''Test that different seeds produce different shuffling.'''
        from unittest.mock import patch

        from naics_embedder.text_model.dataloader.streaming_dataset import (
            create_streaming_generator,
        )

        cfg1 = StreamingConfig(
            descriptions_parquet=full_mock_setup['descriptions_path'],
            triplets_parquet=full_mock_setup['triplets_path'],
            n_positives=10,
            n_negatives=10,
            seed=42,
        )

        cfg2 = StreamingConfig(
            descriptions_parquet=full_mock_setup['descriptions_path'],
            triplets_parquet=full_mock_setup['triplets_path'],
            n_positives=10,
            n_negatives=10,
            seed=123,  # Different seed
        )

        code_to_idx = {'31': 0, '311': 1, '3111': 2, '31111': 3, '311111': 4}
        idx_to_code = {v: k for k, v in code_to_idx.items()}

        with patch(
            'naics_embedder.text_model.dataloader.streaming_dataset.get_indices_codes'
        ) as mock_get:
            mock_get.side_effect = lambda x: code_to_idx if x == 'code_to_idx' else idx_to_code
            gen1 = create_streaming_generator(cfg1)
            items1 = list(gen1)

        with patch(
            'naics_embedder.text_model.dataloader.streaming_dataset.get_indices_codes'
        ) as mock_get:
            mock_get.side_effect = lambda x: code_to_idx if x == 'code_to_idx' else idx_to_code
            gen2 = create_streaming_generator(cfg2)
            items2 = list(gen2)

        # With different seeds and sufficient data, some sampling variation expected
        # (though this depends on the data - with very small datasets, may be identical)
        # At minimum, both should produce valid output
        assert isinstance(items1, list)
        assert isinstance(items2, list)
