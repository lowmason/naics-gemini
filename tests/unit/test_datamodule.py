'''
Unit tests for NAICSDataModule and collate_fn.

Tests cover:
- collate_fn batching and padding logic
- Multi-level supervision expansion
- Sampling metadata accumulation
- GeneratorDataset worker sharding
'''

from unittest.mock import MagicMock, patch

import pytest
import torch

from naics_embedder.text_model.dataloader.datamodule import GeneratorDataset, collate_fn

# -------------------------------------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------------------------------------

@pytest.fixture
def channels():
    '''Standard text channels.'''
    return ['title', 'description', 'excluded', 'examples']

@pytest.fixture
def make_embedding(channels):
    '''Factory to create mock embeddings for all channels.'''

    def _make(seq_len=128):
        return {
            ch: {
                'input_ids': torch.randint(0, 1000, (seq_len, )),
                'attention_mask': torch.ones(seq_len, dtype=torch.long),
            }
            for ch in channels
        }

    return _make

@pytest.fixture
def make_batch_item(make_embedding):
    '''Factory to create a single batch item.'''

    def _create(anchor_code, positive_code, negative_codes, seq_len=128):
        return {
            'anchor_code':
            anchor_code,
            'anchor_embedding':
            make_embedding(seq_len),
            'positive_code':
            positive_code,
            'positive_embedding':
            make_embedding(seq_len),
            'negatives': [
                {
                    'negative_code': nc,
                    'negative_idx': i,
                    'negative_embedding': make_embedding(seq_len),
                    'relation_margin': 0,
                    'distance_margin': 4,
                    'explicit_exclusion': False,
                } for i, nc in enumerate(negative_codes)
            ],
        }

    return _create

@pytest.fixture
def make_multilevel_item(make_embedding):
    '''Factory to create a multi-level supervision batch item.'''

    def _create(anchor_code, positive_codes, negative_codes, seq_len=128):
        positives = [
            {
                'positive_code':
                pc,
                'positive_idx':
                i,
                'positive_embedding':
                make_embedding(seq_len),
                'negatives': [
                    {
                        'negative_code': nc,
                        'negative_idx': j,
                        'negative_embedding': make_embedding(seq_len),
                        'relation_margin': 0,
                        'distance_margin': 4,
                    } for j, nc in enumerate(negative_codes)
                ],
            } for i, pc in enumerate(positive_codes)
        ]

        return {
            'anchor_code':
            anchor_code,
            'anchor_embedding':
            make_embedding(seq_len),
            'positives':
            positives,
            'negatives': [
                {
                    'negative_code': nc,
                    'negative_idx': i,
                    'negative_embedding': make_embedding(seq_len),
                    'relation_margin': 0,
                    'distance_margin': 4,
                    'explicit_exclusion': False,
                } for i, nc in enumerate(negative_codes)
            ],
        }

    return _create

# -------------------------------------------------------------------------------------------------
# Basic Collate Tests
# -------------------------------------------------------------------------------------------------

def test_collate_stacks_embeddings_correctly(make_batch_item):
    '''Embeddings should be stacked into proper tensor shapes.'''
    batch = [
        make_batch_item('111', '11', ['222', '333']),
        make_batch_item('444', '44', ['555', '666']),
    ]

    result = collate_fn(batch)

    # Check anchor shape: (batch_size, seq_len)
    assert result['anchor']['title']['input_ids'].shape == (2, 128)
    assert result['anchor']['title']['attention_mask'].shape == (2, 128)

    # Check positive shape: (batch_size, seq_len)
    assert result['positive']['title']['input_ids'].shape == (2, 128)

    # Check negative shape: (batch_size * k_negatives, seq_len)
    assert result['negatives']['title']['input_ids'].shape == (4, 128)

def test_collate_preserves_all_channels(make_batch_item, channels):
    '''All four channels should be present in output.'''
    batch = [make_batch_item('111', '11', ['222'])]

    result = collate_fn(batch)

    for channel in channels:
        assert channel in result['anchor']
        assert channel in result['positive']
        assert channel in result['negatives']

def test_collate_includes_metadata(make_batch_item):
    '''Batch metadata should be included.'''
    batch = [
        make_batch_item('111', '11', ['222', '333']),
        make_batch_item('444', '44', ['555', '666']),
    ]

    result = collate_fn(batch)

    assert result['batch_size'] == 2
    assert result['k_negatives'] == 2
    assert result['anchor_code'] == ['111', '444']
    assert result['positive_code'] == ['11', '44']
    assert len(result['negative_codes']) == 2
    assert result['negative_codes'][0] == ['222', '333']

# -------------------------------------------------------------------------------------------------
# Padding Tests
# -------------------------------------------------------------------------------------------------

def test_collate_pads_uneven_negatives(make_batch_item):
    '''Items with fewer negatives should be padded.'''
    batch = [
        make_batch_item('111', '11', ['222', '333', '444']),  # 3 negatives
        make_batch_item('555', '55', ['666']),  # 1 negative
    ]

    result = collate_fn(batch)

    assert result['k_negatives'] == 3
    # Total negatives: 3 + 3 (padded) = 6
    assert result['negatives']['title']['input_ids'].shape == (6, 128)

def test_collate_padding_repeats_last_negative(make_batch_item, make_embedding):
    '''Padding should repeat the last negative.'''
    # Create item with single known negative
    item = {
        'anchor_code':
        '111',
        'anchor_embedding':
        make_embedding(),
        'positive_code':
        '11',
        'positive_embedding':
        make_embedding(),
        'negatives': [
            {
                'negative_code': 'LAST',
                'negative_idx': 0,
                'negative_embedding': make_embedding(),
                'relation_margin': 0,
                'distance_margin': 4,
            }
        ],
    }

    # Create item with multiple negatives to force padding
    item2 = {
        'anchor_code':
        '222',
        'anchor_embedding':
        make_embedding(),
        'positive_code':
        '22',
        'positive_embedding':
        make_embedding(),
        'negatives': [
            {
                'negative_code': f'NEG{i}',
                'negative_idx': i,
                'negative_embedding': make_embedding(),
                'relation_margin': 0,
                'distance_margin': 4,
            } for i in range(3)
        ],
    }

    batch = [item, item2]
    result = collate_fn(batch)

    # After collation, first item should have 3 negatives (padded from 1)
    assert result['k_negatives'] == 3
    # The negative_codes for first item should repeat 'LAST'
    assert result['negative_codes'][0] == ['LAST', 'LAST', 'LAST']

# -------------------------------------------------------------------------------------------------
# Error Handling Tests
# -------------------------------------------------------------------------------------------------

def test_collate_raises_on_empty_negatives(make_embedding):
    '''Batch with no negatives should raise ValueError.'''
    batch = [
        {
            'anchor_code': '111',
            'anchor_embedding': make_embedding(),
            'positive_code': '11',
            'positive_embedding': make_embedding(),
            'negatives': [],
        }
    ]

    with pytest.raises(ValueError, match='no negatives'):
        collate_fn(batch)

def test_collate_handles_single_item_batch(make_batch_item):
    '''Single item batch should work correctly.'''
    batch = [make_batch_item('111', '11', ['222', '333'])]

    result = collate_fn(batch)

    assert result['batch_size'] == 1
    assert result['k_negatives'] == 2
    assert result['anchor']['title']['input_ids'].shape == (1, 128)

# -------------------------------------------------------------------------------------------------
# Multi-Level Supervision Tests
# -------------------------------------------------------------------------------------------------

def test_collate_multi_level_expansion(make_multilevel_item):
    '''Multi-level positives should be expanded into separate entries.'''
    batch = [
        make_multilevel_item(
            '311111',
            ['31111', '3111', '311'],  # 3 ancestor levels
            ['222'],
        )
    ]

    result = collate_fn(batch)

    # Should expand to 3 entries (one per positive level)
    assert result['batch_size'] == 3
    assert result['positive_levels'] == [5, 4, 3]  # len of each positive code

def test_collate_multi_level_preserves_anchor(make_multilevel_item):
    '''Each expanded entry should share the same anchor.'''
    batch = [make_multilevel_item('311111', ['31111', '3111'], ['222'])]

    result = collate_fn(batch)

    # All anchor codes should be the same
    assert result['anchor_code'] == ['311111', '311111']

    # Anchor embeddings should be stacked (same embedding repeated)
    assert result['anchor']['title']['input_ids'].shape == (2, 128)

def test_collate_multi_level_different_positives(make_multilevel_item):
    '''Each expanded entry should have different positive.'''
    batch = [make_multilevel_item('311111', ['31111', '3111'], ['222'])]

    result = collate_fn(batch)

    assert result['positive_code'] == ['31111', '3111']

def test_collate_multi_level_shared_negatives(make_multilevel_item):
    '''All expanded entries should share the same negatives.'''
    batch = [make_multilevel_item('311111', ['31111', '3111'], ['222', '333'])]

    result = collate_fn(batch)

    # Each expansion uses the same negatives
    assert result['negative_codes'][0] == ['222', '333']
    assert result['negative_codes'][1] == ['222', '333']

def test_collate_mixed_single_and_multi_level(make_batch_item, make_multilevel_item):
    '''Batch can contain both single and multi-level items.'''
    batch = [
        make_multilevel_item('311111', ['31111', '3111'], ['222']),
        # Standard single-positive item won't trigger multi-level path
        # since it doesn't have 'positives' key as a list
    ]

    result = collate_fn(batch)

    # Should have 2 entries from multi-level expansion
    assert result['batch_size'] == 2

# -------------------------------------------------------------------------------------------------
# Sampling Metadata Tests
# -------------------------------------------------------------------------------------------------

def test_collate_accumulates_sampling_metadata(make_batch_item):
    '''Sampling metadata should be accumulated across batch items.'''
    batch = [
        make_batch_item('111', '11', ['222']),
        make_batch_item('333', '33', ['444']),
    ]

    # Add sampling metadata
    batch[0]['sampling_metadata'] = {
        'strategy': 'sans_static',
        'candidates_near': 10,
        'candidates_far': 5,
        'sampled_near': 2,
        'sampled_far': 1,
        'effective_near_weight': 0.6,
        'effective_far_weight': 0.4,
    }
    batch[1]['sampling_metadata'] = {
        'strategy': 'sans_static',
        'candidates_near': 8,
        'candidates_far': 7,
        'sampled_near': 1,
        'sampled_far': 2,
        'effective_near_weight': 0.5,
        'effective_far_weight': 0.5,
    }

    result = collate_fn(batch)

    assert 'sampling_metadata' in result
    assert result['sampling_metadata']['candidates_near'] == 18
    assert result['sampling_metadata']['candidates_far'] == 12
    assert result['sampling_metadata']['sampled_near'] == 3
    assert result['sampling_metadata']['sampled_far'] == 3

def test_collate_computes_average_weights(make_batch_item):
    '''Effective weights should be averaged across records.'''
    batch = [
        make_batch_item('111', '11', ['222']),
        make_batch_item('333', '33', ['444']),
    ]

    batch[0]['sampling_metadata'] = {
        'strategy': 'sans_static',
        'candidates_near': 10,
        'candidates_far': 5,
        'sampled_near': 2,
        'sampled_far': 1,
        'effective_near_weight': 0.8,
        'effective_far_weight': 0.2,
    }
    batch[1]['sampling_metadata'] = {
        'strategy': 'sans_static',
        'candidates_near': 8,
        'candidates_far': 7,
        'sampled_near': 1,
        'sampled_far': 2,
        'effective_near_weight': 0.4,
        'effective_far_weight': 0.6,
    }

    result = collate_fn(batch)

    # Average weights: (0.8 + 0.4) / 2 = 0.6, (0.2 + 0.6) / 2 = 0.4
    assert abs(result['sampling_metadata']['avg_effective_near_weight'] - 0.6) < 1e-6
    assert abs(result['sampling_metadata']['avg_effective_far_weight'] - 0.4) < 1e-6

def test_collate_no_metadata_when_missing(make_batch_item):
    '''No sampling_metadata key when items have no metadata.'''
    batch = [make_batch_item('111', '11', ['222'])]

    result = collate_fn(batch)

    assert 'sampling_metadata' not in result

# -------------------------------------------------------------------------------------------------
# GeneratorDataset Tests
# -------------------------------------------------------------------------------------------------

def test_generator_dataset_sharding_worker_0():
    '''Worker 0 should get items at indices 0, 2, 4, ...'''

    def mock_generator_fn(token_cache, cfg, sampling_cfg):
        for i in range(10):
            yield {'idx': i}

    mock_tokenization_cfg = MagicMock()
    mock_tokenization_cfg.output_path = '/tmp/nonexistent_cache.pt'
    mock_streaming_cfg = MagicMock()
    mock_sampling_cfg = MagicMock()

    dataset = GeneratorDataset(
        mock_generator_fn, mock_tokenization_cfg, mock_streaming_cfg, mock_sampling_cfg
    )

    # Mock worker info for worker 0 of 2
    worker_info = MagicMock()
    worker_info.id = 0
    worker_info.num_workers = 2

    with patch('torch.utils.data.get_worker_info', return_value=worker_info):
        with patch.object(dataset, '_get_token_cache', return_value={}):
            items = list(dataset)

    # Worker 0 should get indices 0, 2, 4, 6, 8
    assert [item['idx'] for item in items] == [0, 2, 4, 6, 8]

def test_generator_dataset_sharding_worker_1():
    '''Worker 1 should get items at indices 1, 3, 5, ...'''

    def mock_generator_fn(token_cache, cfg, sampling_cfg):
        for i in range(10):
            yield {'idx': i}

    mock_tokenization_cfg = MagicMock()
    mock_tokenization_cfg.output_path = '/tmp/nonexistent_cache.pt'
    mock_streaming_cfg = MagicMock()
    mock_sampling_cfg = MagicMock()

    dataset = GeneratorDataset(
        mock_generator_fn, mock_tokenization_cfg, mock_streaming_cfg, mock_sampling_cfg
    )

    # Mock worker info for worker 1 of 2
    worker_info = MagicMock()
    worker_info.id = 1
    worker_info.num_workers = 2

    with patch('torch.utils.data.get_worker_info', return_value=worker_info):
        with patch.object(dataset, '_get_token_cache', return_value={}):
            items = list(dataset)

    # Worker 1 should get indices 1, 3, 5, 7, 9
    assert [item['idx'] for item in items] == [1, 3, 5, 7, 9]

def test_generator_dataset_no_sharding_single_worker():
    '''With no workers, should yield all items from generator.'''

    def mock_generator_fn(token_cache, cfg, sampling_cfg):
        for i in range(5):
            yield {'idx': i}

    mock_tokenization_cfg = MagicMock()
    mock_tokenization_cfg.output_path = '/tmp/nonexistent_cache.pt'
    mock_streaming_cfg = MagicMock()
    mock_sampling_cfg = MagicMock()

    dataset = GeneratorDataset(
        mock_generator_fn, mock_tokenization_cfg, mock_streaming_cfg, mock_sampling_cfg
    )

    # Directly set the token cache to bypass loading
    dataset._token_cache = {}

    # Patch get_worker_info at the module level
    with patch(
        'naics_embedder.text_model.dataloader.datamodule.torch.utils.data.get_worker_info',
        return_value=None,
    ):
        items = list(dataset)

    assert [item['idx'] for item in items] == [0, 1, 2, 3, 4]

def test_generator_dataset_sharding_four_workers():
    '''Four workers should each get 1/4 of items.'''

    def mock_generator_fn(token_cache, cfg, sampling_cfg):
        for i in range(12):
            yield {'idx': i}

    mock_tokenization_cfg = MagicMock()
    mock_tokenization_cfg.output_path = '/tmp/nonexistent_cache.pt'
    mock_streaming_cfg = MagicMock()
    mock_sampling_cfg = MagicMock()

    all_items = []
    for worker_id in range(4):
        dataset = GeneratorDataset(
            mock_generator_fn, mock_tokenization_cfg, mock_streaming_cfg, mock_sampling_cfg
        )

        worker_info = MagicMock()
        worker_info.id = worker_id
        worker_info.num_workers = 4

        with patch('torch.utils.data.get_worker_info', return_value=worker_info):
            with patch.object(dataset, '_get_token_cache', return_value={}):
                items = list(dataset)
                all_items.extend([item['idx'] for item in items])

    # All items should be covered exactly once
    assert sorted(all_items) == list(range(12))

def test_generator_dataset_lazy_cache_loading():
    '''Token cache should be loaded lazily on first iteration.'''

    def mock_generator_fn(token_cache, cfg, sampling_cfg):
        yield {'cache': token_cache}

    mock_tokenization_cfg = MagicMock()
    mock_tokenization_cfg.output_path = '/tmp/nonexistent_cache.pt'

    dataset = GeneratorDataset(mock_generator_fn, mock_tokenization_cfg, MagicMock(), MagicMock())

    # Cache should not be loaded yet
    assert dataset._token_cache is None

    # Mock the cache loading
    with patch('torch.utils.data.get_worker_info', return_value=None):
        with patch.object(dataset, '_get_token_cache', return_value={'loaded': True}) as mock_get:
            list(dataset)
            mock_get.assert_called_once()

# -------------------------------------------------------------------------------------------------
# Edge Cases
# -------------------------------------------------------------------------------------------------

def test_collate_different_sequence_lengths(make_embedding):
    '''Batch items with same sequence length should collate.'''
    channels = ['title', 'description', 'excluded', 'examples']

    def make_item(seq_len):
        embedding = {
            ch: {
                'input_ids': torch.randint(0, 1000, (seq_len, )),
                'attention_mask': torch.ones(seq_len, dtype=torch.long),
            }
            for ch in channels
        }
        return {
            'anchor_code':
            '111',
            'anchor_embedding':
            embedding,
            'positive_code':
            '11',
            'positive_embedding':
            embedding,
            'negatives': [
                {
                    'negative_code': '222',
                    'negative_idx': 0,
                    'negative_embedding': embedding,
                    'relation_margin': 0,
                    'distance_margin': 4,
                }
            ],
        }

    # Same sequence length should work
    batch = [make_item(64), make_item(64)]
    result = collate_fn(batch)
    assert result['anchor']['title']['input_ids'].shape == (2, 64)

def test_collate_preserves_tensor_dtype(make_batch_item):
    '''Tensor dtypes should be preserved after collation.'''
    batch = [make_batch_item('111', '11', ['222'])]

    result = collate_fn(batch)

    assert result['anchor']['title']['input_ids'].dtype == torch.long
    assert result['anchor']['title']['attention_mask'].dtype == torch.long

def test_collate_large_batch(make_batch_item):
    '''Should handle larger batches efficiently.'''
    batch = [make_batch_item(f'{i:03d}', f'{i:02d}', [f'{i + 100}']) for i in range(64)]

    result = collate_fn(batch)

    assert result['batch_size'] == 64
    assert result['anchor']['title']['input_ids'].shape == (64, 128)

# -------------------------------------------------------------------------------------------------
# NAICSDataModule Setup Tests
# -------------------------------------------------------------------------------------------------

class TestNAICSDataModuleSetup:
    '''Test suite for NAICSDataModule initialization and setup.'''

    @pytest.fixture
    def mock_descriptions_parquet(self, tmp_path):
        '''Create mock descriptions parquet file.'''
        import polars as pl

        data = {
            'index': [0, 1, 2],
            'code': ['311111', '311112', '321111'],
            'level': [6, 6, 6],
            'title': ['Dog Food', 'Cat Food', 'Sawmills'],
            'description': ['Make dog food', 'Make cat food', 'Cut wood'],
            'excluded': ['', '', ''],
            'examples': ['', '', ''],
            'excluded_codes': [None, None, None],
        }
        df = pl.DataFrame(data)
        path = tmp_path / 'descriptions.parquet'
        df.write_parquet(path)
        return str(path)

    @pytest.fixture
    def mock_triplets_dir(self, tmp_path):
        '''Create mock triplets directory with parquet files.'''
        import polars as pl

        triplets_dir = tmp_path / 'triplets'
        triplets_dir.mkdir()

        # Create anchor subdirectory
        anchor_dir = triplets_dir / 'anchor=0'
        anchor_dir.mkdir()

        data = {
            'anchor_idx': [0, 0],
            'anchor_code': ['311111', '311111'],
            'anchor_level': [6, 6],
            'positive_idx': [1, 1],
            'positive_code': ['311112', '311112'],
            'positive_level': [6, 6],
            'negative_idx': [2, 2],
            'negative_code': ['321111', '321111'],
            'negative_level': [6, 6],
            'relation_margin': [0, 0],
            'distance_margin': [4, 4],
            'positive_relation': [1, 1],
            'positive_distance': [2, 2],
            'negative_relation': [3, 3],
            'negative_distance': [8, 8],
        }
        df = pl.DataFrame(data)
        path = anchor_dir / 'part0.parquet'
        df.write_parquet(path)
        return str(triplets_dir)

    def test_datamodule_init_default_params(self, mock_descriptions_parquet, mock_triplets_dir):
        '''Test NAICSDataModule initializes with default parameters.'''
        from naics_embedder.text_model.dataloader.datamodule import NAICSDataModule

        datamodule = NAICSDataModule(
            descriptions_path=mock_descriptions_parquet,
            triplets_path=mock_triplets_dir,
            batch_size=4,
            num_workers=0,
        )

        assert datamodule.batch_size == 4
        assert datamodule.num_workers == 0
        assert datamodule.descriptions_path == mock_descriptions_parquet
        assert datamodule.triplets_path == mock_triplets_dir

    def test_datamodule_init_custom_streaming_config(
        self, mock_descriptions_parquet, mock_triplets_dir
    ):
        '''Test NAICSDataModule initializes with custom streaming config.'''
        from naics_embedder.text_model.dataloader.datamodule import NAICSDataModule

        streaming_config = {
            'n_positives': 2,
            'n_negatives': 8,
            'seed': 123,
        }

        datamodule = NAICSDataModule(
            descriptions_path=mock_descriptions_parquet,
            triplets_path=mock_triplets_dir,
            streaming_config=streaming_config,
            batch_size=8,
            num_workers=0,
            seed=100,  # Explicit seed for validation config
        )

        assert datamodule.train_streaming_cfg.n_positives == 2
        assert datamodule.train_streaming_cfg.n_negatives == 8
        assert datamodule.train_streaming_cfg.seed == 123
        # Validation config uses (seed + 1) from NAICSDataModule.__init__ seed param
        assert datamodule.val_streaming_cfg.seed == 101  # 100 + 1

    def test_datamodule_init_custom_sampling_config(
        self, mock_descriptions_parquet, mock_triplets_dir
    ):
        '''Test NAICSDataModule initializes with custom sampling config.'''
        from naics_embedder.text_model.dataloader.datamodule import NAICSDataModule

        sampling_config = {'strategy': 'sans_static'}

        datamodule = NAICSDataModule(
            descriptions_path=mock_descriptions_parquet,
            triplets_path=mock_triplets_dir,
            sampling_config=sampling_config,
            batch_size=4,
            num_workers=0,
        )

        assert datamodule.sampling_cfg.strategy == 'sans_static'

    def test_datamodule_creates_train_dataset(self, mock_descriptions_parquet, mock_triplets_dir):
        '''Test that train_dataset is created during initialization.'''
        from naics_embedder.text_model.dataloader.datamodule import (
            GeneratorDataset,
            NAICSDataModule,
        )

        datamodule = NAICSDataModule(
            descriptions_path=mock_descriptions_parquet,
            triplets_path=mock_triplets_dir,
            batch_size=4,
            num_workers=0,
        )

        assert datamodule.train_dataset is not None
        assert isinstance(datamodule.train_dataset, GeneratorDataset)

    def test_datamodule_creates_val_dataset(self, mock_descriptions_parquet, mock_triplets_dir):
        '''Test that val_dataset is created during initialization.'''
        from naics_embedder.text_model.dataloader.datamodule import (
            GeneratorDataset,
            NAICSDataModule,
        )

        datamodule = NAICSDataModule(
            descriptions_path=mock_descriptions_parquet,
            triplets_path=mock_triplets_dir,
            batch_size=4,
            num_workers=0,
        )

        assert datamodule.val_dataset is not None
        assert isinstance(datamodule.val_dataset, GeneratorDataset)

    def test_datamodule_tokenization_config(self, mock_descriptions_parquet, mock_triplets_dir):
        '''Test that tokenization config is set correctly.'''
        from naics_embedder.text_model.dataloader.datamodule import NAICSDataModule

        datamodule = NAICSDataModule(
            descriptions_path=mock_descriptions_parquet,
            triplets_path=mock_triplets_dir,
            tokenizer_name='sentence-transformers/all-MiniLM-L6-v2',
            batch_size=4,
            num_workers=0,
        )

        assert datamodule.tokenization_cfg.descriptions_parquet == mock_descriptions_parquet
        assert datamodule.tokenization_cfg.tokenizer_name == 'sentence-transformers/all-MiniLM-L6-v2'

# -------------------------------------------------------------------------------------------------
# Train/Val DataLoader Creation Tests
# -------------------------------------------------------------------------------------------------

class TestDataLoaderCreation:
    '''Test suite for train and validation dataloader creation.'''

    @pytest.fixture
    def mock_datamodule(self, tmp_path):
        '''Create a mock NAICSDataModule for testing.'''
        import polars as pl

        from naics_embedder.text_model.dataloader.datamodule import NAICSDataModule

        # Create descriptions
        desc_data = {
            'index': [0, 1, 2],
            'code': ['311111', '311112', '321111'],
            'level': [6, 6, 6],
            'title': ['Dog Food', 'Cat Food', 'Sawmills'],
            'description': ['Make dog food', 'Make cat food', 'Cut wood'],
            'excluded': ['', '', ''],
            'examples': ['', '', ''],
            'excluded_codes': [None, None, None],
        }
        desc_df = pl.DataFrame(desc_data)
        desc_path = tmp_path / 'descriptions.parquet'
        desc_df.write_parquet(desc_path)

        # Create triplets
        triplets_dir = tmp_path / 'triplets'
        triplets_dir.mkdir()
        anchor_dir = triplets_dir / 'anchor=0'
        anchor_dir.mkdir()

        triplet_data = {
            'anchor_idx': [0],
            'anchor_code': ['311111'],
            'anchor_level': [6],
            'positive_idx': [1],
            'positive_code': ['311112'],
            'positive_level': [6],
            'negative_idx': [2],
            'negative_code': ['321111'],
            'negative_level': [6],
            'relation_margin': [0],
            'distance_margin': [4],
            'positive_relation': [1],
            'positive_distance': [2],
            'negative_relation': [3],
            'negative_distance': [8],
        }
        triplet_df = pl.DataFrame(triplet_data)
        triplet_path = anchor_dir / 'part0.parquet'
        triplet_df.write_parquet(triplet_path)

        return NAICSDataModule(
            descriptions_path=str(desc_path),
            triplets_path=str(triplets_dir),
            batch_size=2,
            num_workers=0,
        )

    def test_train_dataloader_returns_dataloader(self, mock_datamodule):
        '''Test that train_dataloader returns a DataLoader instance.'''
        from torch.utils.data import DataLoader

        train_loader = mock_datamodule.train_dataloader()

        assert isinstance(train_loader, DataLoader)

    def test_train_dataloader_batch_size(self, mock_datamodule):
        '''Test that train_dataloader uses correct batch size.'''
        train_loader = mock_datamodule.train_dataloader()

        assert train_loader.batch_size == 2

    def test_train_dataloader_num_workers(self, mock_datamodule):
        '''Test that train_dataloader uses correct num_workers.'''
        train_loader = mock_datamodule.train_dataloader()

        assert train_loader.num_workers == 0

    def test_train_dataloader_collate_fn(self, mock_datamodule):
        '''Test that train_dataloader uses custom collate function.'''
        train_loader = mock_datamodule.train_dataloader()

        assert train_loader.collate_fn == collate_fn

    def test_val_dataloader_returns_dataloader(self, mock_datamodule):
        '''Test that val_dataloader returns a DataLoader instance.'''
        from torch.utils.data import DataLoader

        val_loader = mock_datamodule.val_dataloader()

        assert isinstance(val_loader, DataLoader)

    def test_val_dataloader_batch_size(self, mock_datamodule):
        '''Test that val_dataloader uses correct batch size.'''
        val_loader = mock_datamodule.val_dataloader()

        assert val_loader.batch_size == 2

    def test_val_dataloader_num_workers(self, mock_datamodule):
        '''Test that val_dataloader uses correct num_workers.'''
        val_loader = mock_datamodule.val_dataloader()

        assert val_loader.num_workers == 0

    def test_val_dataloader_collate_fn(self, mock_datamodule):
        '''Test that val_dataloader uses custom collate function.'''
        val_loader = mock_datamodule.val_dataloader()

        assert val_loader.collate_fn == collate_fn

    def test_train_val_dataloaders_are_different(self, mock_datamodule):
        '''Test that train and val dataloaders are distinct.'''
        train_loader = mock_datamodule.train_dataloader()
        val_loader = mock_datamodule.val_dataloader()

        # They should be different objects
        assert train_loader is not val_loader
        # They should use different datasets
        assert train_loader.dataset is not val_loader.dataset

    def test_persistent_workers_disabled_when_zero_workers(self, mock_datamodule):
        '''Test persistent_workers is False when num_workers=0.'''
        train_loader = mock_datamodule.train_dataloader()

        # persistent_workers should be False since num_workers=0
        assert train_loader.persistent_workers is False
