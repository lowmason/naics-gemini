'''
Unit tests for tokenization cache module.

Tests cover:
- Tokenization cache building
- Cache saving and loading
- File locking for multi-worker safety
- Cache path resolution
- get_tokens utility function
'''

from pathlib import Path
from unittest.mock import patch

import polars as pl
import pytest
import torch

from naics_embedder.text_model.dataloader.tokenization_cache import (
    _acquire_lock,
    _build_tokenization_cache,
    _load_tokenization_cache,
    _release_lock,
    _save_tokenization_cache,
    get_tokens,
    tokenization_cache,
)
from naics_embedder.utils.config import TokenizationConfig

# -------------------------------------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------------------------------------

@pytest.fixture
def sample_descriptions_parquet(tmp_path):
    '''Create sample descriptions parquet file.'''
    data = {
        'index': [0, 1, 2],
        'code': ['311111', '311112', '321111'],
        'title': ['Dog Food Manufacturing', 'Cat Food Manufacturing', 'Sawmills'],
        'description': [
            'Manufacture dog food',
            'Manufacture cat food',
            'Saw logs into lumber',
        ],
        'excluded': ['', '', ''],
        'examples': ['', '', ''],
    }
    df = pl.DataFrame(data)
    path = tmp_path / 'descriptions.parquet'
    df.write_parquet(path)
    return str(path)

@pytest.fixture
def sample_tokenization_cache():
    '''Create sample tokenization cache.'''
    cache = {
        0: {
            'code': '311111',
            'title': {
                'input_ids': torch.randint(0, 1000, (24, )),
                'attention_mask': torch.ones(24, dtype=torch.long),
            },
            'description': {
                'input_ids': torch.randint(0, 1000, (128, )),
                'attention_mask': torch.ones(128, dtype=torch.long),
            },
            'excluded': {
                'input_ids': torch.randint(0, 1000, (128, )),
                'attention_mask': torch.ones(128, dtype=torch.long),
            },
            'examples': {
                'input_ids': torch.randint(0, 1000, (128, )),
                'attention_mask': torch.ones(128, dtype=torch.long),
            },
        },
        1: {
            'code': '311112',
            'title': {
                'input_ids': torch.randint(0, 1000, (24, )),
                'attention_mask': torch.ones(24, dtype=torch.long),
            },
            'description': {
                'input_ids': torch.randint(0, 1000, (128, )),
                'attention_mask': torch.ones(128, dtype=torch.long),
            },
            'excluded': {
                'input_ids': torch.randint(0, 1000, (128, )),
                'attention_mask': torch.ones(128, dtype=torch.long),
            },
            'examples': {
                'input_ids': torch.randint(0, 1000, (128, )),
                'attention_mask': torch.ones(128, dtype=torch.long),
            },
        },
    }
    return cache

@pytest.fixture
def tokenization_config(sample_descriptions_parquet, tmp_path):
    '''Create tokenization config for testing.'''
    cache_path = tmp_path / 'token_cache' / 'token_cache.pt'
    return TokenizationConfig(
        descriptions_parquet=sample_descriptions_parquet,
        tokenizer_name='sentence-transformers/all-MiniLM-L6-v2',
        max_length=128,
        output_path=str(cache_path),
    )

# -------------------------------------------------------------------------------------------------
# Tokenization Cache Building Tests
# -------------------------------------------------------------------------------------------------

@pytest.mark.unit
class TestBuildTokenizationCache:
    '''Test suite for tokenization cache building.'''

    def test_build_tokenization_cache(self, sample_descriptions_parquet):
        '''Test building tokenization cache from descriptions.'''
        cache = _build_tokenization_cache(
            sample_descriptions_parquet, 'sentence-transformers/all-MiniLM-L6-v2', 128
        )

        assert isinstance(cache, dict)
        assert len(cache) == 3  # Three items in sample data
        assert 0 in cache
        assert 1 in cache
        assert 2 in cache

        # Check structure
        item = cache[0]
        assert 'code' in item
        assert 'title' in item
        assert 'description' in item
        assert 'excluded' in item
        assert 'examples' in item

        # Check tokenization structure
        title_dict = item['title']  # type: ignore
        desc_dict = item['description']  # type: ignore
        assert 'input_ids' in title_dict
        assert 'attention_mask' in title_dict
        title_input_ids = title_dict['input_ids']  # type: ignore
        desc_input_ids = desc_dict['input_ids']  # type: ignore
        assert title_input_ids.shape == (24, )
        assert desc_input_ids.shape == (128, )

    def test_build_tokenization_cache_empty_text(self, tmp_path):
        '''Test building cache with empty text fields.'''
        # Create data with empty fields
        data = {
            'index': [0],
            'code': ['311111'],
            'title': [''],
            'description': [''],
            'excluded': [''],
            'examples': [''],
        }
        df = pl.DataFrame(data)
        path = tmp_path / 'empty_descriptions.parquet'
        df.write_parquet(path)

        cache = _build_tokenization_cache(str(path), 'sentence-transformers/all-MiniLM-L6-v2', 128)

        assert len(cache) == 1
        # Empty text should be replaced with [EMPTY]
        assert cache[0]['code'] == '311111'

    def test_build_tokenization_cache_max_length(self, sample_descriptions_parquet):
        '''Test cache building respects max_length.'''
        cache = _build_tokenization_cache(
            sample_descriptions_parquet,
            'sentence-transformers/all-MiniLM-L6-v2',
            64,  # Shorter max_length
        )

        desc_dict = cache[0]['description']  # type: ignore
        desc_input_ids = desc_dict['input_ids']  # type: ignore
        desc_attention_mask = desc_dict['attention_mask']  # type: ignore
        assert desc_input_ids.shape == (64, )
        assert desc_attention_mask.shape == (64, )

# -------------------------------------------------------------------------------------------------
# Cache Save/Load Tests
# -------------------------------------------------------------------------------------------------

@pytest.mark.unit
class TestCacheSaveLoad:
    '''Test suite for cache saving and loading.'''

    def test_save_tokenization_cache(self, sample_tokenization_cache, tmp_path):
        '''Test saving tokenization cache to disk.'''
        cache_path = tmp_path / 'test_cache.pt'
        result_path = _save_tokenization_cache(sample_tokenization_cache, str(cache_path))

        assert result_path == cache_path
        assert cache_path.exists()

        # Verify file can be loaded
        loaded = torch.load(cache_path, weights_only=True, map_location='cpu')
        assert isinstance(loaded, dict)
        assert len(loaded) == len(sample_tokenization_cache)

    def test_save_tokenization_cache_creates_directory(self, sample_tokenization_cache, tmp_path):
        '''Test that save creates parent directories.'''
        cache_path = tmp_path / 'nested' / 'dir' / 'cache.pt'
        _save_tokenization_cache(sample_tokenization_cache, str(cache_path))

        assert cache_path.exists()
        assert cache_path.parent.exists()

    def test_load_tokenization_cache_exists(self, sample_tokenization_cache, tmp_path):
        '''Test loading existing cache.'''
        cache_path = tmp_path / 'test_cache.pt'
        torch.save(sample_tokenization_cache, cache_path)

        loaded = _load_tokenization_cache(str(cache_path), verbose=True)

        assert loaded is not None
        assert len(loaded) == len(sample_tokenization_cache)
        assert 0 in loaded
        assert loaded[0]['code'] == sample_tokenization_cache[0]['code']

    def test_load_tokenization_cache_not_exists(self, tmp_path):
        '''Test loading non-existent cache.'''
        cache_path = tmp_path / 'nonexistent.pt'
        loaded = _load_tokenization_cache(str(cache_path), verbose=False)

        assert loaded is None

    def test_load_tokenization_cache_corrupted(self, tmp_path):
        '''Test loading corrupted cache file.'''
        cache_path = tmp_path / 'corrupted.pt'
        # Write invalid data
        cache_path.write_text('invalid data')

        with pytest.raises(Exception):
            _load_tokenization_cache(str(cache_path), verbose=True)

# -------------------------------------------------------------------------------------------------
# File Locking Tests
# -------------------------------------------------------------------------------------------------

@pytest.mark.unit
class TestFileLocking:
    '''Test suite for file locking utilities.'''

    def test_acquire_lock(self, tmp_path):
        '''Test acquiring a lock.'''
        lock_path = tmp_path / 'test.lock'
        lock_file = _acquire_lock(lock_path, timeout=5)

        assert lock_file is not None
        assert lock_path.exists()

        # Clean up
        _release_lock(lock_file)

    def test_acquire_lock_timeout(self, tmp_path):
        '''Test lock acquisition timeout.'''
        lock_path = tmp_path / 'test.lock'

        # Acquire lock in another "process" (simulated by not releasing)
        lock_file1 = _acquire_lock(lock_path, timeout=1)
        assert lock_file1 is not None

        # Try to acquire again (should timeout)
        lock_file2 = _acquire_lock(lock_path, timeout=1)
        assert lock_file2 is None

        # Clean up
        _release_lock(lock_file1)

    def test_release_lock(self, tmp_path):
        '''Test releasing a lock.'''
        lock_path = tmp_path / 'test.lock'
        lock_file = _acquire_lock(lock_path, timeout=5)
        assert lock_file is not None

        _release_lock(lock_file)
        # Lock should be released (can acquire again)
        lock_file2 = _acquire_lock(lock_path, timeout=1)
        assert lock_file2 is not None
        _release_lock(lock_file2)

    def test_release_lock_none(self):
        '''Test releasing None lock (should not error).'''
        _release_lock(None)  # Should not raise

# -------------------------------------------------------------------------------------------------
# Main Tokenization Cache Function Tests
# -------------------------------------------------------------------------------------------------

@pytest.mark.unit
class TestTokenizationCache:
    '''Test suite for main tokenization_cache function.'''

    def test_tokenization_cache_loads_existing(
        self, tokenization_config, sample_tokenization_cache
    ):
        '''Test loading existing cache.'''
        # Create cache file
        cache_path = Path(tokenization_config.output_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(sample_tokenization_cache, cache_path)

        # Load cache
        cache = tokenization_cache(tokenization_config, use_locking=False)

        assert cache is not None
        assert len(cache) == len(sample_tokenization_cache)

    def test_tokenization_cache_builds_new(self, tokenization_config):
        '''Test building new cache when it doesn't exist.'''
        cache_path = Path(tokenization_config.output_path)
        if cache_path.exists():
            cache_path.unlink()

        # Build cache
        cache = tokenization_cache(tokenization_config, use_locking=True)

        assert cache is not None
        assert len(cache) == 3  # From sample_descriptions_parquet
        assert cache_path.exists()

    def test_tokenization_cache_no_locking_fails_when_missing(self, tokenization_config):
        '''Test that disabling locking fails when cache doesn't exist.'''
        cache_path = Path(tokenization_config.output_path)
        if cache_path.exists():
            cache_path.unlink()

        with pytest.raises(RuntimeError, match='Tokenization cache not found'):
            tokenization_cache(tokenization_config, use_locking=False)

    @patch('naics_embedder.text_model.dataloader.tokenization_cache._acquire_lock')
    @patch('naics_embedder.text_model.dataloader.tokenization_cache._load_tokenization_cache')
    def test_tokenization_cache_wait_for_other_worker(
        self, mock_load, mock_acquire, tokenization_config, sample_tokenization_cache
    ):
        '''Test waiting for another worker to build cache.'''
        cache_path = Path(tokenization_config.output_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        # Simulate another worker building cache
        mock_acquire.return_value = None  # Can't acquire lock
        # First call returns None (cache not ready), second returns cache
        mock_load.side_effect = [None, sample_tokenization_cache]

        # Use a short timeout for testing
        with patch('naics_embedder.text_model.dataloader.tokenization_cache.time.sleep'):
            cache = tokenization_cache(tokenization_config, use_locking=True)

        assert cache is not None
        assert mock_load.call_count >= 2  # Should retry loading

    def test_tokenization_cache_atomic_save(self, tokenization_config):
        '''Test that cache is saved atomically (via temp file + rename).'''
        cache_path = Path(tokenization_config.output_path)
        if cache_path.exists():
            cache_path.unlink()

        # Build cache
        tokenization_cache(tokenization_config, use_locking=True)

        assert cache_path.exists()
        # Verify no .tmp file remains
        tmp_files = list(cache_path.parent.glob('*.tmp'))
        assert len(tmp_files) == 0

# -------------------------------------------------------------------------------------------------
# Get Tokens Utility Tests
# -------------------------------------------------------------------------------------------------

@pytest.mark.unit
class TestGetTokens:
    '''Test suite for get_tokens utility function.'''

    def test_get_tokens_by_index(self, sample_tokenization_cache):
        '''Test getting tokens by index.'''
        result = get_tokens(0, sample_tokenization_cache)

        assert isinstance(result, dict)
        assert 0 in result
        assert result[0]['code'] == '311111'

    def test_get_tokens_by_code(self, sample_tokenization_cache):
        '''Test getting tokens by code string.'''
        result = get_tokens('311112', sample_tokenization_cache)

        assert isinstance(result, dict)
        assert 1 in result
        assert result[1]['code'] == '311112'

    def test_get_tokens_invalid_type(self, sample_tokenization_cache):
        '''Test error handling for invalid input type.'''
        with pytest.raises(ValueError, match='idx_code must be an int or str'):
            # Use a type that's not int or str - use a list
            get_tokens([0], sample_tokenization_cache)  # type: ignore

    def test_get_tokens_code_not_found(self, sample_tokenization_cache):
        '''Test behavior when code is not found.'''
        # Should raise KeyError when code not found
        with pytest.raises(KeyError):
            get_tokens('999999', sample_tokenization_cache)

# -------------------------------------------------------------------------------------------------
# Cache Invalidation Tests
# -------------------------------------------------------------------------------------------------

@pytest.mark.unit
class TestCacheInvalidation:
    '''Test suite for cache invalidation scenarios.

    Tests that the cache is properly rebuilt when underlying data changes.
    '''

    def test_cache_invalidation_when_data_changes(self, tmp_path):
        '''Test that rebuilding cache with changed data produces different results.'''
        # Create initial descriptions
        data_v1 = {
            'index': [0, 1],
            'code': ['311111', '311112'],
            'title': ['Dog Food Manufacturing', 'Cat Food Manufacturing'],
            'description': ['Manufacture dog food', 'Manufacture cat food'],
            'excluded': ['', ''],
            'examples': ['', ''],
        }
        df_v1 = pl.DataFrame(data_v1)
        desc_path = tmp_path / 'descriptions.parquet'
        df_v1.write_parquet(desc_path)

        # Build initial cache
        cache_v1 = _build_tokenization_cache(
            str(desc_path), 'sentence-transformers/all-MiniLM-L6-v2', 128
        )

        # Verify initial cache
        assert len(cache_v1) == 2
        assert cache_v1[0]['code'] == '311111'
        title_dict_v1 = cache_v1[0]['title']  # type: ignore
        v1_title_tokens = title_dict_v1['input_ids'].clone()  # type: ignore

        # Update descriptions with different content
        data_v2 = {
            'index': [0, 1, 2],  # Added new item
            'code': ['311111', '311112', '321111'],
            'title': [
                'Pet Food Manufacturing',  # Changed title
                'Cat Food Manufacturing',
                'Sawmills',  # New item
            ],
            'description': [
                'Manufacture pet food',  # Changed
                'Manufacture cat food',
                'Saw logs',  # New
            ],
            'excluded': ['', '', ''],
            'examples': ['', '', ''],
        }
        df_v2 = pl.DataFrame(data_v2)
        df_v2.write_parquet(desc_path)

        # Build new cache
        cache_v2 = _build_tokenization_cache(
            str(desc_path), 'sentence-transformers/all-MiniLM-L6-v2', 128
        )

        # Verify cache reflects changes
        assert len(cache_v2) == 3  # New item added
        # Title tokens should be different after content change
        title_dict_v2 = cache_v2[0]['title']  # type: ignore
        v2_title_tokens = title_dict_v2['input_ids']  # type: ignore
        assert not torch.equal(v1_title_tokens, v2_title_tokens)
        # New item should exist
        assert 2 in cache_v2
        assert cache_v2[2]['code'] == '321111'

    def test_cache_invalidation_max_length_change(self, tmp_path):
        '''Test that cache is invalid when max_length changes.'''
        # Create descriptions
        data = {
            'index': [0],
            'code': ['311111'],
            'title': ['Dog Food'],
            'description': ['A ' * 200],  # Long description
            'excluded': [''],
            'examples': [''],
        }
        df = pl.DataFrame(data)
        desc_path = tmp_path / 'descriptions.parquet'
        df.write_parquet(desc_path)

        # Build cache with max_length=128
        cache_128 = _build_tokenization_cache(
            str(desc_path), 'sentence-transformers/all-MiniLM-L6-v2', 128
        )

        # Build cache with max_length=64
        cache_64 = _build_tokenization_cache(
            str(desc_path), 'sentence-transformers/all-MiniLM-L6-v2', 64
        )

        # Tokens should have different shapes
        desc_dict_128 = cache_128[0]['description']  # type: ignore
        desc_dict_64 = cache_64[0]['description']  # type: ignore
        assert desc_dict_128['input_ids'].shape == (128, )  # type: ignore
        assert desc_dict_64['input_ids'].shape == (64, )  # type: ignore

    def test_cache_file_update_after_invalidation(self, tmp_path):
        '''Test that saved cache file is updated when data changes.'''
        # Create initial descriptions
        data_v1 = {
            'index': [0],
            'code': ['311111'],
            'title': ['Dog Food'],
            'description': ['Make dog food'],
            'excluded': [''],
            'examples': [''],
        }
        df_v1 = pl.DataFrame(data_v1)
        desc_path = tmp_path / 'descriptions.parquet'
        df_v1.write_parquet(desc_path)

        cache_path = tmp_path / 'cache.pt'

        # Build and save initial cache
        cache_v1 = _build_tokenization_cache(
            str(desc_path), 'sentence-transformers/all-MiniLM-L6-v2', 128
        )
        _save_tokenization_cache(cache_v1, str(cache_path))

        # Record modification time
        import os
        import time

        mtime_v1 = os.path.getmtime(cache_path)
        time.sleep(0.1)  # Ensure time difference

        # Update data
        data_v2 = {
            'index': [0, 1],
            'code': ['311111', '311112'],
            'title': ['Dog Food', 'Cat Food'],
            'description': ['Make dog food', 'Make cat food'],
            'excluded': ['', ''],
            'examples': ['', ''],
        }
        df_v2 = pl.DataFrame(data_v2)
        df_v2.write_parquet(desc_path)

        # Build and save new cache
        cache_v2 = _build_tokenization_cache(
            str(desc_path), 'sentence-transformers/all-MiniLM-L6-v2', 128
        )
        _save_tokenization_cache(cache_v2, str(cache_path))

        # Verify file was updated
        mtime_v2 = os.path.getmtime(cache_path)
        assert mtime_v2 > mtime_v1

        # Verify loaded cache has new data
        loaded = _load_tokenization_cache(str(cache_path))
        assert loaded is not None
        assert len(loaded) == 2

    def test_tokenization_cache_function_rebuilds_on_missing(self, tmp_path):
        '''Test tokenization_cache() rebuilds when cache file is deleted.'''
        # Create descriptions
        data = {
            'index': [0, 1],
            'code': ['311111', '311112'],
            'title': ['Dog Food', 'Cat Food'],
            'description': ['Make dog food', 'Make cat food'],
            'excluded': ['', ''],
            'examples': ['', ''],
        }
        df = pl.DataFrame(data)
        desc_path = tmp_path / 'descriptions.parquet'
        df.write_parquet(desc_path)

        cache_path = tmp_path / 'token_cache.pt'
        cfg = TokenizationConfig(
            descriptions_parquet=str(desc_path),
            tokenizer_name='sentence-transformers/all-MiniLM-L6-v2',
            max_length=128,
            output_path=str(cache_path),
        )

        # Build initial cache
        cache_v1 = tokenization_cache(cfg, use_locking=True)
        assert cache_path.exists()
        assert len(cache_v1) == 2

        # Delete cache file (simulating invalidation)
        cache_path.unlink()
        assert not cache_path.exists()

        # Should rebuild cache
        cache_v2 = tokenization_cache(cfg, use_locking=True)
        assert cache_path.exists()
        assert cache_v2 is not None
        assert len(cache_v2) == 2

    def test_cache_invalidation_different_tokenizer(self, tmp_path):
        '''Test that different tokenizers produce different caches.

        Note: This test uses the same tokenizer with different max_lengths
        as a proxy for demonstrating that configuration changes affect output.
        '''
        # Create descriptions
        data = {
            'index': [0],
            'code': ['311111'],
            'title': ['Dog Food Manufacturing'],
            'description': ['Manufacture dog and cat food products'],
            'excluded': [''],
            'examples': [''],
        }
        df = pl.DataFrame(data)
        desc_path = tmp_path / 'descriptions.parquet'
        df.write_parquet(desc_path)

        # Build cache with different configurations
        cache_a = _build_tokenization_cache(
            str(desc_path), 'sentence-transformers/all-MiniLM-L6-v2', 64
        )
        cache_b = _build_tokenization_cache(
            str(desc_path), 'sentence-transformers/all-MiniLM-L6-v2', 128
        )

        # Same tokenizer but different max_length should produce different results
        desc_dict_a = cache_a[0]['description']  # type: ignore
        desc_dict_b = cache_b[0]['description']  # type: ignore
        assert desc_dict_a['input_ids'].shape != desc_dict_b['input_ids'].shape  # type: ignore
