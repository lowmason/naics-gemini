# -------------------------------------------------------------------------------------------------
# Imports and settings
# -------------------------------------------------------------------------------------------------

import fcntl
import logging
import os
import time
from pathlib import Path
from typing import Dict, Optional, TextIO, Tuple, Union

import polars as pl
import torch
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from naics_embedder.utils.config import TokenizationConfig

logger = logging.getLogger(__name__)

# Disable tokenizer parallelism to avoid fork issues with multiprocessing
os.environ['TOKENIZERS_PARALLELISM'] = 'false'


# -------------------------------------------------------------------------------------------------
# Tokenization functions
# -------------------------------------------------------------------------------------------------

def _tokenize_text(
    dict: Dict[str, str],
    field: str,
    counter: Dict[str, int],
    tokenizer: PreTrainedTokenizerBase,
    max_length: int
) -> Tuple[Dict[str, torch.Tensor], Dict[str, int]]:
    
    '''Tokenize a single text field.'''

    text = dict.get(field, '')
    
    if not text or text == '':
        text = '[EMPTY]'
    else:
        counter[field] += 1
        
    encoded = tokenizer(
        text,
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    )

    input_ids = encoded['input_ids']
    attention_mask = encoded['attention_mask']

    encoding = {
        'input_ids': torch.squeeze(input_ids), # type: ignore
        'attention_mask': torch.squeeze(attention_mask) # type: ignore
    }

    return encoding, counter

def _build_tokenization_cache(
    descriptions_path: str,
    tokenizer_name: str,
    max_length: int
) -> Dict[int, Dict[str, torch.Tensor]]:
    
    '''Build tokenization cache from descriptions file.'''
    
    logger.info('Building tokenization cache...')
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    # DataFrame iterator
    df_iter = (
        pl
        .read_parquet(
            descriptions_path
        )
        .sort('index')
        .iter_rows(named=True)
    )
    
    # Tokenization cache
    cache, cnt = {}, { 'title': 0, 'description': 0, 'excluded': 0, 'examples': 0 }
    for row in df_iter:

        idx, code = row['index'], row['code']
        
        title, cnt = _tokenize_text(row, 'title', cnt, tokenizer, 24)
        description, cnt = _tokenize_text(row, 'description', cnt, tokenizer, max_length)
        excluded, cnt = _tokenize_text(row, 'excluded', cnt, tokenizer, max_length)
        examples, cnt = _tokenize_text(row, 'examples', cnt, tokenizer, max_length)
        
        cache[idx] = {
            'code': code,
            'title': title,
            'description': description,
            'excluded': excluded,
            'examples': examples
        }
    
    logger.info('Cache built with:')
    logger.info(f'  {cnt["title"]: ,} titles')
    logger.info(f'  {cnt["description"]: ,} descriptions')
    logger.info(f'  {cnt["excluded"]: ,} exclusions')
    logger.info(f'  {cnt["examples"]: ,} examples')

    return cache


def _save_tokenization_cache(
    cache: Dict[int, Dict[str, torch.Tensor]],
    cache_path: str
) -> Path:
    
    '''Save tokenization cache to disk.'''
    
    cache_file = Path(cache_path)
    cache_dir = cache_file.parent
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    torch.save(cache, cache_file)
    logger.info(f'Saved tokenization cache to: {cache_file.resolve()}')
    
    return cache_file


def _load_tokenization_cache(
    cache_path: str,
    verbose: bool = True
) -> Optional[Dict[int, Dict[str, torch.Tensor]]]:
    
    '''Load tokenization cache from disk if it exists.'''
    
    cache_file = Path(cache_path)
    
    if cache_file.exists():
        if verbose:
            logger.info(f'Loading tokenization cache from: {cache_file.resolve()}')
        else:
            # Still log for workers but at debug level
            logger.debug('Loading tokenization cache (worker process)')
        try:
            import time
            start_time = time.time()
            cache = torch.load(cache_file, weights_only=True, map_location='cpu')
            load_time = time.time() - start_time
            if verbose:
                logger.info(f'Tokenization cache loaded in {load_time:.2f}s')
            else:
                logger.debug(f'Tokenization cache loaded in {load_time:.2f}s (worker)')
            return cache
        except Exception as e:
            if verbose:
                logger.error(f'Failed to load tokenization cache: {e}')
            raise
    
    return None


# -------------------------------------------------------------------------------------------------
# File locking utilities for multi-worker safety
# -------------------------------------------------------------------------------------------------

def _acquire_lock(lock_path: Path, timeout: int = 300) -> Optional[TextIO]:
    '''
    Acquire an exclusive lock on a lock file.
    
    Args:
        lock_path: Path to lock file
        timeout: Maximum time to wait for lock (seconds)
        
    Returns:
        Lock file object if acquired, None if timeout
    '''
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        lock_file = lock_path.open('w')
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                return lock_file
            except BlockingIOError:
                time.sleep(0.1)
                continue
        
        # Timeout - close file and return None
        lock_file.close()
        return None
        
    except Exception as e:
        logger.warning(f'Error acquiring lock: {e}')
        return None


def _release_lock(lock_file: Optional[TextIO]) -> None:
    '''Release the lock and close the file.'''
    if lock_file is None:
        return
        
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        lock_file.close()
    except Exception as e:
        logger.debug(f'Error releasing lock: {e}')


# -------------------------------------------------------------------------------------------------
# Main tokenization functions
# -------------------------------------------------------------------------------------------------

def tokenization_cache(
    cfg: TokenizationConfig = TokenizationConfig(),
    use_locking: bool = True
) -> Dict[int, Dict[str, torch.Tensor]]:
    
    '''
    Get tokenization cache, loading from disk or building if necessary.
    
    This function is safe for multi-worker environments. It uses file locking
    to ensure only one worker builds the cache, while others wait and then load it.
    
    Args:
        cfg: TokenizationConfig
        use_locking: If False, skip locking (for fast reads when cache exists)
    '''

    cache_path = Path(cfg.output_path)
    
    # Fast path: try to load existing cache first (no locking needed for reads)
    if cache_path.exists():
        try:
            cache = _load_tokenization_cache(cfg.output_path)
            if cache is not None:
                return cache
        except Exception as e:
            logger.warning(f'Error loading cache, will try to rebuild: {e}')
    
    # If we're not using locking (e.g., cache should already exist), fail fast
    if not use_locking:
        raise RuntimeError(
            f'Tokenization cache not found at {cache_path} and locking disabled. '
            f'Cache should be built in prepare_data() before workers are spawned.'
        )
    
    lock_path = cache_path.with_suffix('.lock')
    
    # Cache doesn't exist - need to build it with locking
    lock_file = None
    try:
        # Try to acquire lock
        lock_file = _acquire_lock(lock_path, timeout=300)
        
        if lock_file is None:
            # Could not acquire lock - another worker is building the cache
            # Wait for cache to be built by that worker
            logger.info('Another worker is building cache, waiting...')
            max_wait = 300  # 5 minutes
            check_interval = 0.5  # Check every 500ms
            start_time = time.time()
            
            while time.time() - start_time < max_wait:
                cache = _load_tokenization_cache(cfg.output_path)
                if cache is not None:
                    logger.info('Cache was built by another worker, loaded successfully')
                    return cache
                time.sleep(check_interval)
            
            raise RuntimeError(
                f'Timeout waiting for cache to be built by another process. '
                f'Cache file should appear at: {cache_path}'
            )
        
        # We have the lock - double-check cache wasn't built while we waited
        cache = _load_tokenization_cache(cfg.output_path)
        if cache is not None:
            logger.info('Cache was built while waiting for lock, loaded successfully')
            return cache
        
        # Build cache (we're the only one doing this)
        logger.info('Building tokenization cache (this may take a few minutes)...')
        cache = _build_tokenization_cache(
            cfg.descriptions_parquet, 
            cfg.tokenizer_name, 
            cfg.max_length  # type: ignore
        )
        
        # Save to temporary file first, then rename (atomic operation)
        temp_path = cache_path.with_suffix('.tmp')
        _save_tokenization_cache(cache, str(temp_path))
        
        # Atomic rename - ensures cache file appears all at once
        temp_path.replace(cache_path)
        
        logger.info('Tokenization cache built and saved successfully')
        return cache
        
    finally:
        # Always release lock and clean up
        _release_lock(lock_file)
        # Clean up lock file if it exists
        try:
            if lock_path.exists():
                lock_path.unlink()
        except Exception:
            pass  # Ignore cleanup errors


def get_tokens(
    idx_code: Union[int, str],
    cache: Dict[int, Dict[str, torch.Tensor]]
) -> Dict[int, Dict[str, torch.Tensor]]:
    
    '''Get tokens for a specific NAICS index or code from cache.'''

    if isinstance(idx_code, int):
        key = idx_code

    elif isinstance(idx_code, str):
        for k, v in cache.items():
            if v['code'] == idx_code:
                key = k
                break

    else:
        raise ValueError('idx_code must be an int or str')
    
    return {key: cache[key]} # type: ignore