# -------------------------------------------------------------------------------------------------
# Imports and settings
# -------------------------------------------------------------------------------------------------

import logging
import os
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import polars as pl
import torch
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from naics_gemini.utils.config import TokenizationConfig

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
    cache_path: str
) -> Optional[Dict[int, Dict[str, torch.Tensor]]]:
    
    '''Load tokenization cache from disk if it exists.'''
    
    cache_file = Path(cache_path)
    
    if cache_file.exists():
        logger.info(f'Loading tokenization cache from: {cache_file.resolve()}')
        return torch.load(cache_file, weights_only=True)
    
    return None


# -------------------------------------------------------------------------------------------------
# Main tokenization functions
# -------------------------------------------------------------------------------------------------

def tokenization_cache(
    cfg: TokenizationConfig = TokenizationConfig()
) -> Dict[int, Dict[str, torch.Tensor]]:
    
    '''Get tokenization cache, loading from disk or building if necessary.'''

    # Try to load from cache
    cache = _load_tokenization_cache(cfg.output_path)
    if cache is not None:
        return cache
    
    # Build cache if it doesn't exist
    cache = _build_tokenization_cache(
        cfg.descriptions_parquet, 
        cfg.tokenizer_name, 
        cfg.max_length # type: ignore
    )
    _save_tokenization_cache(cache, cfg.output_path)

    return cache


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