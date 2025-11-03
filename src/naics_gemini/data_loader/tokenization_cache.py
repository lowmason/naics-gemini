# -------------------------------------------------------------------------------------------------
# Imports and settings
# -------------------------------------------------------------------------------------------------

import logging
from pathlib import Path
from typing import Dict, Optional

import polars as pl
import torch
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------------------------------------
# Cache tokenized text fields for NAICS codes
# -------------------------------------------------------------------------------------------------

class TokenizationCache:
    
    def __init__(
        self,
        descriptions_path: str = './data/naics_descriptions.parquet',
        tokenizer_name: str = 'sentence-transformers/all-mpnet-base-v2',
        cache_dir: str = './data/token_cache',
        max_length: int = 512
    ):
        
        self.descriptions_path = descriptions_path
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.cache_dir = Path(cache_dir)
        self.max_length = max_length
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self._cache: Optional[Dict[str, Dict[str, torch.Tensor]]] = None
        

    # Build or load tokenization cache
    def build_cache(self) -> Dict[str, Dict[str, torch.Tensor]]:
        
        # Cache file path
        cache_file = Path(f'{self.cache_dir}/token_cache.pt')
        
        # Load from cache if it exists
        if cache_file.exists():
            logger.info(f'Loading tokenization cache from {cache_file}')
            self._cache = torch.load(cache_file)
            return self._cache #type: ignore
        
        # Otherwise, build the cache
        logger.info('Building tokenization cache...')
        
        # DataFrame iterator
        df_iter = (
            pl
            .read_parquet(
                self.descriptions_path
            )
            .iter_rows(named=True)
        )
        
        # Tokenization cache
        cache = {}        
        for row in df_iter:
            code = row['code']            
            title = row.get('title', '')
            description = row.get('description', '')
            excluded = row.get('excluded', '')
            examples = row.get('examples', '')
            
            cache[code] = {
                'title': self._tokenize(title or ''),
                'description': self._tokenize(description or ''),
                'excluded': self._tokenize(excluded or ''),
                'examples': self._tokenize(examples or '')
            }
        
        torch.save(cache, cache_file)
        logger.info(f'Saved tokenization cache to {cache_file}')
        
        self._cache = cache
        return cache
    
    
    # Tokenize text field
    def _tokenize(self, text: str) -> Dict[str, torch.Tensor]:
        
        if not text or text == '':
            text = '[EMPTY]'
            
        encoded = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0)
        }
    
    # Get tokens for a specific NAICS code
    def get_tokens(self, code: str) -> Dict[str, Dict[str, torch.Tensor]]:
        
        if self._cache is None:
            self.build_cache()
            
        return self._cache[code] #type: ignore
    

    @property
    def cache(self) -> Dict[str, Dict[str, torch.Tensor]]:
        if self._cache is None:
            self.build_cache()
        return self._cache #type: ignore
