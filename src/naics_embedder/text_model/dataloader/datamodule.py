# -------------------------------------------------------------------------------------------------
# Imports and settings
# -------------------------------------------------------------------------------------------------

import logging
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, IterableDataset

from naics_embedder.text_model.dataloader.streaming_dataset import (
    _get_final_cache_path,
    create_streaming_dataset,
    create_streaming_generator,
)
from naics_embedder.utils.config import StreamingConfig, TokenizationConfig

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------------------------------------
# Collate function for DataLoader
# -------------------------------------------------------------------------------------------------

def collate_fn(batch: List[Dict]) -> Dict:
    '''Collate function to batch triplets for training.
    
    Supports multi-level supervision (Issue #18):
    - Each item can have multiple positives (one per ancestor level)
    - All positives for an anchor share the same negatives
    '''
    channels = ['title', 'description', 'excluded', 'examples']
    
    # Find maximum number of negatives in batch and pad shorter lists
    max_negatives = max(len(item['negatives']) for item in batch) if batch else 0
    if max_negatives == 0:
        raise ValueError('Batch contains items with no negatives - cannot create training batch')
    
    for item in batch:
        if len(item['negatives']) < max_negatives:
            # Pad by repeating the last negative
            last_negative = item['negatives'][-1] if item['negatives'] else None
            if last_negative is None:
                anchor_code = item.get('anchor_code', 'unknown')
                raise ValueError(f'Item has no negatives to pad from: {anchor_code}')
            padding_needed = max_negatives - len(item['negatives'])
            item['negatives'].extend([last_negative] * padding_needed)
    
    # Check if we have multi-level supervision (positives is a list)
    has_multiple_positives = (
        batch and 'positives' in batch[0] 
        and isinstance(batch[0]['positives'], list)
    )
    
    if has_multiple_positives:
        # Multi-level supervision: expand batch to include all positives
        expanded_batch = []
        for item in batch:
            anchor_embedding = item['anchor_embedding']
            positives_list = item['positives']
            negatives = item['negatives']
            
            # Create one entry per positive
            for positive in positives_list:
                expanded_batch.append({
                    'anchor_embedding': anchor_embedding,
                    'positive_embedding': positive['positive_embedding'],
                    'negatives': negatives,
                    'anchor_code': item['anchor_code'],
                    'positive_code': positive['positive_code'],
                    'positive_level': len(positive['positive_code'])  # Store level for logging
                })
        batch = expanded_batch
    
    # Initialize batch dictionaries
    anchor_batch = {channel: {} for channel in channels}
    positive_batch = {channel: {} for channel in channels}
    negatives_batch = {channel: {} for channel in channels}
    
    # Collect codes and indices for evaluation tracking
    anchor_codes = []
    positive_codes = []
    negative_codes = []
    positive_levels = [] if has_multiple_positives else None
    
    # Process each channel
    for channel in channels:
        anchor_ids = []
        anchor_masks = []
        positive_ids = []
        positive_masks = []
        
        # Collect anchor and positive for this channel
        for item in batch:
            anchor_ids.append(item['anchor_embedding'][channel]['input_ids'])
            anchor_masks.append(item['anchor_embedding'][channel]['attention_mask'])
            positive_ids.append(item['positive_embedding'][channel]['input_ids'])
            positive_masks.append(item['positive_embedding'][channel]['attention_mask'])
        
        # Stack anchor
        anchor_batch[channel]['input_ids'] = torch.stack(anchor_ids)
        anchor_batch[channel]['attention_mask'] = torch.stack(anchor_masks)
        
        # Stack positive
        positive_batch[channel]['input_ids'] = torch.stack(positive_ids)
        positive_batch[channel]['attention_mask'] = torch.stack(positive_masks)
        
        # Collect all negatives for this channel
        all_neg_ids = []
        all_neg_masks = []
        for item in batch:
            for neg_dict in item['negatives']:
                all_neg_ids.append(neg_dict['negative_embedding'][channel]['input_ids'])
                all_neg_masks.append(neg_dict['negative_embedding'][channel]['attention_mask'])
        
        # Stack negatives
        negatives_batch[channel]['input_ids'] = torch.stack(all_neg_ids)
        negatives_batch[channel]['attention_mask'] = torch.stack(all_neg_masks)
    
    # Extract codes from batch items
    for item in batch:
        anchor_codes.append(item['anchor_code'])
        positive_codes.append(item['positive_code'])
        negative_codes.append([neg_dict['negative_code'] for neg_dict in item['negatives']])
        if has_multiple_positives and 'positive_level' in item:
            if positive_levels is None:
                positive_levels = []
            positive_levels.append(item['positive_level'])
    
    result = {
        'anchor': anchor_batch,
        'positive': positive_batch,
        'negatives': negatives_batch,
        'batch_size': len(batch),
        'k_negatives': max_negatives,
        'anchor_code': anchor_codes,
        'positive_code': positive_codes,
        'negative_codes': negative_codes
    }
    
    # Add positive_levels for multi-level supervision tracking
    if positive_levels is not None:
        result['positive_levels'] = positive_levels
    
    return result


# -------------------------------------------------------------------------------------------------
# Wrapper to make generator function work with DataLoader
# -------------------------------------------------------------------------------------------------

class GeneratorDataset(IterableDataset):
    '''Dataset wrapper for streaming generators.'''
    
    def __init__(self, generator_fn, tokenization_cfg, *args, **kwargs):
        self.generator_fn = generator_fn
        self.tokenization_cfg = tokenization_cfg
        self.args = args
        self.kwargs = kwargs
        self._token_cache = None
    
    def _get_token_cache(self):
        '''Lazily load token cache once per worker process.'''
        if self._token_cache is None:
            import random
            import time
            
            # Set tokenizer parallelism to false in each worker
            os.environ['TOKENIZERS_PARALLELISM'] = 'false'
            
            from naics_embedder.text_model.dataloader.tokenization_cache import (
                _load_tokenization_cache,
                tokenization_cache,
            )
            
            # Fast path: load cache directly (should already exist from prepare_data)
            cache_path = Path(self.tokenization_cfg.output_path)
            if cache_path.exists():
                try:
                    # Add small random delay to stagger worker loading
                    worker_info = torch.utils.data.get_worker_info()
                    if worker_info is not None:
                        delay = random.uniform(0, 0.5) * (worker_info.id + 1)
                        time.sleep(delay)
                    
                    self._token_cache = _load_tokenization_cache(
                        self.tokenization_cfg.output_path,
                        verbose=False
                    )
                    if self._token_cache is not None:
                        return self._token_cache
                except Exception as e:
                    worker_pid = os.getpid()
                    logger.warning(
                        f'Worker {worker_pid} failed to load cache: {e}, '
                        f'will try with locking'
                    )
            
            # Fallback: use full tokenization_cache() with locking
            worker_pid = os.getpid()
            logger.warning(
                f'Worker {worker_pid} cache not found, loading with locking '
                f'(this should be rare)'
            )
            self._token_cache = tokenization_cache(self.tokenization_cfg, use_locking=True)
            logger.debug(f'Worker {os.getpid()} loaded tokenization cache')
        
        return self._token_cache
    
    def __iter__(self):
        '''Iterate over dataset with worker sharding.'''
        worker_info = torch.utils.data.get_worker_info()
        token_cache = self._get_token_cache()
        generator = self.generator_fn(token_cache, *self.args, **self.kwargs)
        
        if worker_info is None:
            return generator
        else:
            # Shard the generator efficiently
            # Each worker takes every num_workers-th item starting from worker_id
            count = 0
            for item in generator:
                if count % worker_info.num_workers == worker_info.id:
                    yield item
                count += 1


# -------------------------------------------------------------------------------------------------
# Main DataModule for PyTorch Lightning
# -------------------------------------------------------------------------------------------------

class NAICSDataModule(LightningDataModule):
    '''DataModule for NAICS embedding training.'''
    
    def __init__(
        self,
        descriptions_path: str = './data/naics_descriptions.parquet',
        triplets_path: str = './data/naics_training_pairs',
        tokenizer_name: str = 'sentence-transformers/all-MiniLM-L6-v2',
        streaming_config: Optional[Dict] = None,
        batch_size: int = 32,
        num_workers: int = 4,
        seed: int = 42,
        val_split: float = 0.1,
        **kwargs: Any
    ):
        super().__init__()
        
        self.descriptions_path = descriptions_path
        self.triplets_path = triplets_path
        self.tokenizer_name = tokenizer_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # Create streaming configs
        if streaming_config is not None:
            val_streaming_config = streaming_config.copy()
            val_streaming_config['seed'] = seed + 1
            curriculum = StreamingConfig(**streaming_config)
            val_curriculum = StreamingConfig(**val_streaming_config)
        else:
            curriculum = StreamingConfig()
            val_curriculum = StreamingConfig(seed=seed + 1)
        
        self.tokenization_cfg = TokenizationConfig(
            descriptions_parquet=descriptions_path,
            tokenizer_name=tokenizer_name,
            max_length=curriculum.max_length
        )
        
        # Store streaming configs for use in prepare_data()
        self.train_streaming_cfg = curriculum
        self.val_streaming_cfg = val_curriculum
        
        # Create datasets
        logger.info('  • Creating training dataset')
        self.train_dataset = GeneratorDataset(
            create_streaming_dataset,
            self.tokenization_cfg,
            curriculum
        )
        
        logger.info('  • Creating validation dataset\n')
        self.val_dataset = GeneratorDataset(
            create_streaming_dataset,
            self.tokenization_cfg,
            val_curriculum
        )
    
    def prepare_data(self):
        '''Build all caches before worker processes are spawned.'''
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        
        from naics_embedder.text_model.dataloader.tokenization_cache import tokenization_cache
        from naics_embedder.utils.utilities import get_indices_codes
        
        # Build tokenization cache
        logger.info('Preparing tokenization cache in main process...')
        tokenization_cache(self.tokenization_cfg)
        
        # Build codes/indices cache
        logger.info('Preparing codes/indices cache in main process...')
        cache_dir = Path(self.tokenization_cfg.descriptions_parquet).parent / 'codes_cache'
        cache_dir.mkdir(parents=True, exist_ok=True)
        codes_cache_path = cache_dir / 'codes_indices.pkl'
        
        if not codes_cache_path.exists():
            logger.info('Loading codes and indices for caching...')
            codes = get_indices_codes('codes')
            code_to_idx = get_indices_codes('code_to_idx')
            
            with open(codes_cache_path, 'wb') as f:
                pickle.dump({'codes': codes, 'code_to_idx': code_to_idx}, f)
            logger.info(f'Cached codes/indices to {codes_cache_path}')
        else:
            logger.info('Codes/indices cache already exists')
        
        # Build streaming query caches (complete pipeline including weighted sampling)
        self._build_streaming_cache(self.train_streaming_cfg, 'training')
        self._build_streaming_cache(self.val_streaming_cfg, 'validation')
    
    def _build_streaming_cache(self, cfg: StreamingConfig, name: str):
        '''Build streaming query cache for a given config.'''
        logger.info(f'Preparing streaming query cache ({name}) in main process...')
        cache_path = _get_final_cache_path(cfg)
        
        if cache_path.exists():
            logger.info(f'{name.capitalize()} streaming query cache already exists')
            return
        
        logger.info(f'Building {name} streaming query cache (this may take 30-60 seconds)...')
        gen = create_streaming_generator(cfg)
        
        try:
            # Consume first item to trigger cache build
            # Cache is saved before iteration starts
            next(gen)
            
            # Verify cache was created
            if cache_path.exists():
                logger.info(f'{name.capitalize()} streaming query cache built successfully')
            else:
                logger.warning('Cache build completed but file not found - this should not happen')
        except StopIteration:
            # Generator was empty, but cache should still be saved
            if cache_path.exists():
                logger.info(f'{name.capitalize()} streaming query cache built successfully')
            else:
                logger.warning('Cache file not found after build attempt')
    
    def train_dataloader(self) -> DataLoader:
        '''Create training dataloader.'''
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            persistent_workers=self.num_workers > 0
        )
    
    def val_dataloader(self) -> DataLoader:
        '''Create validation dataloader.'''
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            persistent_workers=self.num_workers > 0
        )
