# -------------------------------------------------------------------------------------------------
# Imports and settings
# -------------------------------------------------------------------------------------------------

import logging
from typing import Any, Dict, List, Optional

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, IterableDataset

from naics_embedder.data_loader.streaming_dataset import create_streaming_dataset
from naics_embedder.utils.config import StreamingConfig, TokenizationConfig

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------------------------------------
# Collate function for DataLoader
# -------------------------------------------------------------------------------------------------

def collate_fn(batch: List[Dict]) -> Dict:
    
    channels = ['title', 'description', 'excluded', 'examples']
    
    # Initialize batch dictionaries
    anchor_batch = {channel: {} for channel in channels}
    positive_batch = {channel: {} for channel in channels}
    negatives_batch = {channel: {} for channel in channels}
    
    # Collect codes and indices for evaluation tracking
    anchor_codes = []
    positive_codes = []
    negative_codes = []
    
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
        for neg_dict in item['negatives']:
            negative_codes.append(neg_dict['negative_code'])
    
    return {
        'anchor': anchor_batch,
        'positive': positive_batch,
        'negatives': negatives_batch,
        'batch_size': len(batch),
        'k_negatives': len(batch[0]['negatives']),
        'anchor_code': anchor_codes,
        'positive_code': positive_codes,
        'negative_codes': negative_codes
    }


# -------------------------------------------------------------------------------------------------
# Wrapper to make generator function work with DataLoader
# -------------------------------------------------------------------------------------------------

class GeneratorDataset(IterableDataset):
    
    def __init__(self, generator_fn, tokenization_cfg, *args, **kwargs):
        '''
        Args:
            generator_fn: Function to create the streaming generator
            tokenization_cfg: TokenizationConfig for loading cache (not the cache itself)
            *args, **kwargs: Additional arguments for generator_fn
        '''
        self.generator_fn = generator_fn
        self.tokenization_cfg = tokenization_cfg
        self.args = args
        self.kwargs = kwargs
        # Cache will be loaded lazily per worker
        self._token_cache = None
    
    def _get_token_cache(self):
        '''Lazily load token cache once per worker process.'''
        if self._token_cache is None:
            from naics_embedder.data_loader.tokenization_cache import tokenization_cache
            self._token_cache = tokenization_cache(self.tokenization_cfg)
        return self._token_cache
    
    def __iter__(self):
        # Get worker info for proper data sharding across workers
        worker_info = torch.utils.data.get_worker_info()
        
        # Load cache in this worker process
        token_cache = self._get_token_cache()
        
        # Create the generator, passing the token cache
        generator = self.generator_fn(token_cache, *self.args, **self.kwargs)
        
        # If using multiple workers, each worker should only process a subset
        if worker_info is None:
            # Single-process data loading, return the full iterator
            return generator
        else:
            # Multi-process data loading: split data by worker
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            
            # Skip items not assigned to this worker (round-robin distribution)
            return (item for i, item in enumerate(generator) if i % num_workers == worker_id)


# -------------------------------------------------------------------------------------------------
# Main DataModule for PyTorch Lightning (optional but recommended)
# -------------------------------------------------------------------------------------------------

class NAICSDataModule(LightningDataModule):
    
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

        if streaming_config is not None:    
            val_streaming_config = streaming_config.copy()
            val_streaming_config['seed'] = seed + 1

            curriculum = StreamingConfig(**streaming_config)
            val_curriculum = StreamingConfig(**val_streaming_config)
        
        else:
            curriculum = StreamingConfig()
            val_curriculum = StreamingConfig(seed=seed + 1)

        # Create tokenization config (workers will load cache lazily)
        self.tokenization_cfg = TokenizationConfig(
            descriptions_parquet=descriptions_path,
            tokenizer_name=tokenizer_name,
            max_length=curriculum.max_length
        )

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
        '''Build tokenization cache before worker processes are spawned.'''
        # This is called only on the main process before worker setup
        # Build the cache so all workers can load it instead of building it
        import os
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        from naics_embedder.data_loader.tokenization_cache import tokenization_cache
        logger.info('Preparing tokenization cache in main process...')
        tokenization_cache(self.tokenization_cfg)
    
    def train_dataloader(self) -> DataLoader:

        '''Create training dataloader.'''
        
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=0,  # Use single process to avoid multiprocessing issues with tokenizer
            collate_fn=collate_fn,
            persistent_workers=False
        )
    
    def val_dataloader(self) -> DataLoader:
        
        '''Create validation dataloader.'''
        
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=0,  # Use single process to avoid multiprocessing issues with tokenizer
            collate_fn=collate_fn,
            persistent_workers=False
        )