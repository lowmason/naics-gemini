# -------------------------------------------------------------------------------------------------
# Imports and settings
# -------------------------------------------------------------------------------------------------

import logging
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader, IterableDataset

from naics_gemini.data_loader.streaming_dataset import (
    CurriculumConfig,
    create_streaming_dataset,
)
from naics_gemini.data_loader.tokenization_cache import tokenization_cache

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------------------------------------
# Collate function for DataLoader
# -------------------------------------------------------------------------------------------------

def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    '''
    Collate function for batching training examples.
    
    Args:
        batch: List of training examples from streaming dataset
        
    Returns:
        Batched tensors ready for model input
    '''
    
    channels = ['title', 'description', 'excluded', 'examples']
    
    # Initialize batch dictionaries
    anchor_batch = {channel: {} for channel in channels}
    positive_batch = {channel: {} for channel in channels}
    negatives_batch = {channel: {} for channel in channels}
    
    # Collect codes for evaluation tracking
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
            anchor_ids.append(item['anchor'][channel]['input_ids'])
            anchor_masks.append(item['anchor'][channel]['attention_mask'])
            positive_ids.append(item['positive'][channel]['input_ids'])
            positive_masks.append(item['positive'][channel]['attention_mask'])
        
        # Stack anchor and positive
        anchor_batch[channel]['input_ids'] = torch.stack(anchor_ids)
        anchor_batch[channel]['attention_mask'] = torch.stack(anchor_masks)
        positive_batch[channel]['input_ids'] = torch.stack(positive_ids)
        positive_batch[channel]['attention_mask'] = torch.stack(positive_masks)
        
        # Collect all negatives for this channel
        all_neg_ids = []
        all_neg_masks = []
        for item in batch:
            for neg_tokens in item['negatives']:
                all_neg_ids.append(neg_tokens[channel]['input_ids'])
                all_neg_masks.append(neg_tokens[channel]['attention_mask'])
        
        # Stack negatives
        negatives_batch[channel]['input_ids'] = torch.stack(all_neg_ids)
        negatives_batch[channel]['attention_mask'] = torch.stack(all_neg_masks)
    
    # Extract codes from batch items
    for item in batch:
        anchor_codes.append(item.get('anchor_code', ''))
        positive_codes.append(item.get('positive_code', ''))
        if 'negative_codes' in item:
            negative_codes.extend(item['negative_codes'])
    
    return {
        'anchor': anchor_batch,
        'positive': positive_batch,
        'negatives': negatives_batch,
        'batch_size': len(batch),
        'k_negatives': len(batch[0]['negatives']),
        # Add codes for evaluation tracking
        'anchor_code': anchor_codes,
        'positive_code': positive_codes,
        'negative_codes': negative_codes
    }


# -------------------------------------------------------------------------------------------------
# Wrapper to make generator function work with DataLoader
# -------------------------------------------------------------------------------------------------

class GeneratorDataset(IterableDataset):
    '''Wrapper to make a generator function work with PyTorch DataLoader.'''
    
    def __init__(self, generator_fn, *args, **kwargs):
        self.generator_fn = generator_fn
        self.args = args
        self.kwargs = kwargs
    
    def __iter__(self):
        return self.generator_fn(*self.args, **self.kwargs)


# -------------------------------------------------------------------------------------------------
# Main DataModule for PyTorch Lightning (optional but recommended)
# -------------------------------------------------------------------------------------------------

class NAICSDataModule:
    '''
    Data module for NAICS contrastive learning.
    
    This class encapsulates all data loading logic including:
    - Tokenization caching
    - Curriculum-based filtering
    - Streaming dataset creation
    - DataLoader configuration
    '''
    
    def __init__(
        self,
        descriptions_path: str = './data/naics_descriptions.parquet',
        triplets_path: str = './data/naics_training_pairs',
        tokenizer_name: str = 'sentence-transformers/all-mpnet-base-v2',
        curriculum_config: Optional[Dict] = None,
        batch_size: int = 32,
        num_workers: int = 4,
        val_split: float = 0.05,
        seed: int = 42,
        cache_dir: str = './data/token_cache'
    ):
        '''
        Initialize NAICS DataModule.
        
        Args:
            descriptions_path: Path to NAICS descriptions parquet
            triplets_path: Path to triplets parquet directory
            tokenizer_name: HuggingFace tokenizer name
            curriculum_config: Dictionary with curriculum settings
            batch_size: Batch size for training
            num_workers: Number of dataloader workers
            val_split: Fraction of data for validation (not used with streaming)
            seed: Random seed
            cache_dir: Directory for tokenization cache
        '''
        self.descriptions_path = descriptions_path
        self.triplets_path = triplets_path
        self.tokenizer_name = tokenizer_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.seed = seed
        self.cache_dir = cache_dir
        
        # Convert curriculum config dict to CurriculumConfig
        curriculum_config = curriculum_config or {}
        self.curriculum = CurriculumConfig(**curriculum_config)
        
        # Will be set during setup
        self.token_cache = None
        self.train_dataset = None
        self.val_dataset = None
    
    
    def setup(self, stage: Optional[str] = None):
        '''
        Setup datasets and cache.
        
        Args:
            stage: 'fit', 'validate', 'test', or 'predict'
        '''
        logger.info('Setting up NAICS DataModule...')
        
        # Load or create tokenization cache
        logger.info('Loading tokenization cache...')
        self.token_cache = tokenization_cache(
            fields_path=self.descriptions_path,
            tokenizer_name=self.tokenizer_name,
            max_length=512,
            cache_dir=self.cache_dir
        )
        logger.info(f'Token cache loaded: {len(self.token_cache)} codes')
        
        if stage == 'fit' or stage is None:
            # Create training dataset
            logger.info('Creating training dataset...')
            self.train_dataset = GeneratorDataset(
                create_streaming_dataset,
                descriptions_path=self.descriptions_path,
                triplets_path=self.triplets_path,
                token_cache=self.token_cache,
                curriculum=self.curriculum,
                seed=self.seed
            )
            
            # Create validation dataset (same as train but different seed)
            logger.info('Creating validation dataset...')
            self.val_dataset = GeneratorDataset(
                create_streaming_dataset,
                descriptions_path=self.descriptions_path,
                triplets_path=self.triplets_path,
                token_cache=self.token_cache,
                curriculum=self.curriculum,
                seed=self.seed + 1  # Different seed for validation
            )
            
            logger.info('DataModule setup complete!')
    
    
    def train_dataloader(self) -> DataLoader:
        '''Create training dataloader.'''
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            persistent_workers=True if self.num_workers > 0 else False
        )
    
    
    def val_dataloader(self) -> DataLoader:
        '''Create validation dataloader.'''
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            persistent_workers=True if self.num_workers > 0 else False
        )