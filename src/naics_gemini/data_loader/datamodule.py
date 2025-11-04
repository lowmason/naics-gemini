# -------------------------------------------------------------------------------------------------
# Imports and settings
# -------------------------------------------------------------------------------------------------

import logging
from typing import Dict, List, Optional

import pytorch_lightning as pyl
import torch
from torch.utils.data import DataLoader

from naics_gemini.data_loader.streaming_dataset import CurriculumConfig, NAICSStreamingDataset
from naics_gemini.data_loader.tokenization_cache import TokenizationCache

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------------------------------------
# Collate function for DataLoader
# -------------------------------------------------------------------------------------------------

def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    
    channels = ['title', 'description', 'excluded', 'examples']
    
    anchor_batch = {channel: {} for channel in channels}
    positive_batch = {channel: {} for channel in channels}
    negatives_batch = {channel: {} for channel in channels}
    
    # Collect anchor and positive codes for evaluation
    anchor_codes = []
    positive_codes = []
    negative_codes = []
    
    for channel in channels:
        anchor_ids = []
        anchor_masks = []
        positive_ids = []
        positive_masks = []
        
        for item in batch:
            anchor_ids.append(item['anchor'][channel]['input_ids'])
            anchor_masks.append(item['anchor'][channel]['attention_mask'])
            positive_ids.append(item['positive'][channel]['input_ids'])
            positive_masks.append(item['positive'][channel]['attention_mask'])
        
        anchor_batch[channel]['input_ids'] = torch.stack(anchor_ids)
        anchor_batch[channel]['attention_mask'] = torch.stack(anchor_masks)
        positive_batch[channel]['input_ids'] = torch.stack(positive_ids)
        positive_batch[channel]['attention_mask'] = torch.stack(positive_masks)
        
        all_neg_ids = []
        all_neg_masks = []
        for item in batch:
            for neg_tokens in item['negatives']:
                all_neg_ids.append(neg_tokens[channel]['input_ids'])
                all_neg_masks.append(neg_tokens[channel]['attention_mask'])
        
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
    } #type: ignore


# -------------------------------------------------------------------------------------------------
# PyTorch Lightning DataLoader
# -------------------------------------------------------------------------------------------------

class NAICSDataModule(pyl.LightningDataModule):
    
    def __init__(
        self,
        descriptions_path: str = './data/naics_descriptions.parquet',
        triplets_path: str = './data/naics_training_pairs.parquet',
        tokenizer_name: str = 'sentence-transformers/all-mpnet-base-v2',
        curriculum_config: Optional[Dict] = None,
        batch_size: int = 32,
        num_workers: int = 4,
        val_split: float = 0.05,
        seed: int = 42
    ):
        super().__init__()
        
        self.descriptions_path = descriptions_path
        self.triplets_path = triplets_path
        self.tokenizer_name = tokenizer_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.seed = seed
        
        curriculum_config = curriculum_config or {}
        self.curriculum = CurriculumConfig(**curriculum_config)
        
        self.token_cache = None
        self.train_dataset = None
        self.val_dataset = None
    

    def prepare_data(self):
        
        pass
    

    def setup(self, stage: Optional[str] = None):
        
        if self.token_cache is None:
            self.token_cache = TokenizationCache(
                descriptions_path=self.descriptions_path,
                tokenizer_name=self.tokenizer_name
            )
            self.token_cache.build_cache()
        
        if stage == 'fit' or stage is None:
            
            self.train_dataset = NAICSStreamingDataset(
                triplets_path=self.triplets_path,
                token_cache=self.token_cache,
                curriculum=self.curriculum,
                seed=self.seed
            )
            
            self.val_dataset = NAICSStreamingDataset(
                triplets_path=self.triplets_path,
                token_cache=self.token_cache,
                curriculum=self.curriculum,
                seed=self.seed + 1
            )
    

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, #type: ignore
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            persistent_workers=True if self.num_workers > 0 else False
        )
    

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, #type: ignore
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            persistent_workers=True if self.num_workers > 0 else False
        )
