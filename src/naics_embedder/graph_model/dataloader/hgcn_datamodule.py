import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

from naics_embedder.graph_model.dataloader.hgcn_streaming_dataset import load_streaming_triplets
from naics_embedder.utils.config import GraphConfig, StreamingConfig

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------------------------------
# Config
# -------------------------------------------------------------------------------------------------

@dataclass
class Config:
    '''Configuration for training data loader.'''

    training_pairs_path: str = './data/naics_training_pairs.parquet'
    descriptions_parquet: str = './data/naics_descriptions.parquet'

    batch_size: int = 64
    shuffle: bool = True
    num_workers: int = 4
    pin_memory: bool = True

    n_positive_samples: int = 2048
    n_negatives: int = 32

    # Curriculum filtering parameters
    anchor_level: Optional[List[int]] = None
    relation_margin: Optional[List[int]] = None
    distance_margin: Optional[List[float]] = None
    positive_level: Optional[List[int]] = None
    positive_relation: Optional[List[int]] = None
    positive_distance: Optional[List[float]] = None
    negative_level: Optional[List[int]] = None
    negative_relation: Optional[List[int]] = None
    negative_distance: Optional[List[int]] = None

    allowed_relations: Optional[List[str]] = None
    min_code_level: Optional[int] = None
    max_code_level: Optional[int] = None

    seed: int = 42

def _infer_descriptions_path(encodings_parquet: str) -> str:
    '''Infer descriptions parquet from the encodings path, falling back to defaults.'''
    enc_path = Path(encodings_parquet).expanduser()
    candidate = enc_path.parent.parent / 'naics_descriptions.parquet'
    return str(candidate) if candidate.exists() else './data/naics_descriptions.parquet'

def _loader_cfg_from_graph(
    cfg: GraphConfig,
    descriptions_override: Optional[str] = None,
) -> Config:
    '''Derive loader config values from GraphConfig.'''
    descriptions_path = descriptions_override or _infer_descriptions_path(cfg.encodings_parquet)
    return Config(
        training_pairs_path=cfg.training_pairs_path,
        descriptions_parquet=descriptions_path,
        batch_size=cfg.batch_size,
        shuffle=cfg.shuffle,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        n_positive_samples=cfg.n_positive_samples,
        n_negatives=cfg.k_total,
        anchor_level=None,
        relation_margin=None,
        distance_margin=None,
        positive_level=None,
        positive_relation=None,
        positive_distance=None,
        negative_level=None,
        negative_relation=None,
        negative_distance=None,
        allowed_relations=cfg.allowed_relations,
        min_code_level=cfg.min_code_level,
        max_code_level=cfg.max_code_level,
        seed=cfg.seed,
    )

def _streaming_cfg_from_loader(cfg: Config) -> StreamingConfig:
    '''Convert loader config into a StreamingConfig for cache materialization.'''
    return StreamingConfig(
        descriptions_parquet=cfg.descriptions_parquet,
        triplets_parquet=cfg.training_pairs_path,
        anchor_level=cfg.anchor_level,
        relation_margin=cfg.relation_margin,
        distance_margin=cfg.distance_margin,
        positive_level=cfg.positive_level,
        positive_relation=cfg.positive_relation,
        positive_distance=cfg.positive_distance,
        negative_level=cfg.negative_level,
        negative_relation=cfg.negative_relation,
        negative_distance=cfg.negative_distance,
        n_positives=cfg.n_positive_samples,
        n_negatives=cfg.n_negatives,
        seed=cfg.seed,
    )

# -------------------------------------------------------------------------------------------------
# Dataset
# -------------------------------------------------------------------------------------------------

class TripletDataset(Dataset):
    '''Dataset for loading triplets.'''

    def __init__(self, data: List[Dict]):
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.data[idx]

        anchor_idx = row['anchors']['anchor_idx']
        positive_idx = row['positives']['positive_idx']

        negatives = row['negatives']
        negative_indices = torch.tensor(
            [neg['negative_idx'] for neg in negatives], dtype=torch.long
        )

        return {
            'anchor_idx': anchor_idx,
            'positive_idx': positive_idx,
            'negative_indices': negative_indices,
        }

def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    '''Collate function for batching.'''
    anchor_idx = torch.tensor([item['anchor_idx'] for item in batch], dtype=torch.long)
    positive_idx = torch.tensor([item['positive_idx'] for item in batch], dtype=torch.long)

    # Pad negative indices to same length
    max_negatives = max((len(item['negative_indices']) for item in batch), default=0)
    if max_negatives == 0:
        raise ValueError('Batch contains no negative indices; check triplet generation logic.')
    negative_indices = []
    for item in batch:
        negs = item['negative_indices']
        if len(negs) < max_negatives:
            # Pad with last value
            padding = negs[-1:].repeat(max_negatives - len(negs))
            negs = torch.cat([negs, padding])
        negative_indices.append(negs)

    negative_indices = torch.stack(negative_indices)

    return {
        'anchor_idx': anchor_idx,
        'positive_idx': positive_idx,
        'negative_indices': negative_indices,
    }

def create_dataloader(cfg: Config) -> DataLoader:
    '''Create a PyTorch DataLoader for graph training using the shared streaming cache.'''
    streaming_cfg = _streaming_cfg_from_loader(cfg)
    data_list = load_streaming_triplets(streaming_cfg)
    dataset = TripletDataset(data_list)

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=cfg.shuffle,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        collate_fn=collate_fn,
    )

    logger.info(
        f'Created DataLoader: {len(dataset)} samples, '
        f'batch_size={cfg.batch_size}, num_workers={cfg.num_workers}'
    )

    return dataloader

class HGCNDataModule(pl.LightningDataModule):
    '''Lightning DataModule that materializes triplets once and reuses cached data.'''

    def __init__(
        self,
        graph_cfg: GraphConfig,
        *,
        val_split: float = 0.05,
        descriptions_parquet: Optional[str] = None,
    ):
        super().__init__()
        self.graph_cfg = graph_cfg
        self.val_split = max(0.0, min(0.9, val_split))
        self.loader_cfg = _loader_cfg_from_graph(
            graph_cfg, descriptions_override=descriptions_parquet
        )
        self._streaming_cfg = _streaming_cfg_from_loader(self.loader_cfg)
        self._materialized_triplets: Optional[List[Dict[str, Any]]] = None
        self._train_dataset: Optional[TripletDataset] = None
        self._val_dataset: Optional[TripletDataset] = None

    def prepare_data(self) -> None:
        # Ensure the expensive Polars query runs once on rank 0.
        self._materialized_triplets = load_streaming_triplets(self._streaming_cfg)

    def setup(self, stage: Optional[str] = None) -> None:
        if self._train_dataset is not None and (
            self._val_dataset is not None or self.val_split == 0
        ):
            return

        data = self._materialized_triplets or load_streaming_triplets(self._streaming_cfg)
        if self.val_split <= 0 or len(data) < 2:
            train_data = data
            val_data: Optional[List[Dict[str, Any]]] = None
        else:
            split_idx = int(len(data) * (1 - self.val_split))
            split_idx = max(1, min(len(data) - 1, split_idx))
            train_data = data[:split_idx]
            val_data = data[split_idx:]

        self._train_dataset = TripletDataset(train_data)
        self._val_dataset = TripletDataset(val_data) if val_data else None

        logger.info(
            'Split triplets into %s train / %s val samples',
            len(self._train_dataset),
            len(self._val_dataset) if self._val_dataset else 0,
        )

    def train_dataloader(self) -> DataLoader:
        if self._train_dataset is None:
            raise RuntimeError(
                'DataModule.setup() must be called before requesting train_dataloader.'
            )

        return DataLoader(
            self._train_dataset,
            batch_size=self.loader_cfg.batch_size,
            shuffle=self.loader_cfg.shuffle,
            num_workers=self.loader_cfg.num_workers,
            pin_memory=self.loader_cfg.pin_memory,
            collate_fn=collate_fn,
        )

    def val_dataloader(self) -> Optional[DataLoader]:
        if self._val_dataset is None:
            return None

        return DataLoader(
            self._val_dataset,
            batch_size=self.loader_cfg.batch_size,
            shuffle=False,
            num_workers=self.loader_cfg.num_workers,
            pin_memory=self.loader_cfg.pin_memory,
            collate_fn=collate_fn,
        )
