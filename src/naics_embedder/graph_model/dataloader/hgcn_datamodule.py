# -------------------------------------------------------------------------------------------------
# Imports and settings
# -------------------------------------------------------------------------------------------------

import logging
import operator
from collections import defaultdict
from dataclasses import dataclass
from functools import reduce
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader, Dataset

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


# -------------------------------------------------------------------------------------------------
# Utility functions
# -------------------------------------------------------------------------------------------------


def _get_config_dict(cfg: Config) -> Dict[str, Any]:
    '''Extract relevant config values for filtering.'''
    keep = [
        'anchor_level',
        'relation_margin',
        'distance_margin',
        'positive_level',
        'positive_relation',
        'positive_distance',
        'negative_level',
        'negative_relation',
        'negative_distance',
        'n_positive_samples',
        'n_negatives',
    ]

    cfg_dict: Dict[str, Any] = {}
    for k in keep:
        v = getattr(cfg, k, None)
        if v is not None:
            cfg_dict[k] = v

    return cfg_dict


def _get_weighted_sample(
    df: pl.DataFrame,
    group_col: List[str],
    weight_col: str,
    n_samples: int,
    seed: Optional[int] = None,
) -> pl.DataFrame:
    '''Apply weighted sampling using Gumbel-max trick.'''
    rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()

    return (
        df.with_columns(rnd=pl.Series('rnd', rng.uniform(size=df.height)))
        .with_columns(norm_wgt=pl.col(weight_col).truediv(pl.col(weight_col).sum().over(group_col)))
        .with_columns(gm_sort=pl.col('rnd').log().mul(-1).truediv(pl.col('norm_wgt')))
        .sort('gm_sort')
        .group_by(group_col, maintain_order=True)
        .head(n_samples)
        .drop('rnd', 'norm_wgt', 'gm_sort')
    )


def _build_polars_query(cfg: Config, codes: List[str], code_to_idx: Dict[str, int]) -> pl.DataFrame:
    '''Build and execute the Polars query to get triplets.'''
    # Organize codes by level
    level_dict = defaultdict(list)
    for code in codes:
        level_dict[len(code)].append(code)  # type: ignore

    # Get list of dataset files
    if cfg.anchor_level is not None:
        dataset_files = []
        for level in cfg.anchor_level:
            for code in level_dict[level]:
                idx = code_to_idx[code]
                for pq_path in Path(f'{cfg.training_pairs_path}/anchor={idx}/').glob('*.parquet'):
                    dataset_files.append(pq_path.as_posix())
    else:
        dataset_files = [str(p) for p in Path(cfg.training_pairs_path).glob('**/*.parquet')]

    # Build filters
    cfg_dict = _get_config_dict(cfg)
    exprs = []
    for k, v in cfg_dict.items():
        if isinstance(v, list):
            exprs.append(pl.col(k).is_in(v))
        elif isinstance(v, bool):
            exprs.append(pl.col(k).eq(v))

    if not exprs:
        exprs = [pl.col('anchor_idx').ge(0)]

    filters = reduce(operator.and_, exprs)

    # Build lazy query
    logger.debug(f'Scanning {len(dataset_files)} parquet files...')
    df_0 = pl.scan_parquet(dataset_files).filter(filters)

    # Build query with sampled positives and weighted negatives
    df_1 = (
        df_0.with_columns(
            relation_margin=pl.when(pl.col('excluded'))
            .then(pl.col('relation_margin').add(1))
            .otherwise(pl.col('relation_margin')),
            distance_margin=pl.when(pl.col('excluded'))
            .then(pl.col('distance_margin').add(1))
            .otherwise(pl.col('distance_margin')),
        )
        .with_columns(sample_wgt=pl.mean_horizontal('relation_margin', 'distance_margin').pow(-1))
        .select(
            anchors=pl.struct(pl.col('anchor_idx'), pl.col('anchor_code')),
            positives=pl.struct(pl.col('positive_idx'), pl.col('positive_code')),
            negatives=pl.struct(
                pl.struct(
                    pl.col('negative_idx'),
                    pl.col('negative_code'),
                    pl.col('relation_margin'),
                    pl.col('distance_margin'),
                ).alias('negatives'),
                pl.col('sample_wgt'),
            ),
        )
        .group_by('anchors', 'positives')
        .agg(negatives=pl.col('negatives'))
        .select(
            anchors=pl.col('anchors'),
            positives_negatives=pl.struct(pl.col('positives'), pl.col('negatives')),
        )
        .group_by('anchors')
        .agg(
            positives_negatives_len=pl.col('positives_negatives').len(),
            positives_negatives=pl.col('positives_negatives'),
        )
        .with_columns(
            positives_negatives_len=pl.min_horizontal(
                pl.col('positives_negatives_len'), pl.lit(cfg.n_positive_samples)
            )
        )
        .with_columns(
            positives_negatives=pl.col('positives_negatives').list.sample(
                pl.col('positives_negatives_len'), shuffle=True, seed=cfg.seed
            )
        )
        .drop('positives_negatives_len')
        .explode('positives_negatives')
        .unnest('positives_negatives')
        .explode('negatives')
        .unnest('negatives')
    )

    # Execute query
    logger.info('Executing Polars query (this may take 30-60 seconds for large datasets)...')
    df_1 = df_1.collect()
    logger.info(f'✓ Polars query complete: {len(df_1)} rows')

    return df_1


def _apply_weighted_sampling(df_1: pl.DataFrame, cfg: Config) -> List[Dict]:
    '''Apply weighted sampling and convert to list of dicts.'''
    df = (
        _get_weighted_sample(
            df_1, ['anchors', 'positives'], 'sample_wgt', cfg.n_negatives, seed=cfg.seed
        )
        .group_by('anchors', 'positives')
        .agg(negatives=pl.col('negatives'), negatives_len=pl.col('negatives').len())
        .select(
            anchors=pl.col('anchors'),
            positives=pl.col('positives'),
            negatives=pl.col('negatives'),
            negatives_len=pl.col('negatives_len'),
        )
    )

    return df.to_dicts()


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
    max_negatives = max(len(item['negative_indices']) for item in batch)
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


# -------------------------------------------------------------------------------------------------
# Create dataloader
# -------------------------------------------------------------------------------------------------


def _load_codes_and_indices(descriptions_parquet: str) -> tuple[List[str], Dict[str, int]]:
    '''Load codes and code_to_idx mapping from parquet file.'''
    df_codes = pl.read_parquet(descriptions_parquet).select('index', 'code')
    codes = df_codes['code'].to_list()
    code_to_idx = {row['code']: row['index'] for row in df_codes.iter_rows(named=True)}
    return codes, code_to_idx


def create_dataloader(cfg: Config) -> DataLoader:
    '''
    Create a PyTorch DataLoader for graph training.

    Uses weighted sampling by inverse of relation id (similar to streaming_dataset.py).

    Args:
        cfg: Configuration object

    Returns:
        DataLoader instance
    '''
    # Load codes and indices
    codes, code_to_idx = _load_codes_and_indices(cfg.descriptions_parquet)

    # Build Polars query
    df_1 = _build_polars_query(cfg, codes, code_to_idx)

    # Apply weighted sampling
    data_list = _apply_weighted_sampling(df_1, cfg)

    # Create dataset
    dataset = TripletDataset(data_list)

    # Create dataloader
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
