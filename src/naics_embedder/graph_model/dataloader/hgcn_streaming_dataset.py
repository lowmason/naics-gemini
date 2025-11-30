# -------------------------------------------------------------------------------------------------
# Imports and settings
# -------------------------------------------------------------------------------------------------

import hashlib
import json
import logging
import pickle
import random
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union

import numpy as np
import polars as pl

from naics_embedder.data.positive_sampling import create_positive_sampler
from naics_embedder.utils.config import StreamingConfig
from naics_embedder.utils.utilities import get_indices_codes

logger = logging.getLogger(__name__)

# Track which configurations have already been logged
_logged_configs: set = set()

def _log_dataset_stats(df_list: List[Dict], *, worker_id: str, cfg: StreamingConfig) -> None:
    '''Log dataset statistics once per unique configuration.'''
    config_id = f'{cfg.descriptions_parquet}_{cfg.relations_parquet}_{cfg.n_negatives}_{cfg.seed}'

    if config_id in _logged_configs:
        return

    num_anchors = len(set(row['anchor_idx'] for row in df_list))
    num_positives = len(df_list)
    num_negatives = sum(len(row.get('negatives', [])) for row in df_list)

    logger.info(
        f'{worker_id} Dataset stats — anchors: {num_anchors:,}, '
        f'positives: {num_positives:,}, negatives: {num_negatives:,}'
    )
    _logged_configs.add(config_id)

# -------------------------------------------------------------------------------------------------
# Cache utilities
# -------------------------------------------------------------------------------------------------

def _get_cache_key(cfg: StreamingConfig) -> str:
    '''Generate a cache key from StreamingConfig.'''
    cache_dict = {
        'descriptions_parquet': str(cfg.descriptions_parquet),
        'relations_parquet': str(cfg.relations_parquet),
        'n_negatives': cfg.n_negatives,
        'seed': cfg.seed,
    }
    config_str = json.dumps(cache_dict, sort_keys=True)
    return hashlib.sha256(config_str.encode()).hexdigest()[:16]

def _get_final_cache_path(cfg: StreamingConfig) -> Path:
    '''Get the cache file path for the final processed data (after weighted sampling).'''
    cache_key = _get_cache_key(cfg)
    descriptions_path = Path(cfg.descriptions_parquet)
    cache_dir = descriptions_path.parent / 'streaming_cache'
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f'streaming_final_{cache_key}.pkl'

def _load_final_cache(cfg: StreamingConfig) -> Optional[List[Dict]]:
    '''Load final cached data if it exists.'''
    cache_path = _get_final_cache_path(cfg)

    if not cache_path.exists():
        return None

    try:
        with open(cache_path, 'rb') as f:
            data_list = pickle.load(f)
        logger.info(f'Loaded streaming cache from {cache_path}')
        return data_list
    except Exception as e:
        logger.warning(f'Failed to load final cache: {e}, will recompute')
        return None

def _save_final_cache(data: List[Dict], cfg: StreamingConfig) -> None:
    '''Save final processed data (after weighted sampling) to cache.'''
    cache_path = _get_final_cache_path(cfg)

    try:
        # Save to temp file first, then rename (atomic operation)
        temp_path = cache_path.with_suffix('.tmp')
        with open(temp_path, 'wb') as f:
            pickle.dump(data, f)

        # Atomic rename
        temp_path.replace(cache_path)
        logger.info(f'Saved streaming cache to {cache_path}')
    except Exception as e:
        logger.warning(f'Failed to save final cache: {e}')

# -------------------------------------------------------------------------------------------------
# Utility functions
# -------------------------------------------------------------------------------------------------

def _get_weighted_sample(
    df: pl.DataFrame,
    group_col: Union[str, List[str]],
    weight_col: str,
    n_samples: int,
    seed: Optional[int] = None,
) -> pl.DataFrame:
    '''Apply weighted sampling using Gumbel-max trick.'''
    rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()

    return (
        df.with_columns(rnd=pl.Series('rnd', rng.uniform(size=df.height))).with_columns(
            norm_wgt=pl.col(weight_col).truediv(pl.col(weight_col).sum().over(group_col))
        ).with_columns(gm_sort=pl.col('rnd').log().mul(-1).truediv(pl.col('norm_wgt'))).sort(
            'gm_sort'
        ).group_by(group_col,
                   maintain_order=True).head(n_samples).drop('rnd', 'norm_wgt', 'gm_sort')
    )

def _load_codes_and_indices(descriptions_parquet: str,
                            worker_id: str) -> tuple[List[str], Dict[str, int]]:
    '''Load codes and code_to_idx mapping from cache or parquet.'''
    cache_dir = Path(descriptions_parquet).parent / 'codes_cache'
    codes_cache_path = cache_dir / 'codes_indices.pkl'

    if codes_cache_path.exists():
        with open(codes_cache_path, 'rb') as f:
            cached_data = pickle.load(f)
        codes = cached_data['codes']
        code_to_idx = cached_data['code_to_idx']
        return codes, code_to_idx
    else:
        # Fallback: read from parquet
        logger.warning(f'{worker_id} Cache not found, reading from parquet (this may be slow)...')
        df_codes = pl.read_parquet(descriptions_parquet).select('index', 'code')
        codes = df_codes['code'].to_list()
        code_to_idx = {row['code']: row['index'] for row in df_codes.iter_rows(named=True)}
        return codes, code_to_idx

def _load_negative_candidates(triplets_parquet: str,
                              ) -> Dict[tuple[int, int], List[Dict[str, Any]]]:
    '''Load all candidate negatives from triplets parquet files.

    Returns a dictionary mapping (anchor_idx, positive_idx) -> list of negative dicts.
    '''
    logger.info('Loading negative candidates from triplets parquet...')

    dataset_files = [str(p) for p in Path(triplets_parquet).glob('**/*.parquet')]
    if not dataset_files:
        logger.warning(f'No parquet files found in {triplets_parquet}')
        return {}

    df = (
        pl.scan_parquet(dataset_files).select(
            'anchor_idx',
            'positive_idx',
            'negative_idx',
            'negative_code',
            'relation_margin',
            'distance_margin',
        ).collect()
    )

    logger.info(f'Loaded {len(df):,} negative candidate rows')

    # Group by (anchor, positive)
    result: Dict[tuple[int, int], List[Dict[str, Any]]] = {}
    for row in df.iter_rows(named=True):
        key = (row['anchor_idx'], row['positive_idx'])
        if key not in result:
            result[key] = []
        result[key].append(
            {
                'negative_idx': row['negative_idx'],
                'negative_code': row['negative_code'],
                'relation_margin': row['relation_margin'],
                'distance_margin': row['distance_margin'],
            }
        )

    logger.info(f'Grouped into {len(result):,} (anchor, positive) pairs')
    return result

# -------------------------------------------------------------------------------------------------
# Triplet batch generator
# -------------------------------------------------------------------------------------------------

def load_streaming_triplets(
    cfg: StreamingConfig,
    *,
    worker_id: str = 'Main',
    allow_cache_save: bool = True,
    log_stats: bool = True,
) -> List[Dict[str, Any]]:
    '''
    Materialize the cached streaming triplets for reuse outside of DataLoader workers.

    Uses taxonomy-based stratified positive sampling:
    - Stratum 0 (descendants): for levels 2-5, next-level descendants
    - Stratum 1 (ancestors): for levels 3-6, parent codes up to level 2
    - Stratum 2 (siblings): codes sharing the same parent

    Samples up to 4 positives per stratum, then samples negatives for each.
    '''
    df_list = _load_final_cache(cfg)

    if df_list is not None:
        if log_stats:
            _log_dataset_stats(df_list, worker_id=worker_id, cfg=cfg)
        return df_list

    logger.info(f'{worker_id} Final cache not found, building from scratch...')

    # Load codes and indices
    idx_to_code_raw = get_indices_codes('idx_to_code')
    assert isinstance(idx_to_code_raw, dict), 'idx_to_code must be a dict'
    idx_to_code: Dict[int, str] = idx_to_code_raw  # type: ignore

    # Create positive sampler
    logger.info(f'{worker_id} Creating positive sampler...')
    positive_sampler = create_positive_sampler(
        descriptions_parquet=cfg.descriptions_parquet,
        relations_parquet=cfg.relations_parquet,
        max_per_stratum=4,
        seed=cfg.seed,
    )

    # Load negative candidates
    logger.info(f'{worker_id} Loading negative candidates...')
    negative_candidates = _load_negative_candidates(cfg.triplets_parquet)

    # Build triplet list
    df_list = []
    rng = random.Random(cfg.seed)

    for anchor_idx in positive_sampler.anchors:
        anchor_code = idx_to_code.get(anchor_idx)
        if anchor_code is None:
            continue

        # Sample positives across strata
        positives = positive_sampler.sample_positives(anchor_idx)
        if not positives:
            continue

        for positive in positives:
            positive_idx = positive['positive_idx']
            positive_code = positive['positive_code']

            # Get candidate negatives
            key = (anchor_idx, positive_idx)
            candidates = negative_candidates.get(key, [])

            if not candidates:
                continue

            # Sample negatives uniformly
            n_sample = min(cfg.n_negatives, len(candidates))
            sampled_negatives = rng.sample(candidates, n_sample)

            df_list.append(
                {
                    'anchor_idx': anchor_idx,
                    'anchor_code': anchor_code,
                    'positive_idx': positive_idx,
                    'positive_code': positive_code,
                    'positive_level': positive['positive_level'],
                    'stratum_id': positive['stratum_id'],
                    'stratum_wgt': positive['stratum_wgt'],
                    'negatives': sampled_negatives,
                }
            )

    logger.info(f'{worker_id} Built {len(df_list):,} triplet rows')

    if allow_cache_save:
        _save_final_cache(df_list, cfg)

    if log_stats:
        _log_dataset_stats(df_list, worker_id=worker_id, cfg=cfg)

    return df_list

def create_streaming_generator(cfg: StreamingConfig) -> Iterator[Dict[str, Any]]:
    '''Create a generator that yields triplets for training.'''
    worker_info = None
    try:
        import torch

        worker_info = torch.utils.data.get_worker_info()
    except Exception:
        pass

    worker_id = f'Worker {worker_info.id}' if worker_info else 'Main'

    df_list = load_streaming_triplets(
        cfg,
        worker_id=worker_id,
        allow_cache_save=worker_info is None,
        log_stats=worker_info is None,
    )

    for row in df_list:
        negatives = [
            {
                'negative_idx': neg['negative_idx'],
                'negative_code': neg['negative_code'],
                'relation_margin': neg['relation_margin'],
                'distance_margin': neg['distance_margin'],
            } for neg in row['negatives']
        ]

        yield {
            'anchor_idx': row['anchor_idx'],
            'anchor_code': row['anchor_code'],
            'positive_idx': row['positive_idx'],
            'positive_code': row['positive_code'],
            'positive_level': row.get('positive_level'),
            'stratum_id': row.get('stratum_id'),
            'stratum_wgt': row.get('stratum_wgt'),
            'negatives': negatives,
        }

# -------------------------------------------------------------------------------------------------
# Streaming dataset generator
# -------------------------------------------------------------------------------------------------

def create_streaming_dataset(token_cache: Dict[int, Dict[str, Any]],
                             cfg: StreamingConfig) -> Iterator[Dict[str, Any]]:
    '''Create streaming dataset that yields triplets with tokenized embeddings.'''
    # Get triplets iterator
    triplets_iterator = create_streaming_generator(cfg)

    # Yield triplets with tokenized embeddings
    for triplets in triplets_iterator:
        anchor_idx = triplets['anchor_idx']
        anchor_code = triplets['anchor_code']
        try:
            anchor_embedding = {k: v for k, v in token_cache[anchor_idx].items() if k != 'code'}
        except KeyError as e:
            logger.error(f'KeyError accessing token_cache[{anchor_idx}]: {e}')
            raise

        positive_idx = triplets['positive_idx']
        positive_code = triplets['positive_code']
        positive_embedding = {k: v for k, v in token_cache[positive_idx].items() if k != 'code'}

        negatives = []
        for negative in triplets['negatives']:
            negative_idx = negative['negative_idx']
            negative_code = negative['negative_code']
            negative_embedding = {k: v for k, v in token_cache[negative_idx].items() if k != 'code'}

            negatives.append(
                {
                    'negative_idx': negative_idx,
                    'negative_code': negative_code,
                    'negative_embedding': negative_embedding,
                    'relation_margin': negative['relation_margin'],
                    'distance_margin': negative['distance_margin'],
                }
            )

        yield {
            'anchor_idx': anchor_idx,
            'anchor_code': anchor_code,
            'anchor_embedding': anchor_embedding,
            'positive_idx': positive_idx,
            'positive_code': positive_code,
            'positive_embedding': positive_embedding,
            'positive_level': triplets.get('positive_level'),
            'stratum_id': triplets.get('stratum_id'),
            'stratum_wgt': triplets.get('stratum_wgt'),
            'negatives': negatives,
        }
