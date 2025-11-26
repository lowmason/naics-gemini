# -------------------------------------------------------------------------------------------------
# Imports and settings
# -------------------------------------------------------------------------------------------------

import hashlib
import json
import logging
import operator
import pickle
import time
from collections import defaultdict
from functools import reduce
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union

import numpy as np
import polars as pl

from naics_embedder.utils.config import StreamingConfig

logger = logging.getLogger(__name__)

# Track which configurations have already been logged
_logged_configs = set()

# -------------------------------------------------------------------------------------------------
# Cache utilities
# -------------------------------------------------------------------------------------------------

def _get_cache_key(cfg: StreamingConfig) -> str:
    '''Generate a cache key from StreamingConfig.'''
    cache_dict = {
        'descriptions_parquet': str(cfg.descriptions_parquet),
        'triplets_parquet': str(cfg.triplets_parquet),
        'anchor_level': cfg.anchor_level,
        'n_positives': cfg.n_positives,
        'n_negatives': cfg.n_negatives,
        'seed': cfg.seed,
        'relation_margin': cfg.relation_margin,
        'distance_margin': cfg.distance_margin,
        'positive_level': cfg.positive_level,
        'positive_relation': cfg.positive_relation,
        'positive_distance': cfg.positive_distance,
        'negative_level': cfg.negative_level,
        'negative_relation': cfg.negative_relation,
        'negative_distance': cfg.negative_distance,
    }
    config_str = json.dumps(cache_dict, sort_keys=True)
    return hashlib.sha256(config_str.encode()).hexdigest()[:16]

def _get_final_cache_path(cfg: StreamingConfig) -> Path:
    '''Get the cache file path for the final processed data (after weighted sampling).'''
    cache_key = _get_cache_key(cfg)
    triplets_path = Path(cfg.triplets_parquet)
    cache_dir = triplets_path.parent / 'streaming_cache'
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
        logger.debug(f'Final cache saved successfully ({len(data)} rows)')
    except Exception as e:
        logger.warning(f'Failed to save final cache: {e}')

# -------------------------------------------------------------------------------------------------
# Utility functions
# -------------------------------------------------------------------------------------------------

def _get_config_dict(cfg: StreamingConfig) -> Dict[str, Any]:
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
        'n_positives',
        'n_negatives',
    ]

    cfg_dict: Dict[str, Any] = {}
    for k, v in cfg.model_dump().items():
        if k in keep and v is not None:
            cfg_dict[k] = v

    return cfg_dict

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

def _build_polars_query(
    cfg: StreamingConfig, codes: List[str], code_to_idx: Dict[str, int], worker_id: str
) -> pl.DataFrame:
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
                for pq_path in Path(f'{cfg.triplets_parquet}/anchor={idx}/').glob('*.parquet'):
                    dataset_files.append(pq_path.as_posix())
    else:
        dataset_files = [str(p) for p in Path(cfg.triplets_parquet).glob('**/*.parquet')]

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
    logger.debug(f'{worker_id} Scanning {len(dataset_files)} parquet files...')
    df_0 = pl.scan_parquet(dataset_files).filter(filters)

    # Build query with sampled positives
    df_1 = (
        df_0.with_columns(
            relation_margin=pl.when(pl.col('excluded')).then(pl.col('relation_margin').add(1)
                                                             ).otherwise(pl.col('relation_margin')),
            distance_margin=pl.when(pl.col('excluded')).then(pl.col('distance_margin').add(1)
                                                             ).otherwise(pl.col('distance_margin')),
        ).with_columns(sample_wgt=pl.mean_horizontal('relation_margin', 'distance_margin').pow(-1)
                       ).select(
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
                       ).group_by('anchors', 'positives').agg(negatives=pl.col('negatives')).select(
                           anchors=pl.col('anchors'),
                           positives_negatives=pl.struct(pl.col('positives'), pl.col('negatives')),
                       ).group_by('anchors').agg(
                           positives_negatives_len=pl.col('positives_negatives').len(),
                           positives_negatives=pl.col('positives_negatives'),
                       ).with_columns(
                           positives_negatives_len=pl.min_horizontal(
                               pl.col('positives_negatives_len'), pl.lit(cfg.n_positives)
                           )
                       ).with_columns(
                           positives_negatives=pl.col('positives_negatives').list.sample(
                               pl.col('positives_negatives_len'), shuffle=True, seed=cfg.seed
                           )
                       ).drop('positives_negatives_len').explode('positives_negatives').unnest(
                           'positives_negatives'
                       ).explode('negatives').unnest('negatives')
    )

    # Execute query
    logger.info(
        f'{worker_id} Executing Polars query (this may take 30-60 seconds for large datasets)...'
    )
    start_time = time.time()
    df_1 = df_1.collect()
    query_time = time.time() - start_time
    logger.info(f'{worker_id} ✓ Polars query complete in {query_time:.2f}s: {len(df_1)} rows')

    return df_1

def _apply_weighted_sampling(df_1: pl.DataFrame, cfg: StreamingConfig, worker_id: str) -> List[Dict]:
    '''Apply weighted sampling and convert to list of dicts.'''
    df = (
        _get_weighted_sample(
            df_1, ['anchors', 'positives'], 'sample_wgt', cfg.n_negatives, seed=cfg.seed
        ).group_by('anchors', 'positives').agg(
            negatives=pl.col('negatives'), negatives_len=pl.col('negatives').len()
        ).select(
            anchors=pl.col('anchors'),
            positives=pl.col('positives'),
            negatives=pl.col('negatives'),
            negatives_len=pl.col('negatives_len'),
        )
    )

    return df.to_dicts()

# -------------------------------------------------------------------------------------------------
# Triplet batch generator
# -------------------------------------------------------------------------------------------------

def create_streaming_generator(cfg: StreamingConfig) -> Iterator[Dict[str, Any]]:
    '''Create a generator that yields triplets for training.'''
    # Identify worker process
    worker_info = None
    try:
        import torch

        worker_info = torch.utils.data.get_worker_info()
    except Exception:
        pass

    worker_id = f'Worker {worker_info.id}' if worker_info else 'Main'

    # Try to load final cache first
    df_list = _load_final_cache(cfg)

    if df_list is None:
        # Cache doesn't exist - need to build it
        logger.info(f'{worker_id} Final cache not found, building from scratch...')

        # Load codes and indices
        codes, code_to_idx = _load_codes_and_indices(cfg.descriptions_parquet, worker_id)

        # Build Polars query
        df_1 = _build_polars_query(cfg, codes, code_to_idx, worker_id)

        # Apply weighted sampling
        df_list = _apply_weighted_sampling(df_1, cfg, worker_id)

        # Save final cache if we're in main process (prepare_data)
        if worker_info is None:
            _save_final_cache(df_list, cfg)

    # Log statistics (main process only, once per config)
    if worker_info is None:
        config_id = (
            f'{cfg.descriptions_parquet}_{cfg.triplets_parquet}_{cfg.anchor_level}_'
            f'{cfg.n_positives}_{cfg.n_negatives}_{cfg.seed}'
        )

        if config_id not in _logged_configs:
            num_anchors = len(set(row['anchors']['anchor_idx'] for row in df_list))
            num_positives = len(df_list)
            num_negatives = sum(len(row['negatives']) for row in df_list)

            logger.info(
                f'Number of anchors: {num_anchors: ,}, '
                f'anchors/positives: {num_positives: ,}, '
                f'anchors/positives/negatives: {num_negatives: ,}\n'
            )
            _logged_configs.add(config_id)

    # Iterate from list of dicts (no Polars involved)
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
            'anchor_idx': row['anchors']['anchor_idx'],
            'anchor_code': row['anchors']['anchor_code'],
            'positive_idx': row['positives']['positive_idx'],
            'positive_code': row['positives']['positive_code'],
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
            'negatives': negatives,
        }
