# -------------------------------------------------------------------------------------------------
# Imports and settings
# -------------------------------------------------------------------------------------------------

import logging
import operator
from collections import defaultdict
from functools import reduce
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union

import numpy as np
import polars as pl

from naics_embedder.utils.config import StreamingConfig
from naics_embedder.utils.utilities import get_indices_codes

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------------------------------------
# Utility functions
# -------------------------------------------------------------------------------------------------

def _get_config_dict(cfg: StreamingConfig) -> Dict[str, Any]:

    keep = [
        'anchor_level', 'relation_margin', 'distance_margin', 
        'positive_level', 'positive_relation', 'positive_distance', 
        'negative_level', 'negative_relation', 'negative_distance', 
        'n_positives', 'n_negatives'
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
    seed: Optional[int] = None
):
    
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    df_len = df.height
    
    df = (
        df
        .with_columns(
            rnd=pl.Series('rnd', rng.uniform(size=df_len))
        )
    )
    
    return (
        df
        .with_columns(
            norm_wgt=pl.col(weight_col)
                       .truediv(pl.col(weight_col).sum().over(group_col))
        )
        .with_columns(
            gm_sort=pl.col('rnd').log().mul(-1)
                      .truediv(pl.col('norm_wgt'))
        )
        .sort('gm_sort')
        .group_by(group_col, maintain_order=True)
        .head(n_samples)
        .drop('rnd', 'norm_wgt', 'gm_sort')
    )


# -------------------------------------------------------------------------------------------------
# Triplet batch generator
# -------------------------------------------------------------------------------------------------
    
def create_streaming_generator(
    cfg: StreamingConfig
) -> Iterator[Dict[str, Any]]:
    
    # Parameters from StreamingConfig
    descriptions_parquet = cfg.descriptions_parquet
    triplets_parquet = cfg.triplets_parquet
    anchor_level = cfg.anchor_level
    n_positives = cfg.n_positives
    n_negatives = cfg.n_negatives

    # Get all codes and code to index mapping
    codes = get_indices_codes(descriptions_parquet, return_type='codes')
    code_to_idx = get_indices_codes(descriptions_parquet, return_type='code_to_idx')

    # Organize codes by level
    level_dict = defaultdict(list)
    for code in codes:
        level = len(code) # type: ignore
        level_dict[level].append(code)

    # Get list of dataset files
    if anchor_level is not None:
        dataset_files = []   
        for level in anchor_level:
            for code in level_dict[level]:
                idx = code_to_idx[code]
                for pq_path in Path(f'{triplets_parquet}/anchor={idx}/').glob('*.parquet'):
                    dataset_files.append(pq_path.as_posix())

    else:
        dataset_files = []
        for pq_path in Path(f'{triplets_parquet}/').glob('**/*.parquet'):
            dataset_files.append(pq_path.as_posix())

    cfg_dict = _get_config_dict(cfg)

    # Build filters from cfg_dict
    exprs = []
    for k, v in cfg_dict.items():

        if isinstance(v, list):
            exprs.append(
                pl.col(k).is_in(v)
            )       

        if isinstance(v, bool):
            exprs.append(
                pl.col(k).eq(v)
            )

    if not exprs:
        exprs = [pl.col('anchor_idx').ge(0)]

    filters = reduce(operator.and_, exprs)

    # Build (anchors, positives, negatives, fallbacks) dataframe 
    df_0 = (
        pl
        .scan_parquet(
            dataset_files
        )
        .filter(
            filters
        )
    )

    # Build final dataframe with sampled positives
    df_1 = (
        df_0
        .with_columns(
            relation_margin=pl.when(pl.col('excluded'))
                            .then(pl.col('relation_margin').add(1))
                            .otherwise(pl.col('relation_margin')),
            distance_margin=pl.when(pl.col('excluded'))
                            .then(pl.col('distance_margin').add(1))
                            .otherwise(pl.col('distance_margin'))
        )
        .with_columns(
            sample_wgt=pl.mean_horizontal('relation_margin', 'distance_margin')
                        .pow(-1)
        )
        .select(
            anchors=pl.struct(
                pl.col('anchor_idx'),
                pl.col('anchor_code')
            ),
            positives=pl.struct(
                pl.col('positive_idx'),
                pl.col('positive_code')
            ),
            negatives=pl.struct(
                pl.struct(
                    pl.col('negative_idx'),
                    pl.col('negative_code'),
                    pl.col('relation_margin'),
                    pl.col('distance_margin')
                ).alias('negatives'),
                pl.col('sample_wgt')
            ),
        )
        .group_by('anchors', 'positives')
        .agg(
            negatives=pl.col('negatives')
        )
        .select(
            anchors=pl.col('anchors'), 
            positives_negatives=pl.struct(
                pl.col('positives'),
                pl.col('negatives')
            )
        )
        .group_by('anchors')
        .agg(
            positives_negatives_len=pl.col('positives_negatives').len(),
            positives_negatives=pl.col('positives_negatives')
        )
        .with_columns(
            positives_negatives_len=pl.min_horizontal(
                pl.col('positives_negatives_len'),
                pl.lit(n_positives)
            )
        )
        .with_columns(
            positives_negatives=pl.col('positives_negatives')
                        .list.sample(
                            pl.col('positives_negatives_len'), 
                            shuffle=True, 
                            seed=cfg.seed
                        )
        )
        .drop('positives_negatives_len')
        .explode('positives_negatives')
        .unnest('positives_negatives')
        .explode('negatives')
        .unnest('negatives')
        .collect()
    )

    # Build final dataframe with sampled positives and negatives
    df = (
        _get_weighted_sample(
            df_1,
            ['anchors', 'positives'],
            'sample_wgt',
            n_negatives,
            seed=cfg.seed
        )
        .group_by('anchors', 'positives')
        .agg(
            negatives=pl.col('negatives'),
            negatives_len=pl.col('negatives').len()
        )
        .sort('anchors')
        .select(
            batch=pl.col('anchors')
                    .rank('dense'),
            anchors=pl.col('anchors'),
            positives=pl.col('positives'),
            negatives=pl.col('negatives'),
            negatives_len=pl.col('negatives_len')
        )
    )

    # Create iterator and dictionary of dataframes by batch
    df_dict = (
        df
        .sort('batch')
        .partition_by('batch', include_key=False, as_dict=True)
    )

    df_iter = [k[0] for k in df_dict.keys()]
    df_dict = {k[0]: v for k, v in df_dict.items()}

    # Log dataset statistics only at DEBUG level to reduce overhead
    logger.debug(f'Number of anchors: {len(df_iter): ,}')
    logger.debug(f'Number of anchors/positives: {df.height: ,}')
    logger.debug(f'Number of anchors/positives/negatives: {df.explode("negatives").height: ,}')

    for anchor in df_iter:
        
        anchor_iter = (
            df_dict[anchor]
            .iter_rows(named=True)
        )

        for row in anchor_iter:
            grouped = {}
            key = (row['anchors']['anchor_idx'], row['positives']['positive_idx'])

            negatives = []
            for negative in row['negatives']:
                negative_idx = negative['negative_idx']
                negative_code = negative['negative_code']
                relation_margin = negative['relation_margin']
                distance_margin = negative['distance_margin']

                negatives.append({
                    'negative_idx': negative_idx,
                    'negative_code': negative_code,
                    'relation_margin': relation_margin,
                    'distance_margin': distance_margin
                })

            if key not in grouped:
                grouped[key] = {
                    'anchor_idx': row['anchors']['anchor_idx'],
                    'anchor_code': row['anchors']['anchor_code'],
                    'positive_idx': row['positives']['positive_idx'],
                    'positive_code': row['positives']['positive_code'],
                    'negatives': negatives
                }

            yield grouped[key]

            
# -------------------------------------------------------------------------------------------------
# Streaming dataset generator
# -------------------------------------------------------------------------------------------------

def create_streaming_dataset(
    token_cache: Dict[int, Dict[str, Any]],
    cfg: StreamingConfig
) -> Iterator[Dict[str, Any]]:

    triplets_iterator = create_streaming_generator(cfg)

    for triplets in triplets_iterator:
        
        anchor_idx = triplets['anchor_idx']
        anchor_code = triplets['anchor_code']
        anchor_embedding = {k: v for k, v in token_cache[anchor_idx].items() if k != 'code'}

        positive_idx = triplets['positive_idx']
        positive_code = triplets['positive_code']
        positive_embedding = {k: v for k, v in token_cache[positive_idx].items() if k != 'code'}
        
        negatives = []
        for negative in triplets['negatives']:

            negative_idx = negative['negative_idx']
            negative_code = negative['negative_code']
            negative_embedding = {k: v for k, v in token_cache[negative_idx].items() if k != 'code'}
            
            relation_margin = negative['relation_margin']
            distance_margin = negative['distance_margin']

            negatives.append({
                'negative_idx': negative_idx,
                'negative_code': negative_code,
                'negative_embedding': negative_embedding,
                'relation_margin': relation_margin,
                'distance_margin': distance_margin
            })

        yield {
            'anchor_idx': anchor_idx,
            'anchor_code': anchor_code,
            'anchor_embedding': anchor_embedding,
            'positive_idx': positive_idx,
            'positive_code': positive_code,
            'positive_embedding': positive_embedding,
            'negatives': negatives
        }