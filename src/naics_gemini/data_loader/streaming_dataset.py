# -------------------------------------------------------------------------------------------------
# Imports and settings
# -------------------------------------------------------------------------------------------------

import logging
import operator
from collections import defaultdict
from functools import reduce
from pathlib import Path
from typing import Any, Dict, Iterator

import polars as pl

from naics_gemini.utils.config import StreamingConfig
from naics_gemini.utils.utilities import get_indices_codes

logger = logging.getLogger(__name__)


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

    # Build filters from cfg
    exprs = []
    for k, v in cfg.iter_fields():

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
    df_1 = (
        pl
        .scan_parquet(
            dataset_files
        )
        .with_columns(
            filtered=pl.when(filters)
                       .then(pl.lit('negatives'))
                       .otherwise(pl.lit('fallback'))
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
            filtered=pl.col('filtered'),
            negatives=pl.struct(
                pl.col('negative_idx'),
                pl.col('negative_code'),
                pl.col('relation_margin'),
                pl.col('distance_margin')
            )
        )
        .group_by('anchors', 'positives', 'filtered')
        .agg(
            negatives=pl.col('negatives')
        )
        .select(
            anchors=pl.col('anchors'), 
            positives_negatives=pl.struct(
                pl.col('positives'),
                pl.col('filtered'),
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
        .collect()
    )
    
    # Pivot to separate filtered (negatives) from unfiltered (fallback)
    # When we pivot on the 'filtered' boolean column, Polars names the resulting columns
    # based on the 'values' parameter combined with the boolean values.
    # Since values='negatives' and filtered is True/False, we get 'negatives' and 'fallback'
    df_1 = (
        df_1
        .pivot(
            on='filtered',
            index=['anchors', 'positives'],
            values='negatives'
        )
        .filter(pl.col('negatives').is_not_null())
        .with_columns(
            negatives_len=pl.col('negatives').list.len(),
            # Ensure fallback column exists and handle null
            fallback=pl.when(pl.col('fallback').is_not_null())
                      .then(pl.col('fallback'))
                      .otherwise(pl.lit([]).cast(pl.List(pl.Struct)))
        )
        .select(
            anchors=pl.col('anchors'),
            positives=pl.col('positives'),
            negatives=pl.col('negatives'),
            negatives_len=pl.col('negatives_len'),
            fallback=pl.col('fallback'),
            fallback_len=pl.col('fallback').list.len()
        )
    )

    # Complete batches - those with enough negatives already (>= n_negatives)
    df_2 = (
        df_1
        .filter(
            pl.col('negatives_len') >= n_negatives
        )
        .select(
            anchors=pl.col('anchors'),
            positives=pl.col('positives'),
            negatives=pl.col('negatives')
                        .list.sample(
                            n_negatives, 
                            shuffle=True,
                            with_replacement=False,
                            seed=cfg.seed
                        )
                    
        )
    )

    # Incomplete batches - need fallback sampling to reach n_negatives
    df_3 = (
        df_1
        .filter(
            pl.col('negatives_len') < n_negatives
        )
        .with_columns(
            # Calculate how many more negatives we need from fallback
            need_from_fallback=pl.lit(n_negatives).sub(pl.col('negatives_len'))
        )
        .with_columns(
            # Sample fallbacks - use with_replacement if needed, handle empty fallback
            fallback_sample=pl.when(pl.col('fallback_len') == 0)
                .then(
                    # No fallbacks available: repeat existing negatives to reach n_negatives
                    pl.col('negatives').list.sample(
                        pl.col('need_from_fallback'),
                        shuffle=True,
                        with_replacement=True,
                        seed=cfg.seed
                    )
                )
                .when(pl.col('fallback_len') >= pl.col('need_from_fallback'))
                .then(
                    # Enough fallbacks: sample without replacement
                    pl.col('fallback').list.sample(
                        pl.col('need_from_fallback'),
                        shuffle=True,
                        with_replacement=False,
                        seed=cfg.seed
                    )
                )
                .otherwise(
                    # Not enough fallbacks: sample with replacement to reach n_negatives
                    pl.col('fallback').list.sample(
                        pl.col('need_from_fallback'),
                        shuffle=True,
                        with_replacement=True,
                        seed=cfg.seed
                    )
                )
        )
        .with_columns(
            negatives=pl.col('negatives').list.concat(pl.col('fallback_sample'))
        )
        .drop('fallback', 'negatives_len', 'fallback_len', 'need_from_fallback', 'fallback_sample')
    )

    # Combine complete and completed
    df = (
        pl
        .concat([
            df_2, 
            df_3
        ])
        .select(
            batch=pl.col('anchors')
                    .rank('dense'),
            anchors=pl.col('anchors'),
            positives=pl.col('positives'),
            negatives=pl.col('negatives')
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
    logger.debug(f'  w/o fallbacks: {df_2.height: ,}')
    logger.debug(f'  w/ fallbacks: {df_3.height: ,}')
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
    """
    Create a streaming dataset generator that yields training samples.
    
    Args:
        token_cache: Pre-loaded tokenization cache (loaded once in main process)
        cfg: Streaming configuration
        
    Yields:
        Dictionary containing anchor, positive, and negative samples with embeddings
    """

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