# -------------------------------------------------------------------------------------------------
# Imports and settings
# -------------------------------------------------------------------------------------------------

import logging
import operator
from collections import defaultdict
from dataclasses import dataclass, fields, replace
from functools import reduce
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import polars as pl

from naics_gemini.data_loader.tokenization_cache import tokenization_cache
from naics_gemini.utils.utilities import get_indices_codes

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------------------------------

@dataclass
class CurriculumConfig:

    # File paths
    descriptions_parquet: str = './data/naics_descriptions.parquet'
    distances_parquet: str = './data/naics_distances.parquet'
    relations_parquet: str = './data/naics_relations.parquet'
    triplets_parquet: str = './data/naics_training_pairs'

    # Dataset creation parameters
    seed: int = 42

    excluded: Optional[bool] = None
    unrelated: Optional[bool] = None

    anchor_level: Optional[List[int]] = None
    positive_level: Optional[List[int]] = None
    negative_level: Optional[List[int]] = None

    relation_margins: Optional[List[int]] = None
    distance_margins: Optional[List[float]] = None

    positive_relation: Optional[List[int]] = None
    negative_relation: Optional[List[int]] = None

    positive_distance: Optional[List[float]] = None
    negative_distance: Optional[List[float]] = None
    
    n_positives: int = 2125
    n_negatives: int = 2125

    def items(self):
        for f in fields(self):
            if not f.name.endswith('_parquet') and f.name != 'seed':
                v = getattr(self, f.name)
                if v is not None:
                    yield f.name, v


# -------------------------------------------------------------------------------------------------
# Triplet batch generator
# -------------------------------------------------------------------------------------------------
    
def create_streaming_generator(
    curriculum: CurriculumConfig
) -> Iterator[Dict[str, Any]]:
    
    # Parameters from curriculum
    descriptions_parquet = curriculum.descriptions_parquet
    triplets_parquet = curriculum.triplets_parquet
    anchor_level = curriculum.anchor_level
    n_positives = curriculum.n_positives
    n_negatives = curriculum.n_negatives

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

    # Build filters from curriculum
    exprs = []
    for k, v in curriculum.items():

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
                            seed=curriculum.seed
                        )
        )
        .drop('positives_negatives_len')
        .explode('positives_negatives')
        .unnest('positives_negatives')
        .collect()
        .pivot(
            'filtered',
            index=['anchors', 'positives'],
            values=['negatives']
        )
        .filter(pl.col('negatives').is_not_null())
        .select(
            anchors=pl.col('anchors'),
            positives=pl.col('positives'),
            negatives=pl.col('negatives'),
            fallback=pl.when(pl.col('negatives').list.len().lt(n_negatives))
                       .then(pl.col('fallback'))
                       .otherwise(None)
        )
    )

    # Complete batches
    df_2 = (
        df_1
        .filter(
            pl.col('fallback').is_null()
        )
        .select(
            anchors=pl.col('anchors'),
            positives=pl.col('positives'),
            negatives=pl.col('negatives')
                        .list.sample(
                            n_negatives, 
                            shuffle=True, 
                            seed=curriculum.seed
                        )
                    
        )
    )

    # Completed batches with fallbacks
    df_3 = (
        df_1
        .filter(
            pl.col('fallback').is_not_null()
        )
        .with_columns(
            negatives_len=pl.lit(n_negatives)
                            .sub(pl.col('negatives').list.len()),
            fallback_len=pl.col('fallback').list.len()
        )
        .with_columns(
            sample_len=pl.min_horizontal(
                pl.col('negatives_len'),
                pl.col('fallback_len')
            )
        )
        .drop('negatives_len', 'fallback_len')
        .with_columns(
            fallback=pl.col('fallback')
                        .list.sample(
                            pl.col('sample_len'), 
                            shuffle=True, 
                            seed=curriculum.seed
                        )
        )
        .with_columns(
            negatives=pl.col('negatives')
                        .list.concat(pl.col('fallback'))
        )
        .drop('fallback', 'sample_len')
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

    logging.info(f'Number of anchors: {len(df_iter): ,}')
    logging.info(f'Number of anchors/positives: {df.height: ,}')
    logging.info(f'  w/o fallbacks: {df_2.height: ,}')
    logging.info(f'  w/ fallbacks: {df_3.height: ,}')
    logging.info(f'Number of anchors/positives/negatives: {df.explode("negatives").height: ,}')

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
    curriculum: CurriculumConfig
) -> Iterator[Dict[str, Any]]:
    
    curriculum = replace(
        CurriculumConfig(),
        anchor_level=[2, 3],
        positive_relation=[1, 2],
        excluded=False,
        unrelated=False,
        n_positives=100,
        n_negatives=20
    )

    token_cache = tokenization_cache()

    triplets_iterator = create_streaming_generator(curriculum)

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