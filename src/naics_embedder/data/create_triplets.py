# -------------------------------------------------------------------------------------------------
# Imports and settings
# -------------------------------------------------------------------------------------------------

import logging
import shutil
from pathlib import Path
from typing import Tuple

import polars as pl

from naics_embedder.utils.config import TripletsConfig, load_config
from naics_embedder.utils.console import log_table as _log_table
from naics_embedder.utils.utilities import parquet_stats as _parquet_stats

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------------------------------
# Input
# -------------------------------------------------------------------------------------------------

def _input_parquet_files(descriptions_parquet: str, distances_parquet: str,
                         relations_parquet: str) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    descriptions = pl.read_parquet(descriptions_parquet)

    distances = pl.read_parquet(distances_parquet).select(
        idx_i=pl.col('idx_i'),
        idx_j=pl.col('idx_j'),
        code_i=pl.col('code_i'),
        code_j=pl.col('code_j'),
        distance=pl.col('distance'),
    )

    relations = pl.read_parquet(relations_parquet).select(
        idx_i=pl.col('idx_i'),
        idx_j=pl.col('idx_j'),
        code_i=pl.col('code_i'),
        code_j=pl.col('code_j'),
        relation=pl.col('relation_id'),
    )

    logger.info('Number of input observations:')
    logger.info(f'  descriptions: {descriptions.height: ,}')
    logger.info(f'  distances: {distances.height: ,}\n')

    return descriptions, distances, relations

# -------------------------------------------------------------------------------------------------
# Distances
# -------------------------------------------------------------------------------------------------

def _get_distances(distances_df: pl.DataFrame) -> Tuple[pl.DataFrame, pl.DataFrame]:
    positive_distances = (
        distances_df.filter(pl.col('distance').ne(pl.col('distance').max())).select(
            anchor_code=pl.col('code_i'),
            positive_code=pl.col('code_j'),
            positive_distance=pl.col('distance'),
        ).unique().sort('anchor_code', 'positive_code')
    )

    negative_distances = (
        distances_df.select(
            anchor_code=pl.col('code_i'),
            negative_code=pl.col('code_j'),
            negative_distance=pl.col('distance'),
        ).unique().sort('anchor_code', 'negative_code')
    )

    logger.info('Number of distances:')
    logger.info(f'  positives: {positive_distances.height: ,}')
    logger.info(f'  negatives: {negative_distances.height: ,}\n')

    return positive_distances, negative_distances

# -------------------------------------------------------------------------------------------------
# Relationships
# -------------------------------------------------------------------------------------------------

def _get_relations(relations_df: pl.DataFrame,
                   distances_df: pl.DataFrame) -> Tuple[pl.DataFrame, pl.DataFrame]:
    positive_relations_df = (
        relations_df.select(
            anchor_code=pl.col('code_i'),
            positive_code=pl.col('code_j'),
            positive_relation=pl.col('relation'),
        ).unique().sort('anchor_code', 'positive_code')
    )

    negative_relations_df = (
        relations_df.select(
            anchor_code=pl.col('code_i'),
            negative_code=pl.col('code_j'),
            negative_relation=pl.col('relation'),
        ).unique().sort('anchor_code', 'negative_code')
    )

    positive_distances_df, negative_distances_df = _get_distances(distances_df)

    positive_relations = positive_relations_df.join(
        positive_distances_df, how='inner', on=['anchor_code', 'positive_code']
    )

    negative_relations = negative_relations_df.join(
        negative_distances_df, how='inner', on=['anchor_code', 'negative_code']
    )

    logger.info('Number of relationships:')
    logger.info(f'  positives: {positive_relations.height: ,}')
    logger.info(f'  negatives: {negative_relations.height: ,}\n')

    return positive_relations, negative_relations

# -------------------------------------------------------------------------------------------------
# Pairs
# -------------------------------------------------------------------------------------------------

def _get_pairs(distances_df: pl.DataFrame) -> Tuple[pl.DataFrame, pl.DataFrame]:
    positives = (
        distances_df.filter(pl.col('distance').ne(pl.col('distance').max())).select(
            anchor_idx=pl.col('idx_i'),
            positive_idx=pl.col('idx_j'),
            anchor_code=pl.col('code_i'),
            positive_code=pl.col('code_j'),
        ).unique().sort('anchor_code', 'positive_code')
    )

    negatives = (
        distances_df.select(
            positive_idx=pl.col('idx_i'),
            negative_idx=pl.col('idx_j'),
            positive_code=pl.col('code_i'),
            negative_code=pl.col('code_j'),
            distance=pl.col('distance'),
        ).select(
            negative_idx=pl.col('negative_idx'),
            positive_code=pl.col('positive_code'),
            negative_code=pl.col('negative_code'),
        ).unique().sort('positive_code', 'negative_code')
    )

    logger.info('Number of pairs:')
    logger.info(f'  positives: {positives.height: ,}')
    logger.info(f'  negatives: {negatives.height: ,}\n')

    return positives, negatives

# -------------------------------------------------------------------------------------------------
# Triplets
# -------------------------------------------------------------------------------------------------

def _get_triplets(
    positives_df: pl.DataFrame,
    negatives_df: pl.DataFrame,
    positive_distances_df: pl.DataFrame,
    negative_distances_df: pl.DataFrame,
) -> pl.DataFrame:
    triplets = positives_df.join(negatives_df, how='inner', on='positive_code')
    triplets = triplets.join(
        positive_distances_df, how='inner', on=['anchor_code', 'positive_code']
    )

    triplets = (
        triplets.join(negative_distances_df, how='inner', on=['anchor_code', 'negative_code'])
        .filter(pl.col('positive_distance').gt(0.0)).with_columns(
            unrelated=pl.max_horizontal(pl.col('positive_distance'),
                                        pl.col('negative_distance')).eq(99.0),
            excluded=pl.min_horizontal(pl.col('positive_distance'), pl.col('negative_distance')).eq(
                0.0
            ),
        ).with_columns(
            relation_margin=pl.col('negative_relation').sub(pl.col('positive_relation')),
            distance_margin=pl.col('negative_distance').sub(pl.col('positive_distance')),
        ).with_columns(
            relation_margin=pl.when(pl.col('negative_distance').eq(0.0)).then(pl.lit(0.1)).when(
                pl.col('negative_distance').eq(99.0)
            ).then(pl.lit(15.0)).otherwise(pl.col('relation_margin').cast(pl.Float64)),
            distance_margin=pl.when(pl.col('negative_distance').eq(0.0)).then(pl.lit(0.1)).when(
                pl.col('relation_margin').gt(0),
                pl.col('distance_margin').eq(0.0)
            ).then(0.3333).when(
                pl.col('relation_margin').gt(0),
                pl.col('distance_margin').eq(-0.5)
            ).then(0.6667).when(pl.col('negative_distance').eq(99.0)).then(pl.lit(10.0)).otherwise(
                pl.col('distance_margin').cast(pl.Float64)
            ),
        ).filter(pl.col('relation_margin').gt(0),
                 pl.col('distance_margin').gt(0)).with_columns(
                     margin=pl.sum_horizontal(
                         pl.col('relation_margin').mul(0.3333),
                         pl.col('distance_margin').mul(0.6667)
                     ).pow(-1)
                 )
    )

    triplets_anti = (
        triplets.filter(pl.col('unrelated')).group_by('anchor_idx', 'positive_idx').agg(
            negatives=pl.col('negative_idx')
        ).with_columns(sample=pl.col('negatives').list.sample(100, shuffle=True)).select(
            anchor_idx=pl.col('anchor_idx'),
            positive_idx=pl.col('positive_idx'),
            negative_idx=pl.col('negatives').list.set_difference(pl.col('sample')),
        ).explode('negative_idx')
    )

    triplets = triplets.join(
        triplets_anti, how='anti', on=['anchor_idx', 'positive_idx', 'negative_idx']
    )

    triplets = (
        triplets.with_columns(
            anchor_level=pl.col('anchor_code').str.len_chars(),
            positive_level=pl.col('positive_code').str.len_chars(),
            negative_level=pl.col('negative_code').str.len_chars(),
        ).sort('anchor_idx', 'positive_idx', 'negative_idx').select(
            'anchor_idx',
            'positive_idx',
            'negative_idx',
            'anchor_code',
            'positive_code',
            'negative_code',
            'anchor_level',
            'positive_level',
            'negative_level',
            'excluded',
            'unrelated',
            'relation_margin',
            'distance_margin',
            'margin',
            'positive_relation',
            'negative_relation',
            'positive_distance',
            'negative_distance',
        )
    )

    logger.info('Number of:')
    logger.info(f'  triplets: {triplets.height: ,}')
    logger.info(f'  unique anchors: {triplets.get_column("anchor_idx").unique().len()}\n')
    logger.info(f'  unique positives: {triplets.get_column("positive_idx").unique().len()}\n')
    logger.info(f'  unique negatives: {triplets.get_column("negative_idx").unique().len()}\n')

    return triplets

# -------------------------------------------------------------------------------------------------
# Triplet stats
# -------------------------------------------------------------------------------------------------

def _triplet_stats(triplets_df: pl.DataFrame):
    stats_df = (
        triplets_df.group_by('relation_margin', 'distance_margin',
                             'margin').agg(cnt=pl.len(), ).with_columns(
                                 pct=pl.col('cnt').truediv(pl.col('cnt').sum()).mul(100)
                             ).sort('relation_margin', 'distance_margin', 'margin')
    )

    rels = stats_df.get_column('relation_margin').unique().sort().to_list()
    dists = stats_df.get_column('distance_margin').unique().sort().to_list()
    margin = stats_df.get_column('margin').unique().sort().to_list()

    logger.info(
        'Observed relation margins (differences in positive and negative relations): '
        f'{", ".join(str(r) for r in rels)}\n'
    )

    logger.info(
        'Observed distance margins (differences in positive and negative distances): '
        f'{", ".join(str(d) for d in dists)}\n'
    )

    logger.info(f'Observed margin weights: {", ".join(str(m) for m in margin)}\n')

    _log_table(
        df=stats_df,
        title='Triplet Statistics',
        headers=['Margins:Relation', 'Margins:Distance', 'Margins:Weightscnt', 'pct'],
        logger=logger,
        output='./outputs/triplets_stats.pdf',
    )

# -------------------------------------------------------------------------------------------------
# Generate triplets
# -------------------------------------------------------------------------------------------------

def generate_training_triplets() -> pl.DataFrame:
    # Load configuration from YAML
    cfg = load_config(TripletsConfig, './data/triplets.yaml')

    logger.info('Configuration:')
    logger.info(cfg.model_dump_json(indent=2))
    logger.info('')

    # Load data
    descriptions, distances, relations = _input_parquet_files(
        cfg.descriptions_parquet, cfg.distances_parquet, cfg.relations_parquet
    )

    # All positive and negative distances
    positive_distances, negative_distances = _get_relations(relations, distances)

    # All positive and negative pairs
    positives, negatives = _get_pairs(distances)

    # Combine positives and negatives into triplets
    triplets_df = _get_triplets(positives, negatives, positive_distances, negative_distances)

    _triplet_stats(triplets_df)

    _parquet_stats(
        parquet_df=triplets_df,
        message='NAICS triplets written to',
        output_parquet=cfg.output_parquet,
        logger=logger,
    )

    output_path = Path(cfg.output_parquet)
    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    (
        triplets_df.with_columns(anchor=pl.col('anchor_idx')).write_parquet(
            cfg.output_parquet, use_pyarrow=True, pyarrow_options={'partition_cols': ['anchor']}
        )
    )

    return triplets_df

# -------------------------------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    generate_training_triplets()
