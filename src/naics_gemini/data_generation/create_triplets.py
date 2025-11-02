import json
import logging
from dataclasses import asdict, dataclass
from typing import Tuple

import polars as pl
from rich.console import Console
from rich.padding import Padding
from rich.table import Table
from rich.text import Text

from naics_gemini.utils.utilities import parquet_stats as _parquet_stats

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------------------------------

@dataclass
class Config:

    # Input
    distances_parquet: str = './data/naics_distances.parquet'
    descriptions_parquet: str = './data/naics_descriptions.parquet'
    
    # Output
    output_parquet: str = './data/naics_training_pairs.parquet'


# -------------------------------------------------------------------------------------------------
# Input
# -------------------------------------------------------------------------------------------------

def _input_parquet_files(
    descriptions_parquet: str, 
    distances_parquet: str
) -> Tuple[pl.DataFrame, pl.DataFrame]:

    descriptions = (
        pl
        .read_parquet(
            descriptions_parquet
        )
    )

    distances = (
        pl
        .read_parquet(
            distances_parquet
        )
        .select(
            code_i=pl.col('code_i'),
            code_j=pl.col('code_j'),
            distance=pl.col('distance')
        )
    )
    logger.info('Number of input observations:')
    logger.info(f'  descriptions: {descriptions.height: ,}')
    logger.info(f'  distances: {distances.height: ,}\n')

    return descriptions, distances


# -------------------------------------------------------------------------------------------------
# Exclusions
# -------------------------------------------------------------------------------------------------

def _get_exclusions(
    descriptions_df: pl.DataFrame,
    distances_df: pl.DataFrame
) -> pl.DataFrame:

    codes = set(
        descriptions_df
        .get_column('code')
        .unique()
        .sort()
        .to_list()
    )

    exclusions = (
        descriptions_df
        .filter(
            pl.col('excluded').is_not_null()
        )
        .select(
            positive_code=pl.col('code'),
            negative_code=pl.col('excluded').str.extract_all(r'\b\d{2,6}\b'),
        )
        .explode('negative_code')
        .sort('negative_code')
        .filter(
            pl.col('negative_code').is_not_null(), 
            pl.col('negative_code').is_in(codes)
        )
        .join(
            descriptions_df
            .select(
                negative_code=pl.col('code')
            ),
            on='negative_code',
            how='inner',
        )
        .join(
            distances_df
            .select(
                positive_code=pl.col('code_i'),
                negative_code=pl.col('code_j')
            ),
            on=['positive_code', 'negative_code'],
            how='inner',
        )
        .select(
            positive_code=pl.col('positive_code'),
            negative_code=pl.col('negative_code'),
            excluded=pl.lit(True)
        )
        .unique()
        .sort('positive_code', 'negative_code')
    )

    logger.info(f'Number of exclusions: {exclusions.height: ,}\n')

    return exclusions


# -------------------------------------------------------------------------------------------------
# Distances
# -------------------------------------------------------------------------------------------------

def _get_distances(distances_df: pl.DataFrame) -> Tuple[pl.DataFrame, pl.DataFrame]:

    positive_distances = (
        distances_df
        .filter(
            pl.col('distance')
              .ne(pl.col('distance').max())
        )
        .select(
            anchor_code=pl.col('code_i'),
            positive_code=pl.col('code_j'),
            positive_distance=pl.col('distance')
        )
        .unique()
        .sort('anchor_code', 'positive_code')
    )

    negative_distances = (
        distances_df
        .select(
            anchor_code=pl.col('code_i'),
            negative_code=pl.col('code_j'),
            negative_distance=pl.col('distance')
        )
        .unique()
        .sort('anchor_code', 'negative_code')
    )

    logger.info('Number of distances:')
    logger.info(f'  positives: {positive_distances.height: ,}')
    logger.info(f'  negatives: {negative_distances.height: ,}\n')

    return positive_distances, negative_distances


# -------------------------------------------------------------------------------------------------
# Pairs
# -------------------------------------------------------------------------------------------------

def _get_pairs(
    distances_df: pl.DataFrame,
    exclusions_df: pl.DataFrame
) -> Tuple[pl.DataFrame, pl.DataFrame]:

    positives = (
        distances_df
        .filter(
            pl.col('distance')
              .ne(pl.col('distance').max())
        )
        .select(
            anchor_code=pl.col('code_i'),
            positive_code=pl.col('code_j')
        )
        .unique()
        .sort('anchor_code', 'positive_code')
    )

    negatives = (
        distances_df
        .select(
            positive_code=pl.col('code_i'),
            negative_code=pl.col('code_j'),
            distance=pl.col('distance')
        )
        .join(
            exclusions_df,
            on=['positive_code', 'negative_code'],
            how='left'
        )
        .select(
            positive_code=pl.col('positive_code'),
            negative_code=pl.col('negative_code'),
            excluded=pl.col('excluded')
                       .fill_null(False),
            unrelated=pl.col('distance')
                        .eq(pl.col('distance').max())
        )

        .unique()
        .sort('positive_code', 'negative_code')
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
    negative_distances_df: pl.DataFrame
) -> pl.DataFrame:

    triplets = (
        positives_df
        .join(
            negatives_df,
            how='inner',
            on='positive_code'
        )
    )

    triplets = (
        triplets
        .join(
            positive_distances_df,
            how='inner',
            on=['anchor_code', 'positive_code']
        )
    )

    triplets = (
        triplets
        .join(
            negative_distances_df,
            how='inner',
            on=['anchor_code', 'negative_code']
        )
        .with_columns(
            distance_diff=pl.col('negative_distance') - pl.col('positive_distance')
        )
        .filter(
            pl.col('distance_diff').gt(0.0)
        )
    )

    logger.info(f'Number of triplets: {triplets.height: ,}\n')

    return triplets


# -------------------------------------------------------------------------------------------------
# Triplet stats
# -------------------------------------------------------------------------------------------------

def _triplet_stats(triplets_df: pl.DataFrame):       
    
    stats_df = (
        triplets_df
        .with_columns(
            unrelated=pl.when(pl.col('excluded')).then(pl.lit(True))
                        .otherwise(pl.col('unrelated')),
            distance_diff=pl.when(pl.col('excluded')).then(pl.lit(None))
                            .when(pl.col('unrelated')).then(pl.lit(7))
                            .otherwise(pl.col('distance_diff')),
            hardness=pl.when(pl.col('excluded')).then(pl.lit(8))
                       .when(pl.col('unrelated')).then(pl.lit(1))
                       .when(pl.col('distance_diff').le(0.5)).then(pl.lit(7))
                       .when(pl.col('distance_diff').le(1)).then(pl.lit(6))
                       .when(pl.col('distance_diff').le(2)).then(pl.lit(5))
                       .when(pl.col('distance_diff').le(3)).then(pl.lit(4))
                       .when(pl.col('distance_diff').le(4)).then(pl.lit(3))
                       .otherwise(pl.lit(2))
        )
        .group_by('hardness', 'excluded', 'unrelated')
        .agg(
            distance_diff=pl.col('distance_diff'),
            count=pl.col('distance_diff').len()
        )
        .with_columns(
            distance_diff=pl.col('distance_diff')
                            .list.drop_nulls()
                            .list.unique()
                            .list.sort(),
            pct=pl.col('count')
                .truediv(pl.col('count').sum())
        )
        .sort('hardness', descending=True)
    )

    dists = (
        stats_df
        .explode('distance_diff')
        .filter(
            pl.col('distance_diff').is_not_null()
        )
        .get_column('distance_diff')
        .unique()
        .sort()
        .to_list()
    )

    logger.info(
        'Observed differences in positive and negative distances: '
        f"{', '.join(map(str, dists))}\n"
    )

    console = Console()

    def _render_triplet_table(rows):

        title = Text('Triplet Statistics: by hardness, exclusion, and unrelatedness', style='bold')
        table = Table(
            title=title,
            title_justify='left',
            show_lines=True
        )

        table.add_column('Hardness', justify='center', style='bold cyan')
        table.add_column('Exclusion', justify='center')
        table.add_column('Unrelated', justify='center')
        table.add_column('Distance Difference', justify='center')
        table.add_column('Frequency', justify='right')
        table.add_column('Percent', justify='right')

        for row in rows:
            hardness = str(row.get('hardness', ''))
            excluded = 'True' if row.get('excluded', False) else 'False'
            unrelated = 'True' if row.get('unrelated', False) else 'False'
            
            distance_diff = row.get('distance_diff', [])
            match distance_diff:
                case []:
                    dd = 'N/A'
                case [d]:
                    dd = str(d)
                case _:
                    d_min = min(distance_diff)
                    d_max = max(distance_diff)
                    dd = f'{d_min}-{d_max}'

            n = row.get('count', 0)
            pct = row.get('pct', 0)

            n_cell = Text(f'{n: ,}')
            pct_cell = Text(f'{100 * pct: .2f}%', style='bold')

            table.add_row(hardness, excluded, unrelated, dd, n_cell, pct_cell)

        console.print(Padding(table, (0, 0, 0, 10)))   

    _render_triplet_table(stats_df.to_dicts())


# -------------------------------------------------------------------------------------------------
# Generate triplets
# -------------------------------------------------------------------------------------------------

def generate_training_triplets() -> pl.DataFrame:

    # Configuration
    cfg = Config()

    logger.info('Configuration:')
    logger.info(json.dumps(asdict(cfg), indent=2))
    logger.info('')

    # Load data
    descriptions, distances = _input_parquet_files(
        cfg.descriptions_parquet,
        cfg.distances_parquet
    )

    # Exclusions
    exclusions = _get_exclusions(
        descriptions,
        distances
    )

    # All positive and negative distances
    positive_distances, negative_distances = _get_distances(distances)

    # All positive and negative pairs
    positives, negatives = _get_pairs(distances, exclusions)

    # Combine positives and negatives into triplets
    triplets_df = _get_triplets(
        positives,
        negatives,
        positive_distances,
        negative_distances
    )

    (
        triplets_df
        .write_parquet(
            cfg.output_parquet
        )
    )

    _triplet_stats(triplets_df)  

    _parquet_stats(
        parquet_df=triplets_df,
        message='NAICS triplets written to:',
        output_parquet=cfg.output_parquet,
        logger=logger
    )

    return triplets_df


# -------------------------------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    generate_training_triplets()