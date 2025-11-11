import json
import logging
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Tuple

import polars as pl
from rich.console import Console
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
    descriptions_parquet: str = './data/naics_descriptions.parquet'
    distances_parquet: str = './data/naics_distances.parquet'
    relations_parquet: str = './data/naics_relations.parquet'

    # Output
    output_parquet: str = './data/naics_training_pairs'


# -------------------------------------------------------------------------------------------------
# Input
# -------------------------------------------------------------------------------------------------


def _input_parquet_files(
    descriptions_parquet: str, distances_parquet: str, relations_parquet: str
) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
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
# Exclusions
# -------------------------------------------------------------------------------------------------

def _get_exclusions(descriptions_df: pl.DataFrame, distances_df: pl.DataFrame) -> pl.DataFrame:
    codes = set(descriptions_df.get_column('code').unique().sort().to_list())

    exclusions = (
        descriptions_df.filter(pl.col('excluded').is_not_null())
        .select(
            positive_code=pl.col('code'),
            negative_code=pl.col('excluded_codes'),
        )
        .explode('negative_code')
        .filter(pl.col('negative_code').is_not_null(), pl.col('negative_code').is_in(codes))
        .join(
            descriptions_df.select(negative_code=pl.col('code')),
            on='negative_code',
            how='inner',
        )
        .join(
            distances_df.select(positive_code=pl.col('code_i'), negative_code=pl.col('code_j')),
            on=['positive_code', 'negative_code'],
            how='inner',
        )
        .select(
            positive_code=pl.col('positive_code'),
            negative_code=pl.col('negative_code'),
            excluded=pl.lit(True),
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
        .filter(pl.col('distance').ne(pl.col('distance').max()))
        .select(
            anchor_code=pl.col('code_i'),
            positive_code=pl.col('code_j'),
            positive_distance=pl.col('distance'),
        )
        .unique()
        .sort('anchor_code', 'positive_code')
    )

    negative_distances = (
        distances_df
        .select(
            anchor_code=pl.col('code_i'),
            negative_code=pl.col('code_j'),
            negative_distance=pl.col('distance'),
        )
        .unique()
        .sort('anchor_code', 'negative_code')
    )

    logger.info('Number of distances:')
    logger.info(f'  positives: {positive_distances.height: ,}')
    logger.info(f'  negatives: {negative_distances.height: ,}\n')

    return positive_distances, negative_distances


# -------------------------------------------------------------------------------------------------
# Relationships
# -------------------------------------------------------------------------------------------------

def _get_relations(
    relations_df: pl.DataFrame, 
    distances_df: pl.DataFrame
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    
    positive_relations_df = (
        relations_df
        .select(
            anchor_code=pl.col('code_i'),
            positive_code=pl.col('code_j'),
            positive_relation=pl.col('relation'),
        )
        .unique()
        .sort('anchor_code', 'positive_code')
    )

    negative_relations_df = (
        relations_df
        .select(
            anchor_code=pl.col('code_i'),
            negative_code=pl.col('code_j'),
            negative_relation=pl.col('relation'),
        )
        .unique()
        .sort('anchor_code', 'negative_code')
    )

    positive_distances_df, negative_distances_df = _get_distances(distances_df)

    positive_relations = (
        positive_relations_df
        .join(
            positive_distances_df,
            how='inner',
            on=['anchor_code', 'positive_code']
        )
    )

    negative_relations = (
        negative_relations_df
        .join(
            negative_distances_df,
            how='inner',
            on=['anchor_code', 'negative_code']
        )
    )

    logger.info('Number of relationships:')
    logger.info(f'  positives: {positive_relations.height: ,}')
    logger.info(f'  negatives: {negative_relations.height: ,}\n')

    return positive_relations, negative_relations


# -------------------------------------------------------------------------------------------------
# Pairs
# -------------------------------------------------------------------------------------------------


def _get_pairs(
        distances_df: pl.DataFrame, 
        exclusions_df: pl.DataFrame
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    
    positives = (
        distances_df.filter(pl.col('distance').ne(pl.col('distance').max()))
        .select(
            anchor_idx=pl.col('idx_i'),
            positive_idx=pl.col('idx_j'),
            anchor_code=pl.col('code_i'),
            positive_code=pl.col('code_j'),
        )
        .unique()
        .sort('anchor_code', 'positive_code')
    )

    negatives = (
        distances_df.select(
            positive_idx=pl.col('idx_i'),
            negative_idx=pl.col('idx_j'),
            positive_code=pl.col('code_i'),
            negative_code=pl.col('code_j'),
            distance=pl.col('distance'),
        )
        .join(exclusions_df, on=['positive_code', 'negative_code'], how='left')
        .select(
            negative_idx=pl.col('negative_idx'),
            positive_code=pl.col('positive_code'),
            negative_code=pl.col('negative_code'),
            excluded=pl.col('excluded').fill_null(False)
        )
        .with_columns(
            un_1=pl.col('positive_code')
                .str.slice(0, 2),
            un_2=pl.col('negative_code')
                .str.slice(0, 2)
        )
        .with_columns(
            un_1=pl.when(pl.col('un_1').is_in(['31', '32', '33'])).then(pl.lit('31'))
                .when(pl.col('un_1').is_in(['44', '45'])).then(pl.lit('44'))
                .when(pl.col('un_1').is_in(['48', '49'])).then(pl.lit('48'))
                .otherwise(pl.col('un_1')),
            un_2=pl.when(pl.col('un_2').is_in(['31', '32', '33'])).then(pl.lit('31'))
                .when(pl.col('un_2').is_in(['44', '45'])).then(pl.lit('44'))
                .when(pl.col('un_2').is_in(['48', '49'])).then(pl.lit('48'))
                .otherwise(pl.col('un_2'))
        )
        .with_columns(
            unrelated=pl.col('un_1').ne(pl.col('un_2'))
        )
        .drop('un_1', 'un_2')
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
    negative_distances_df: pl.DataFrame,
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
            positive_distance=pl.col('positive_distance').cast(pl.Int8),
            negative_distance=pl.col('negative_distance').cast(pl.Int8)
        )
        .with_columns(
            relation_margin=pl.col('negative_relation').sub(pl.col('positive_relation')),
            distance_margin=pl.col('negative_distance').sub(pl.col('positive_distance'))
        )
        .with_columns(
            unrelated=pl.when(pl.col('excluded'))
                        .then(pl.lit(False))
                        .otherwise(pl.col('unrelated')),
            relation_margin=pl.when(pl.col('excluded')).then(pl.lit(0.5))
                            .when(pl.col('unrelated')).then(pl.lit(15))
                            .otherwise(pl.col('relation_margin')),
            distance_margin=pl.when(pl.col('excluded')).then(pl.lit(0.25))
                            .when(
                                pl.col('relation_margin').eq(1) & 
                                pl.col('distance_margin').eq(-1)
                            ).then(0.5)
                            .when(
                                pl.col('relation_margin').eq(1) & 
                                pl.col('distance_margin').eq(0)
                            ).then(0.75)
                            .when(pl.col('unrelated')).then(pl.lit(15))
                            .otherwise(pl.col('distance_margin'))
        )
        .filter(
            pl.col('relation_margin').gt(0),
            pl.col('distance_margin').gt(0)
        )
        .with_columns(
            relation_margin=pl.col('relation_margin')
                            .rank('dense')
                            .replace({15: 100})
                            .sub(1),
            distance_margin=pl.col('distance_margin')
                            .rank('dense')
                            .replace({13: 100})
                            .sub(1)
        )
        .with_columns(
            margin=(
                pl.col('relation_margin').mul(0.5) + 
                pl.col('distance_margin').mul(0.5)
            )
            .rank('dense')
            .sub(1)    
        )
        .with_columns(
            margin=pl.when(pl.col('margin').eq(pl.col('margin').max())).then(99)
                    .otherwise(pl.col('margin'))
        )
    )

    triplets_anti = (
        triplets
        .filter(
            pl.col('unrelated')
        )
        .group_by('anchor_idx', 'positive_idx')
        .agg(
            negatives=pl.col('negative_idx')
        )
        .with_columns(
            sample=pl.col('negatives')
                     .list.sample(100, shuffle=True)
        )
        .select(
            anchor_idx=pl.col('anchor_idx'), 
            positive_idx=pl.col('positive_idx'),
            negative_idx=pl.col('negatives')
                           .list.set_difference(pl.col('sample'))
        )
        .explode('negative_idx')
    )

    triplets = (
        triplets
        .join(
            triplets_anti,
            how='anti',
            on=['anchor_idx', 'positive_idx', 'negative_idx']
        )
    )

    triplets = (
        triplets
        .with_columns(
            anchor_level=pl.col('anchor_code')
                           .str.len_chars(),
            positive_level=pl.col('positive_code')
                             .str.len_chars(),
            negative_level=pl.col('negative_code')
                             .str.len_chars(),
        )
        .sort('anchor_idx', 'positive_idx', 'negative_idx')
        .select(
            'anchor_idx', 'positive_idx', 'negative_idx',
            'anchor_code', 'positive_code', 'negative_code',
            'anchor_level', 'positive_level', 'negative_level',
            'excluded', 'unrelated',
            'relation_margin', 'distance_margin',
            'positive_relation', 'negative_relation',
            'positive_distance', 'negative_distance'
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
        triplets_df
        .group_by('excluded', 'unrelated', 'relation_margin', 'distance_margin')
        .agg(
            n=pl.len(),
        )
        .with_columns(
            pct=pl.col('n')
                  .truediv(pl.col('n').sum())
                  .mul(100)
        )
        .sort(
            'excluded', 'unrelated', 'relation_margin', 'distance_margin', 
            descending=[True, False, False, False]
        )
    )

    rels = stats_df.get_column('relation_margin').unique().sort().to_list()
    dists = stats_df.get_column('distance_margin').unique().sort().to_list()

    logger.info(
        'Observed relation margins (differences in positive and negative relations): '
        f'{", ".join(str(r) for r in rels)}\n'
    )

    logger.info(
        'Observed distance margins (differences in positive and negative distances): '
        f'{", ".join(str(d) for d in dists)}\n'
    )

    console = Console()

    def _render_triplet_table(rows):
        
        title = Text('Triplet Statistics:', style='bold')

        table = Table(title=title, title_justify='left', show_lines=True, show_footer=True)

        total_count = sum(row.get('n', 0) for row in rows)
        total_pct = sum(row.get('pct', 0) for row in rows)

        table.add_column('Exclusion', justify='center')
        table.add_column('Unrelated', justify='center')
        table.add_column('Relation Margin', justify='center')
        table.add_column('Distance Margin', justify='center')
        table.add_column('Frequency', justify='right', footer=f'[bold]{total_count: ,}[/bold]')
        table.add_column('Percent', justify='right', footer=f'[bold]{total_pct: .4f}%[/bold]')

        for row in rows:
            excluded = 'True' if row.get('excluded', False) else 'False'
            unrelated = 'True' if row.get('unrelated', False) else 'False'

            relation_margin = row.get('relation_margin')
            distance_margin = row.get('distance_margin')
            
            rm_cell = Text(f'{relation_margin}', style='bold')
            dm_cell = Text(f'{distance_margin: .3f}', style='bold')

            n = row.get('n', 0)
            pct = row.get('pct', 0)

            n_cell = Text(f'{n: ,}')
            pct_cell = Text(f'{pct: .4f}%', style='bold')

            table.add_row(excluded, unrelated, rm_cell, dm_cell, n_cell, pct_cell)

        console.print(table)

    return _render_triplet_table(stats_df.to_dicts())


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
    descriptions, distances, relations = _input_parquet_files(
        cfg.descriptions_parquet, 
        cfg.distances_parquet, 
        cfg.relations_parquet
    )

    # Exclusions
    exclusions = _get_exclusions(descriptions, distances)

    # All positive and negative distances
    positive_distances, negative_distances = _get_relations(relations, distances)

    # All positive and negative pairs
    positives, negatives = _get_pairs(distances, exclusions)

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
        triplets_df
        .with_columns(
            anchor=pl.col('anchor_idx')
        )
        .write_parquet(
            cfg.output_parquet,
            use_pyarrow=True,
            pyarrow_options={
                'partition_cols': ['anchor']
            }
        )
    )

    return triplets_df


# -------------------------------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    generate_training_triplets()