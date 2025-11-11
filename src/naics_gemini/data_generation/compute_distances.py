# -------------------------------------------------------------------------------------------------
# Imports and settings
# -------------------------------------------------------------------------------------------------

import json
import logging
from dataclasses import asdict, dataclass
from itertools import combinations
from typing import Dict, List, Optional, Tuple

import networkx as nx
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

    input_parquet: str = './data/naics_descriptions.parquet'
    output_parquet: str = './data/naics_distances.parquet'


# -------------------------------------------------------------------------------------------------
# Distance utilities
# -------------------------------------------------------------------------------------------------

def _sectors(input_parquet: str) -> List[str]:

    return (
        pl
        .read_parquet(
            input_parquet
        )
        .filter(
            pl.col('level').eq(2)
        )
        .select('code')
        .sort(pl.col('code').cast(pl.UInt32))
        .unique(maintain_order=True)
        .get_column('code')
        .to_list()
    )


def _sector_codes(sector: str, input_parquet: str) -> List[str]:

    if sector == '31':
        sector_list = ['31', '32', '33']

    elif sector == '44':
        sector_list = ['44', '45']

    elif sector == '48':
        sector_list = ['48', '49']

    else:
        sector_list = [sector]

    return (
        pl
        .read_parquet(
            input_parquet
        )
        .filter(
            pl.col('level').eq(6), 
            pl.col('code').str.slice(0, 2).is_in(sector_list)
        )
        .select('code')
        .sort(pl.col('code').cast(pl.UInt32))
        .unique(maintain_order=True)
        .get_column('code')
        .to_list()
    )


def _join_sectors(code: str) -> str:

    if code in ['31', '32', '33']:
        return '31'
    
    elif code in ['44', '45']:
        return '44'
    
    elif code in ['48', '49']:
        return '48'
    
    else:
        return code


def _sector_tree(sector: str, input_parquet: str) -> nx.DiGraph:

    code_6 = _sector_codes(sector, input_parquet)
    code_5 = sorted(set(n[:5] for n in code_6))
    code_4 = sorted(set(n[:4] for n in code_6))
    code_3 = sorted(set(n[:3] for n in code_6))
    code_2 = sorted(set(n[:2] for n in code_6))
    edge_list = []
    for c2 in code_2:
        s = _join_sectors(c2)
        for c3 in [c3 for c3 in code_3 if c3[:2] == c2]:
            edge_list.append((s, c3))
            for c4 in [c4 for c4 in code_4 if c4[:3] == c3]:
                edge_list.append((c3, c4))
                for c5 in [c5 for c5 in code_5 if c5[:4] == c4]:
                    edge_list.append((c4, c5))
                    for c6 in [c6 for c6 in code_6 if c6[:5] == c5]:
                        edge_list.append((c5, c6))

    edges = set(edge_list)

    return nx.DiGraph(edges)


def _compute_tree_metadata(tree: nx.DiGraph, root: str) -> Tuple[Dict, Dict, Dict]:

    depths, ancestors, parents = {}, {}, {}
    queue = [(root, 0, [root])]
    while queue:
        node, depth, ancestor_path = queue.pop(0)
        depths[node] = depth
        ancestors[node] = ancestor_path.copy()
        parents[node] = ancestor_path[-2] if len(ancestor_path) > 1 else None

        for child in tree.successors(node):
            queue.append((child, depth + 1, ancestor_path + [child]))

    return depths, ancestors, parents


def _find_common_ancestor(i: str, j: str, ancestors: Dict[str, List[str]]) -> Optional[str]:

    ancestors_i, ancestors_j = set(ancestors[i]), ancestors[j]
    for ancestor in reversed(ancestors_j):
        
        if ancestor in ancestors_i:
            return ancestor

    return None


# -------------------------------------------------------------------------------------------------
# 2. Compute relationships
# -------------------------------------------------------------------------------------------------


def _get_distance(
    i: str, 
    j: str, 
    depths: Dict[str, int], 
    ancestors: Dict[str, List[str]]
) -> float:

    depth_i, depth_j = depths[i], depths[j]

    common_ancestor = _find_common_ancestor(i, j, ancestors)
    if common_ancestor is None:
        return depths[i] + depths[j]

    depth_ancestor = depths[common_ancestor]

    distance = (
        (depth_i - depth_ancestor) +
        (depth_j - depth_ancestor)
    )

    lineal = 1 if i in ancestors[j] else 0

    return distance - 0.5 * lineal


# -------------------------------------------------------------------------------------------------
# Distance stats
# -------------------------------------------------------------------------------------------------

def _distance_stats(distances_df: pl.DataFrame):       
    
    stats_df = (
        distances_df
        .group_by('distance')
        .agg(
            count=pl.len()
        )
        .with_columns(
            pct=pl.col('count')
                  .truediv(pl.col('count').sum())
        )
        .sort('distance')
    )

    console = Console()

    def _render_triplet_table(rows):

        title = Text('\nDistance Statistics:', style='bold')

        table = Table(title=title, title_justify='left', show_lines=True, show_footer=True)

        total_count = sum(row.get('count', 0) for row in rows)
        total_pct = 100 * sum(row.get('pct', 0) for row in rows)

        table.add_column('Distance', justify='center', style='bold cyan')
        table.add_column('Frequency', justify='right', footer=f'[bold]{total_count: ,}[/bold]')
        table.add_column('Percent', justify='right', footer=f'[bold]{total_pct: .2f}%[/bold]')

        for row in rows:

            distance = str(row.get('distance', ''))

            n = row.get('count', 0)
            pct = row.get('pct', 0)

            n_cell = Text(f'{n: ,}')
            pct_cell = Text(f'{100 * pct: .4f}%', style='bold')

            table.add_row(distance, n_cell, pct_cell)

        console.print(table)

    _render_triplet_table(stats_df.to_dicts())


# -------------------------------------------------------------------------------------------------
# Main Entry Point
# -------------------------------------------------------------------------------------------------

def calculate_pairwise_distances() -> pl.DataFrame:
    
    cfg = Config()

    logger.info('Configuration:')
    logger.info(json.dumps(asdict(cfg), indent=2))
    logger.info('')

    df_list = []
    for sector in _sectors(cfg.input_parquet):

        G = _sector_tree(sector, cfg.input_parquet)

        pairs = [(i, j) if int(i) < int(j) else (j, i) for i, j in combinations(G.nodes, 2)]

        depths, ancestors, parents = _compute_tree_metadata(G, sector)

        distances = []
        for i, j in sorted(pairs, key=lambda x: (x[0], x[1])):
            distance = _get_distance(i, j, depths, ancestors)
            distances.append({'code_i': i, 'code_j': j, 'distance': distance})

        df = pl.DataFrame(
            data=distances,
            schema={'code_i': pl.Utf8, 'code_j': pl.Utf8, 'distance': pl.Float32},
        )

        logger.info(f'Sector {sector}: [{len(depths): ,} nodes, {df.height: ,} pairs]')

        df_list.append(df)

    pair_relations = (
        pl
        .concat(
            df_list
        )
        .select(
            pl.col('code_i'),
            pl.col('code_j'),
            distance=pl.col('distance')
    ))

    naics_i = (
        pl.scan_parquet(cfg.input_parquet)
        .select(
            idx_i=pl.col('index'), 
            lvl_i=pl.col('level'), 
            code_i=pl.col('code')
        )
    )

    naics_j = (
        pl.scan_parquet(cfg.input_parquet)
        .select(
            idx_j=pl.col('index'), 
            lvl_j=pl.col('level'), 
            code_j=pl.col('code')
        )
    )

    naics_distances = (
        naics_i.join(
            naics_j, 
            how='cross'
        )
        .with_columns(
            sector_i=pl.col('code_i').str.slice(0, 2), 
            sector_j=pl.col('code_j').str.slice(0, 2)
        )
        .with_columns(
            keep=(
                (pl.col('lvl_i') <= pl.col('lvl_j'))
                & (pl.col('sector_i') == pl.col('sector_j'))
                & (pl.col('code_i').cast(pl.UInt32) < pl.col('code_j').cast(pl.UInt32))
            )
            | (
                (pl.col('lvl_i') <= pl.col('lvl_j'))
                & (pl.col('sector_i') != pl.col('sector_j'))
            )
        )
        .filter(
            pl.col('keep')
        )
        .collect()
        .join(
            pair_relations, 
            how='left', 
            on=['code_i', 'code_j']
        )
        .select(
            pl.col('idx_i'),
            pl.col('idx_j'),
            pl.col('code_i'),
            pl.col('code_j'),
            distance=pl.col('distance')
                       .fill_null(9.0)
        )
        .with_columns(
            distance=pl.col('distance')
                       .rank('dense')
        )
        .sort('idx_i', 'idx_j')
    )

    (
        naics_distances
        .write_parquet(cfg.output_parquet)
    )

    _distance_stats(naics_distances)   

    _parquet_stats(
        parquet_df=naics_distances,
        message='NAICS pairwise distances written to',
        output_parquet=cfg.output_parquet,
        logger=logger
    )

    return naics_distances


# -------------------------------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------------------------------


if __name__ == '__main__':
    
    calculate_pairwise_distances()
