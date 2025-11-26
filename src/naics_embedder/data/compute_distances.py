# -------------------------------------------------------------------------------------------------
# Imports and settings
# -------------------------------------------------------------------------------------------------

import logging
from itertools import combinations
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import polars as pl

from naics_embedder.utils.config import DistancesConfig, load_config
from naics_embedder.utils.console import log_table as _log_table
from naics_embedder.utils.utilities import parquet_stats as _parquet_stats

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------------------------------------
# Distance utilities
# -------------------------------------------------------------------------------------------------


def _sectors(input_parquet: str) -> List[str]:
    return (
        pl.read_parquet(input_parquet)
        .filter(pl.col('level').eq(2))
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
        pl.read_parquet(input_parquet)
        .filter(pl.col('level').eq(6), pl.col('code').str.slice(0, 2).is_in(sector_list))
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
    if sector == '31':
        sector_list = ['31', '32', '33']

    elif sector == '44':
        sector_list = ['44', '45']

    elif sector == '48':
        sector_list = ['48', '49']

    else:
        sector_list = [sector]

    code_6 = _sector_codes(sector, input_parquet)
    code_5 = sorted(set(n[:5] for n in code_6))
    code_4 = sorted(set(n[:4] for n in code_6))
    code_3 = sorted(set(n[:3] for n in code_6))
    code_2 = sorted(set(n[:2] for n in code_6))

    if not code_6:
        sector_df = pl.read_parquet(input_parquet).filter(
            pl.col('code').str.slice(0, 2).is_in(sector_list)
        )

        codes = sector_df.select('code').unique(maintain_order=True).get_column('code').to_list()

        graph = nx.DiGraph()
        graph.add_nodes_from(codes)

        for code in codes:
            if len(code) <= 2:
                continue

            parent = code[:-1]
            parent = _join_sectors(parent) if len(parent) == 2 else parent

            if parent in codes:
                graph.add_edge(parent, code)

        root = _join_sectors(sector)
        if not graph.has_node(root):
            graph.add_node(root)

        return graph

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


def _get_distance(i: str, j: str, depths: Dict[str, int], ancestors: Dict[str, List[str]]) -> float:
    if i == j:
        return 0.0

    depth_i, depth_j = depths[i], depths[j]
    common_ancestor = _find_common_ancestor(i, j, ancestors)
    if common_ancestor is None:
        return float(depth_i + depth_j)

    depth_ancestor = depths[common_ancestor]

    distance = (depth_i - depth_ancestor) + (depth_j - depth_ancestor)

    is_lineal = (i in ancestors[j]) or (j in ancestors[i])
    if is_lineal:
        distance -= 0.5

    return float(max(distance, 0.0))


# -------------------------------------------------------------------------------------------------
# Exclusions
# -------------------------------------------------------------------------------------------------


def _get_exclusions(distances_df: pl.DataFrame) -> pl.DataFrame:
    descriptions_df = pl.read_parquet('./data/naics_descriptions.parquet')

    codes = set(descriptions_df.get_column('code').unique().sort().to_list())

    exclusions = (
        descriptions_df.filter(pl.col('excluded').is_not_null())
        .select(
            code_i=pl.col('code'),
            code_j=pl.col('excluded_codes'),
        )
        .explode('code_j')
        .filter(pl.col('code_j').is_not_null(), pl.col('code_j').is_in(codes))
        .join(
            descriptions_df.select(code_j=pl.col('code')),
            on='code_j',
            how='inner',
        )
        .join(
            distances_df.select(pl.col('code_i'), pl.col('code_j')),
            on=['code_i', 'code_j'],
            how='inner',
        )
        .select(
            code_i=pl.col('code_i'),
            code_j=pl.col('code_j'),
            excluded=pl.lit(True),
        )
        .unique()
        .sort('code_i', 'code_j')
    )

    print(f'Number of exclusions: {exclusions.height: ,}\n')

    return exclusions


# -------------------------------------------------------------------------------------------------
# Distance matrix
# -------------------------------------------------------------------------------------------------


def _get_distance_matrix(df: pl.DataFrame) -> pl.DataFrame:
    '''Create distance matrix from distances DataFrame.'''

    codes = sorted(set(df['code_i'].to_list() + df['code_j'].to_list()))
    n_codes = len(codes)

    code_to_idx = {code: idx for idx, code in enumerate(codes)}

    dist_matrix = np.zeros((n_codes, n_codes), dtype=float)
    for row in df.iter_rows(named=True):
        i = code_to_idx[row['code_i']]
        j = code_to_idx[row['code_j']]
        dist = row['distance']
        dist_matrix[i, j] = dist
        dist_matrix[j, i] = dist

    dist_matric_schema = []
    for code, idx in code_to_idx.items():
        dist_matric_schema.append((f'idx_{idx}-code_{code}', pl.Float64))

    return pl.from_numpy(data=dist_matrix, schema=dist_matric_schema)


# -------------------------------------------------------------------------------------------------
# Distance stats
# -------------------------------------------------------------------------------------------------


def _distance_stats(distances_df: pl.DataFrame):
    stats_df = (
        distances_df.group_by('distance')
        .agg(cnt=pl.len())
        .with_columns(pct=pl.col('cnt').truediv(pl.col('cnt').sum()).mul(100))
        .sort('distance')
    )

    _log_table(
        df=stats_df,
        title='Distance Statistics',
        headers=['Distance', 'Frequency', 'Percent'],
        logger=logger,
        output='./outputs/distance_stats.pdf',
    )


# -------------------------------------------------------------------------------------------------
# Main Entry Point
# -------------------------------------------------------------------------------------------------


def calculate_pairwise_distances() -> pl.DataFrame:
    # Load configuration from YAML
    cfg = load_config(DistancesConfig, './data/distances.yaml')

    logger.info('Configuration:')
    logger.info(cfg.model_dump_json(indent=2))
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

    pair_relations = pl.concat(df_list).select(
        pl.col('code_i'), pl.col('code_j'), distance=pl.col('distance')
    )

    naics_i = pl.scan_parquet(cfg.input_parquet).select(
        idx_i=pl.col('index'), lvl_i=pl.col('level'), code_i=pl.col('code')
    )

    naics_j = pl.scan_parquet(cfg.input_parquet).select(
        idx_j=pl.col('index'), lvl_j=pl.col('level'), code_j=pl.col('code')
    )

    distances_df = (
        naics_i.join(naics_j, how='cross')
        .with_columns(
            sector_i=pl.col('code_i').str.slice(0, 2), sector_j=pl.col('code_j').str.slice(0, 2)
        )
        .with_columns(
            keep=(
                (pl.col('lvl_i') <= pl.col('lvl_j'))
                & (pl.col('sector_i') == pl.col('sector_j'))
                & (pl.col('code_i').cast(pl.UInt32) < pl.col('code_j').cast(pl.UInt32))
            )
            | ((pl.col('lvl_i') <= pl.col('lvl_j')) & (pl.col('sector_i') != pl.col('sector_j')))
        )
        .filter(pl.col('keep'))
        .collect()
        .join(pair_relations, how='left', on=['code_i', 'code_j'])
        .select(
            pl.col('idx_i'),
            pl.col('idx_j'),
            pl.col('code_i'),
            pl.col('code_j'),
            distance=pl.col('distance').fill_null(99.0),
        )
        .sort('idx_i', 'idx_j')
    )

    exclusions = _get_exclusions(distances_df)

    distances_df = (
        distances_df.join(exclusions, on=['code_i', 'code_j'], how='left')
        .with_columns(excluded=pl.col('excluded').fill_null(False))
        .select(
            pl.col('idx_i'),
            pl.col('idx_j'),
            pl.col('code_i'),
            pl.col('code_j'),
            distance=pl.when(pl.col('excluded')).then(pl.lit(0)).otherwise(pl.col('distance')),
        )
        .sort('idx_i', 'idx_j')
    )

    (distances_df.write_parquet(cfg.distances_parquet))

    _distance_stats(distances_df)

    _parquet_stats(
        parquet_df=distances_df,
        message='NAICS pairwise distances written to',
        output_parquet=cfg.distances_parquet,
        logger=logger,
    )

    dist_matrix = _get_distance_matrix(distances_df)

    (dist_matrix.write_parquet(cfg.distance_matrix_parquet))

    _parquet_stats(
        parquet_df=dist_matrix,
        message='NAICS distance matrix written to',
        output_parquet=cfg.distance_matrix_parquet,
        logger=logger,
    )

    return distances_df


# -------------------------------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------------------------------


if __name__ == '__main__':
    calculate_pairwise_distances()
