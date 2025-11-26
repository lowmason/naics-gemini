# -------------------------------------------------------------------------------------------------
# Imports and settings
# -------------------------------------------------------------------------------------------------

import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

import numpy as np
import polars as pl

from naics_embedder.utils.config import StreamingConfig
from naics_embedder.utils.utilities import get_indices_codes

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------------------------------------
# Utility functions
# -------------------------------------------------------------------------------------------------


def _taxonomy(codes_parquet: str) -> pl.DataFrame:
    return (
        pl.read_parquet(codes_parquet)
        .filter(pl.col('level').eq(6))
        .select('code')
        .sort(pl.col('code').cast(pl.UInt32))
        .unique(maintain_order=True)
        .select(
            code_2=pl.col('code').str.slice(0, 2),
            code_3=pl.col('code').str.slice(0, 3),
            code_4=pl.col('code').str.slice(0, 4),
            code_5=pl.col('code').str.slice(0, 5),
            code_6=pl.col('code').str.slice(0, 6),
        )
        .with_columns(
            code_2=pl.when(pl.col('code_2').is_in(['31', '32', '33']))
            .then(pl.lit('31'))
            .when(pl.col('code_2').is_in(['44', '45']))
            .then(pl.lit('44'))
            .when(pl.col('code_2').is_in(['48', '49']))
            .then(pl.lit('48'))
            .otherwise(pl.col('code_2'))
        )
        .with_columns(
            code=pl.concat_str(pl.col('code_2'), pl.col('code_6').str.slice(2, 4), separator='')
        )
    )


def _anchors(triplets_parquet: str) -> pl.DataFrame:
    return (
        pl.read_parquet(triplets_parquet)
        .select(level=pl.col('anchor_level'), anchor=pl.col('anchor_code'))
        .unique()
        .sort(pl.col('level'), pl.col('anchor').cast(pl.UInt32))
    )


def _linear_skip(anchor: str, taxonomy: pl.DataFrame) -> List[str]:
    lvl = len(anchor)
    anchor_code = f'code_{lvl}'
    codes = [f'code_{i}' for i in range(lvl + 1, 7)]

    for code in codes:
        candidate = (
            taxonomy.filter(pl.col(anchor_code).eq(anchor)).get_column(code).unique().to_list()
        )

        if lvl == 5:
            return candidate
        elif len(candidate) > 1:
            return sorted(set(candidate))

    return taxonomy.filter(pl.col(anchor_code).eq(anchor)).get_column('code_6').unique().to_list()


# -------------------------------------------------------------------------------------------------
# Descendants
# -------------------------------------------------------------------------------------------------


def _descendants(anchors: pl.DataFrame, taxonomy: pl.DataFrame) -> pl.DataFrame:
    parent_anchors = (
        anchors.filter(pl.col('level').lt(6)).get_column('anchor').unique().sort().to_list()
    )

    parent_stratum = []
    for anchor in parent_anchors:
        parent_stratum.append({'anchor': anchor, 'stratum': _linear_skip(anchor, taxonomy)})

    return (
        pl.DataFrame(data=parent_stratum, schema={'anchor': pl.Utf8, 'stratum': pl.List(pl.Utf8)})
        .filter(pl.col('stratum').is_not_null())
        .select(
            level=pl.col('anchor').str.len_chars(),
            anchor=pl.col('anchor'),
            positive=pl.col('stratum'),
        )
    )


# -------------------------------------------------------------------------------------------------
# Ancestors
# -------------------------------------------------------------------------------------------------


def _ancestors(
    anchors: pl.DataFrame,
    taxonomy: pl.DataFrame,
) -> pl.DataFrame:
    return (
        anchors.filter(pl.col('level').eq(6))
        .join(taxonomy, left_on='anchor', right_on='code_6', how='inner')
        .select(
            level=pl.col('level'),
            anchor=pl.col('anchor'),
            code_5=pl.col('code_5'),
            code_4=pl.col('code_4'),
            code_3=pl.col('code_3'),
            code_2=pl.col('code_2'),
        )
        .unpivot(
            ['code_5', 'code_4', 'code_3', 'code_2'],
            index=['level', 'anchor'],
            variable_name='ancestor_level',
            value_name='ancestor',
        )
        .with_columns(
            ancestor_level=pl.col('ancestor_level').str.slice(5, 1).cast(pl.Int8).add(-6).mul(-1)
        )
        .sort('level', 'anchor', 'ancestor_level')
        .group_by('level', 'anchor', maintain_order=True)
        .agg(positive=pl.col('ancestor'))
    )


def sample_positives(
    descriptions_path: str = './data/naics_descriptions.parquet',
    triplets_path: str = './data/naics_training_pairs/*/*.parquet',
) -> pl.DataFrame:
    taxonomy = _taxonomy(descriptions_path)
    anchors = _anchors(triplets_path)
    code_to_idx = get_indices_codes('code_to_idx')
    descendants = _descendants(anchors, taxonomy)
    ancestors = _ancestors(anchors, taxonomy)

    return (
        pl.concat([descendants, ancestors])
        .explode('positive')
        .select(
            anchor_idx=pl.col('anchor').replace(code_to_idx).cast(pl.UInt32),
            positive_idx=pl.col('positive').replace(code_to_idx).cast(pl.UInt32),
            anchor_code=pl.col('anchor'),
            positive_code=pl.col('positive'),
            anchor_level=pl.col('level'),
            positive_level=pl.col('positive').str.len_chars(),
        )
        .sort('anchor_idx', 'positive_idx')
    )


# -------------------------------------------------------------------------------------------------
# Phase 1 Sampling Utilities
# -------------------------------------------------------------------------------------------------


def _load_relation_matrix(
    relation_matrix_path: str, code_to_idx: Dict[str, int], idx_to_code: Dict[int, str]
) -> Dict[Tuple[str, str], float]:
    '''
    Load relation matrix and create a lookup dictionary.

    The distarelationnce matrix has columns like 'idx_0-code_11', 'idx_1-code_111', etc.
    where the column name format is 'idx_{matrix_idx}-code_{code}'.
    Each row corresponds to an anchor code in sorted order.

    Note: The relation matrix uses sorted code order for indices, which may differ
    from the 'index' column in descriptions.parquet. We use code strings as keys.

    Args:
        relation_matrix_path: Path to relation matrix parquet file
        code_to_idx: Mapping from code to index (from descriptions parquet)
        idx_to_code: Mapping from index to code (from descriptions parquet)

    Returns:
        Dictionary mapping (anchor_code, negative_code) -> relation_distance
    '''

    logger.info('Loading relation matrix for Phase 1 sampling...')

    df = pl.read_parquet(relation_matrix_path)

    # Create lookup dictionary using code strings as keys
    relation_lookup = {}

    # Build mapping from column name to negative code
    col_to_negative_code = {}
    for col in df.columns:
        # Column format: 'idx_{matrix_idx}-code_{code}'
        parts = col.split('-')
        if len(parts) != 2:
            continue

        # Extract code from second part (e.g., 'code_111' -> '111')
        code_str = parts[1].replace('code_', '')

        if code_str in code_to_idx:
            col_to_negative_code[col] = code_str

    # Get all codes sorted by code string (matching relation matrix row order)
    # The relation matrix rows are in sorted code order
    codes_sorted = sorted(code_to_idx.keys())

    # Build lookup: (anchor_code, negative_code) -> relation
    for row_idx, anchor_code in enumerate(codes_sorted):
        if row_idx >= df.height:
            break

        # For each column (negative)
        for col, negative_code in col_to_negative_code.items():
            if col not in df.columns:
                continue

            # Get relation value from the row and column
            relation_val = df.select(pl.col(col)).row(row_idx)[0]

            if relation_val is not None:
                relation_lookup[(anchor_code, negative_code)] = float(relation_val)

    logger.info(f'Loaded {len(relation_lookup):,} distance entries')
    return relation_lookup


def _load_distance_matrix(
    distance_matrix_path: str, code_to_idx: Dict[str, int], idx_to_code: Dict[int, str]
) -> Dict[Tuple[str, str], float]:
    '''
    Load distance matrix and create a lookup dictionary.

    The distance matrix has columns like 'idx_0-code_11', 'idx_1-code_111', etc.
    where the column name format is 'idx_{matrix_idx}-code_{code}'.
    Each row corresponds to an anchor code in sorted order.

    Note: The distance matrix uses sorted code order for indices, which may differ
    from the 'index' column in descriptions.parquet. We use code strings as keys.

    Args:
        distance_matrix_path: Path to distance matrix parquet file
        code_to_idx: Mapping from code to index (from descriptions parquet)
        idx_to_code: Mapping from index to code (from descriptions parquet)

    Returns:
        Dictionary mapping (anchor_code, negative_code) -> tree_distance
    '''

    logger.info('Loading distance matrix for Phase 1 sampling...')

    df = pl.read_parquet(distance_matrix_path)

    # Create lookup dictionary using code strings as keys
    distance_lookup = {}

    # Build mapping from column name to negative code
    col_to_negative_code = {}
    for col in df.columns:
        # Column format: 'idx_{matrix_idx}-code_{code}'
        parts = col.split('-')
        if len(parts) != 2:
            continue

        # Extract code from second part (e.g., 'code_111' -> '111')
        code_str = parts[1].replace('code_', '')

        if code_str in code_to_idx:
            col_to_negative_code[col] = code_str

    # Get all codes sorted by code string (matching distance matrix row order)
    # The distance matrix rows are in sorted code order
    codes_sorted = sorted(code_to_idx.keys())

    # Build lookup: (anchor_code, negative_code) -> distance
    for row_idx, anchor_code in enumerate(codes_sorted):
        if row_idx >= df.height:
            break

        # For each column (negative)
        for col, negative_code in col_to_negative_code.items():
            if col not in df.columns:
                continue

            # Get distance value from the row and column
            distance_val = df.select(pl.col(col)).row(row_idx)[0]

            if distance_val is not None:
                distance_lookup[(anchor_code, negative_code)] = float(distance_val)

    logger.info(f'Loaded {len(distance_lookup):,} distance entries')
    return distance_lookup


def _load_excluded_codes(
    descriptions_path: str, code_to_idx: Optional[Dict[str, int]] = None
) -> Dict[str, Set[str]]:
    '''
    Load excluded codes from descriptions parquet.

    Args:
        descriptions_path: Path to descriptions parquet file

    Returns:
        Dictionary mapping code -> set of excluded codes
    '''
    logger.info('Loading excluded codes for Phase 1 sampling...')

    df = (
        pl.read_parquet(descriptions_path)
        .select('code', 'excluded_codes')
        .filter(pl.col('excluded_codes').is_not_null())
    )

    excluded_map = {}
    unknown_codes = 0
    for row in df.iter_rows(named=True):
        code = row['code']
        excluded_list = row['excluded_codes']
        if excluded_list:
            filtered = set()
            for ex_code in excluded_list:
                if code_to_idx is not None and ex_code not in code_to_idx:
                    unknown_codes += 1
                    continue
                filtered.add(ex_code)

            if filtered:
                excluded_map[code] = filtered

    logger.info(f'Loaded excluded codes for {len(excluded_map):,} codes')
    if unknown_codes:
        logger.warning(f'{unknown_codes:,} excluded codes not in taxonomy were ignored')
    return excluded_map


def _compute_phase1_weights(
    anchor_code: str,
    anchor_idx: int,
    candidate_negatives: List[Dict[str, Any]],
    distance_lookup: Dict[Tuple[str, str], float],
    excluded_map: Dict[str, Set[str]],
    code_to_idx: Dict[str, int],
    alpha: float = 1.5,
    exclusion_weight: float = 100.0,
) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Compute Phase 1 sampling weights for candidate negatives.

    Args:
        anchor_code: Anchor code
        anchor_idx: Anchor index
        candidate_negatives: List of candidate negative dictionaries with
            'negative_code' and 'negative_idx'
        distance_lookup: Dictionary mapping (anchor_idx, negative_idx) -> tree_distance
        excluded_map: Dictionary mapping code -> set of excluded codes
        code_to_idx: Mapping from code to index
        alpha: Exponent for inverse tree distance weighting
        exclusion_weight: High constant weight for excluded codes

    Returns:
        Array of sampling weights (unnormalized)
    '''
    weights = np.zeros(len(candidate_negatives))
    excluded_mask = np.zeros(len(candidate_negatives), dtype=bool)

    for i, neg in enumerate(candidate_negatives):
        negative_code = neg['negative_code']
        neg['negative_idx']

        # Check if anchor excludes this negative
        if anchor_code in excluded_map and negative_code in excluded_map[anchor_code]:
            weights[i] = exclusion_weight
            excluded_mask[i] = True
            continue

        # Get tree distance using code strings as keys
        key = (anchor_code, negative_code)
        if key not in distance_lookup:
            # If distance not found, use a default large distance
            tree_distance = 12.0
        else:
            tree_distance = distance_lookup[key]

        # Sibling masking: set weight to 0 if distance == 2 (siblings)
        if tree_distance == 2.0:
            weights[i] = 0.0
            continue

        # Inverse tree distance weighting: P(n) ∝ 1 / D_tree(a, n)^α
        if tree_distance > 0:
            weights[i] = 1.0 / (tree_distance**alpha)
        else:
            weights[i] = 0.0

    return weights, excluded_mask


def _sample_negatives_phase1(
    anchor_code: str,
    anchor_idx: int,
    candidate_negatives: List[Dict[str, Any]],
    n_negatives: int,
    distance_lookup: Dict[Tuple[str, str], float],
    excluded_map: Dict[str, Set[str]],
    code_to_idx: Dict[str, int],
    alpha: float = 1.5,
    exclusion_weight: float = 100.0,
    seed: Optional[int] = None,
) -> List[Dict[str, Any]]:
    '''
    Sample negatives using Phase 1 tree-distance based sampling.

    Args:
        anchor_code: Anchor code
        anchor_idx: Anchor index
        candidate_negatives: List of candidate negative dictionaries
        n_negatives: Number of negatives to sample
        distance_lookup: Dictionary mapping (anchor_idx, negative_idx) -> tree_distance
        excluded_map: Dictionary mapping code -> set of excluded codes
        code_to_idx: Mapping from code to index
        alpha: Exponent for inverse tree distance weighting
        exclusion_weight: High constant weight for excluded codes
        seed: Random seed for sampling

    Returns:
        List of sampled negative dictionaries
    '''
    if len(candidate_negatives) == 0:
        return []

    # Compute weights and exclusion flags
    weights, excluded_mask = _compute_phase1_weights(
        anchor_code=anchor_code,
        anchor_idx=anchor_idx,
        candidate_negatives=candidate_negatives,
        distance_lookup=distance_lookup,
        excluded_map=excluded_map,
        code_to_idx=code_to_idx,
        alpha=alpha,
        exclusion_weight=exclusion_weight,
    )

    # Normalize weights
    total_weight = weights.sum()
    if total_weight == 0:
        # If all weights are zero (e.g., all siblings), fall back to uniform sampling
        logger.warning(f'All weights zero for anchor {anchor_code}, using uniform sampling')
        weights = np.ones(len(candidate_negatives))
        total_weight = len(candidate_negatives)

    probabilities = weights / total_weight

    # Sample using numpy
    rng = np.random.default_rng(seed)
    n_sample = min(n_negatives, len(candidate_negatives))
    sampled_indices = rng.choice(
        len(candidate_negatives), size=n_sample, replace=False, p=probabilities
    )

    excluded_chosen = excluded_mask[sampled_indices].sum()
    if n_sample > 0:
        exclusion_ratio = excluded_chosen / n_sample
        logger.info(
            f'Anchor {anchor_code}: {exclusion_ratio:.2%} negatives from explicit exclusions '
            f'({excluded_chosen}/{n_sample})'
        )

    sampled = []
    for idx in sampled_indices:
        neg = dict(candidate_negatives[idx])
        neg['explicit_exclusion'] = bool(excluded_mask[idx])
        sampled.append(neg)

    return sampled


# -------------------------------------------------------------------------------------------------
# Cache utilities
# -------------------------------------------------------------------------------------------------


def _get_final_cache_path(cfg: StreamingConfig) -> Path:
    '''Get the cache file path for streaming query cache.'''
    import hashlib
    import json

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
    cache_key = hashlib.sha256(config_str.encode()).hexdigest()[:16]
    triplets_path = Path(cfg.triplets_parquet)
    cache_dir = triplets_path.parent / 'streaming_cache'
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f'streaming_final_{cache_key}.pkl'


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

    # Load codes and indices
    code_to_idx_dict = get_indices_codes('code_to_idx')
    idx_to_code_dict = get_indices_codes('idx_to_code')

    # Type assertions for type checker
    assert isinstance(code_to_idx_dict, dict), 'code_to_idx must be a dict'
    assert isinstance(idx_to_code_dict, dict), 'idx_to_code must be a dict'
    code_to_idx: Dict[str, int] = code_to_idx_dict  # type: ignore
    idx_to_code: Dict[int, str] = idx_to_code_dict  # type: ignore

    # Load Phase 1 sampling data if needed
    distance_lookup: Optional[Dict[Tuple[str, str], float]] = None
    excluded_map: Optional[Dict[str, Set[str]]] = None

    if cfg.use_phase1_sampling:
        logger.info(
            f'{worker_id} Phase 1 sampling enabled - loading distance matrix and excluded codes...'
        )
        distance_lookup = _load_distance_matrix(
            cfg.distance_matrix_parquet, code_to_idx, idx_to_code
        )
        excluded_map = _load_excluded_codes(cfg.descriptions_parquet, code_to_idx)

    # Organize codes by level
    level_dict = defaultdict(list)
    for code in code_to_idx.keys():
        if isinstance(code, str):
            level_dict[len(code)].append(code)

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
    filters = []
    if cfg.relation_margin is not None:
        filters.append(pl.col('relation_margin').is_in(cfg.relation_margin))
    if cfg.distance_margin is not None:
        filters.append(pl.col('distance_margin').is_in(cfg.distance_margin))
    if cfg.positive_level is not None:
        filters.append(pl.col('positive_level').is_in(cfg.positive_level))
    if cfg.positive_relation is not None:
        filters.append(pl.col('positive_relation').is_in(cfg.positive_relation))
    if cfg.positive_distance is not None:
        filters.append(pl.col('positive_distance').is_in(cfg.positive_distance))
    if cfg.negative_level is not None:
        filters.append(pl.col('negative_level').is_in(cfg.negative_level))
    if cfg.negative_relation is not None:
        filters.append(pl.col('negative_relation').is_in(cfg.negative_relation))
    if cfg.negative_distance is not None:
        filters.append(pl.col('negative_distance').is_in(cfg.negative_distance))

    if not filters:
        filters = [pl.col('anchor_idx').ge(0)]

    import operator
    from functools import reduce

    filter_expr = reduce(operator.and_, filters)

    # Build lazy query
    logger.debug(f'{worker_id} Scanning {len(dataset_files)} parquet files...')
    df_0 = pl.scan_parquet(dataset_files).filter(filter_expr)

    # Group by anchor and positive, collect negatives
    df_1 = (
        df_0.select(
            'anchor_idx',
            'anchor_code',
            'positive_idx',
            'positive_code',
            'negative_idx',
            'negative_code',
            'relation_margin',
            'distance_margin',
        )
        .group_by('anchor_idx', 'anchor_code', 'positive_idx', 'positive_code', maintain_order=True)
        .agg(
            negatives=pl.struct(
                [
                    pl.col('negative_idx'),
                    pl.col('negative_code'),
                    pl.col('relation_margin'),
                    pl.col('distance_margin'),
                ]
            )
        )
        .with_columns(positives_count=pl.col('negatives').list.len())
        .filter(pl.col('positives_count').gt(0))
        .with_columns(
            positives_count=pl.min_horizontal(pl.col('positives_count'), pl.lit(cfg.n_positives))
        )
        .with_columns(
            negatives=pl.col('negatives').list.sample(
                pl.col('positives_count'), shuffle=True, seed=cfg.seed
            )
        )
        .drop('positives_count')
        .explode('negatives')
        .unnest('negatives')
    )

    # Execute query
    logger.info(f'{worker_id} Executing Polars query...')
    df_1 = df_1.collect()
    logger.info(f'{worker_id} Query complete: {len(df_1)} rows')

    # Group by (anchor, positive) for negative sampling
    df_2 = df_1.group_by(
        'anchor_idx', 'anchor_code', 'positive_idx', 'positive_code', maintain_order=True
    ).agg(
        candidate_negatives=pl.struct(
            [
                pl.col('negative_idx'),
                pl.col('negative_code'),
                pl.col('relation_margin'),
                pl.col('distance_margin'),
            ]
        )
    )

    # Convert to list of dicts for iteration
    rows = df_2.to_dicts()

    # Sample negatives for each (anchor, positive) pair
    for row in rows:
        anchor_idx = row['anchor_idx']
        anchor_code = row['anchor_code']
        positive_idx = row['positive_idx']
        positive_code = row['positive_code']
        candidate_negatives = row['candidate_negatives']

        # Apply Phase 1 sampling if enabled
        if cfg.use_phase1_sampling and distance_lookup is not None and excluded_map is not None:
            sampled_negatives = _sample_negatives_phase1(
                anchor_code=anchor_code,
                anchor_idx=anchor_idx,
                candidate_negatives=candidate_negatives,
                n_negatives=cfg.n_negatives,
                distance_lookup=distance_lookup,
                excluded_map=excluded_map,
                code_to_idx=code_to_idx,
                alpha=cfg.phase1_alpha,
                exclusion_weight=cfg.phase1_exclusion_weight,
                seed=cfg.seed,
            )
        else:
            # Default: sample uniformly
            import random

            rng = random.Random(cfg.seed)
            n_sample = min(cfg.n_negatives, len(candidate_negatives))
            sampled_negatives = rng.sample(candidate_negatives, n_sample)

        yield {
            'anchor_idx': anchor_idx,
            'anchor_code': anchor_code,
            'positive_idx': positive_idx,
            'positive_code': positive_code,
            'negatives': sampled_negatives,
        }


# -------------------------------------------------------------------------------------------------
# Streaming dataset generator
# -------------------------------------------------------------------------------------------------


def create_streaming_dataset(
    token_cache: Dict[int, Dict[str, Any]], cfg: StreamingConfig
) -> Iterator[Dict[str, Any]]:
    '''Create streaming dataset that yields triplets with tokenized embeddings.

    For multi-level supervision (Issue #18):
    - For 6-digit anchors, yields all ancestor positives (levels 5, 4, 3, 2)
    - Each positive uses the same set of negatives
    - Gradients are accumulated across all positive levels in training_step

    Sampling responsibilities:
    - Applies Phase 1 sampling (inverse tree-distance weighting, sibling masking,
      explicit exclusion prioritization) when `cfg.use_phase1_sampling` is enabled.
    - Emits negatives with an `explicit_exclusion` flag for downstream logging/analysis.
    - Model layer performs Phase 2/3 tasks (hard negative mining, router-guided sampling,
      false-negative masking) after this iterator provides candidate negatives.
    '''

    # Load taxonomy and code mappings for ancestor lookup
    code_to_idx = get_indices_codes('code_to_idx')
    taxonomy = _taxonomy(cfg.descriptions_parquet)

    # Get triplets iterator
    triplets_iterator = create_streaming_generator(cfg)

    # Buffer triplets by anchor to collect all positives
    current_anchor: Optional[Tuple[int, str]] = None
    buffered_triplets: List[Dict[str, Any]] = []

    def yield_anchor_group(anchor_key: Tuple[int, str], triplets_list: List[Dict[str, Any]]):
        '''Yield grouped triplets with all ancestor positives for an anchor.'''
        anchor_idx, anchor_code = anchor_key

        try:
            anchor_embedding = {k: v for k, v in token_cache[anchor_idx].items() if k != 'code'}
        except KeyError as e:
            logger.error(f'KeyError accessing token_cache[{anchor_idx}]: {e}')
            raise

        # For 6-digit anchors, get all ancestor positives (levels 5, 4, 3, 2)
        anchor_level = len(anchor_code)
        positives_list = []

        if anchor_level == 6:
            # Get all ancestor codes using taxonomy
            anchor_row = taxonomy.filter(pl.col('code_6').eq(anchor_code))
            if anchor_row.height > 0:
                row = anchor_row.row(0, named=True)
                ancestor_codes = [row['code_5'], row['code_4'], row['code_3'], row['code_2']]

                # For each ancestor level, find matching positives from triplets_list
                for ancestor_code in ancestor_codes:
                    if ancestor_code in code_to_idx:
                        ancestor_idx = code_to_idx[ancestor_code]
                        # Find triplets with this positive
                        matching_triplets = [
                            t for t in triplets_list if t['positive_idx'] == ancestor_idx
                        ]
                        if matching_triplets:
                            # Use the first matching triplet's positive
                            triplet = matching_triplets[0]
                            # Ensure ancestor_idx is an int for token_cache access
                            if not isinstance(ancestor_idx, int):
                                ancestor_idx_int = int(ancestor_idx)
                            else:
                                ancestor_idx_int = ancestor_idx
                            positive_embedding = {
                                k: v
                                for k, v in token_cache[ancestor_idx_int].items()
                                if k != 'code'
                            }
                            positives_list.append(
                                {
                                    'positive_idx': ancestor_idx,
                                    'positive_code': ancestor_code,
                                    'positive_embedding': positive_embedding,
                                    'negatives': triplet['negatives'],
                                }
                            )

        # If no ancestor positives found (or anchor is not 6-digit), use the original triplets
        if not positives_list:
            # Fall back to original behavior: use first triplet
            if triplets_list:
                triplet = triplets_list[0]
                positive_idx = triplet['positive_idx']
                positive_code = triplet['positive_code']
                # Ensure positive_idx is an int for token_cache access
                if not isinstance(positive_idx, int):
                    positive_idx_int = int(positive_idx)
                else:
                    positive_idx_int = positive_idx
                positive_embedding = {
                    k: v for k, v in token_cache[positive_idx_int].items() if k != 'code'
                }
                positives_list.append(
                    {
                        'positive_idx': positive_idx,
                        'positive_code': positive_code,
                        'positive_embedding': positive_embedding,
                        'negatives': triplet['negatives'],
                    }
                )

        # Process negatives for all positives (use the same negatives for all)
        # Get negatives from the first positive (they should be the same for all ancestor levels)
        negatives = []
        if positives_list:
            for negative in positives_list[0]['negatives']:
                negative_idx = negative['negative_idx']
                negative_code = negative['negative_code']
                negative_embedding = {
                    k: v for k, v in token_cache[negative_idx].items() if k != 'code'
                }

                negatives.append(
                    {
                        'negative_idx': negative_idx,
                        'negative_code': negative_code,
                        'negative_embedding': negative_embedding,
                        'relation_margin': negative['relation_margin'],
                        'distance_margin': negative['distance_margin'],
                        'explicit_exclusion': negative.get('explicit_exclusion', False),
                    }
                )

        yield {
            'anchor_idx': anchor_idx,
            'anchor_code': anchor_code,
            'anchor_embedding': anchor_embedding,
            'positives': positives_list,  # List of positives (one per ancestor level)
            'negatives': negatives,
        }

    # Iterate through triplets and group by anchor
    for triplets in triplets_iterator:
        anchor_key = (triplets['anchor_idx'], triplets['anchor_code'])

        if current_anchor is None:
            current_anchor = anchor_key
            buffered_triplets = [triplets]
        elif current_anchor == anchor_key:
            # Same anchor, add to buffer
            buffered_triplets.append(triplets)
        else:
            # New anchor, yield previous group
            yield from yield_anchor_group(current_anchor, buffered_triplets)
            current_anchor = anchor_key
            buffered_triplets = [triplets]

    # Yield final group
    if current_anchor is not None:
        yield from yield_anchor_group(current_anchor, buffered_triplets)
