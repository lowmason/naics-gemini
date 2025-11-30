# -------------------------------------------------------------------------------------------------
# Imports and settings
# -------------------------------------------------------------------------------------------------

import hashlib
import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

import numpy as np
import polars as pl

from naics_embedder.data.positive_sampling import create_positive_sampler
from naics_embedder.utils.config import SamplingConfig, SansStaticConfig, StreamingConfig
from naics_embedder.utils.utilities import get_indices_codes

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------------------------------
# Phase 1 Sampling Utilities
# -------------------------------------------------------------------------------------------------

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
    distance_lookup: Dict[Tuple[str, str], float] = {}

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

def _load_excluded_codes(descriptions_path: str,
                         code_to_idx: Optional[Dict[str, int]] = None) -> Dict[str, Set[str]]:
    '''
    Load excluded codes from descriptions parquet.

    Args:
        descriptions_path: Path to descriptions parquet file

    Returns:
        Dictionary mapping code -> set of excluded codes
    '''
    logger.info('Loading excluded codes for Phase 1 sampling...')

    df = (
        pl.read_parquet(descriptions_path).select('code', 'excluded_codes').filter(
            pl.col('excluded_codes').is_not_null()
        )
    )

    excluded_map: Dict[str, Set[str]] = {}
    unknown_codes = 0
    for row in df.iter_rows(named=True):
        code = row['code']
        excluded_list = row['excluded_codes']
        if excluded_list:
            filtered: Set[str] = set()
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
        logger.debug(
            f'Anchor {anchor_code}: {exclusion_ratio:.2%} negatives from explicit exclusions '
            f'({excluded_chosen}/{n_sample})'
        )

    sampled = []
    for idx in sampled_indices:
        neg = dict(candidate_negatives[idx])
        neg['explicit_exclusion'] = bool(excluded_mask[idx])
        sampled.append(neg)

    return sampled

def _sample_negatives_sans_static(
    anchor_code: str,
    candidate_negatives: List[Dict[str, Any]],
    n_negatives: int,
    distance_lookup: Dict[Tuple[str, str], float],
    sans_cfg: 'SansStaticConfig',
    seed: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    '''
    Sample negatives using static near/far buckets (SANS baseline).

    Args:
        anchor_code: Anchor code
        candidate_negatives: Candidate negatives list
        n_negatives: Number of negatives to sample
        distance_lookup: Tree distance lookup (code pairs -> distance)
        sans_cfg: SANS configuration parameters
        seed: Optional RNG seed

    Returns:
        Tuple of sampled negatives and sampling metadata for diagnostics.
    '''

    metadata = {
        'strategy': 'sans_static',
        'candidates_near': 0,
        'candidates_far': 0,
        'sampled_near': 0,
        'sampled_far': 0,
        'effective_near_weight': 0.0,
        'effective_far_weight': 0.0,
    }

    if not candidate_negatives:
        return [], metadata

    distances = []
    for neg in candidate_negatives:
        negative_code = neg['negative_code']
        distance = distance_lookup.get((anchor_code, negative_code))
        if distance is None:
            distance = distance_lookup.get((negative_code, anchor_code), sans_cfg.default_distance)
        distances.append(distance)

    near_indices = [
        idx for idx, dist in enumerate(distances) if dist <= sans_cfg.near_distance_threshold
    ]
    far_indices = [idx for idx in range(len(distances)) if idx not in near_indices]

    metadata['candidates_near'] = len(near_indices)
    metadata['candidates_far'] = len(far_indices)

    near_weight = sans_cfg.near_bucket_weight
    far_weight = sans_cfg.far_bucket_weight
    total_candidates = len(candidate_negatives)

    weights = np.zeros(total_candidates, dtype=np.float64)
    bucket_weight_total = 0.0

    if near_indices and near_weight > 0:
        bucket_weight_total += near_weight
        weights[near_indices] = near_weight / len(near_indices)
    if far_indices and far_weight > 0:
        bucket_weight_total += far_weight
        weights[far_indices] = far_weight / len(far_indices)

    if bucket_weight_total == 0:
        weights[:] = 1.0 / total_candidates
    else:
        weights /= bucket_weight_total

    metadata['effective_near_weight'] = float(weights[near_indices].sum()) if near_indices else 0.0
    metadata['effective_far_weight'] = float(weights[far_indices].sum()) if far_indices else 0.0

    rng = np.random.default_rng(seed)
    n_sample = min(n_negatives, total_candidates)
    sampled_indices = rng.choice(total_candidates, size=n_sample, replace=False, p=weights)

    near_index_set = set(near_indices)

    sampled = []
    for idx in sampled_indices:
        neg = dict(candidate_negatives[idx])
        neg.setdefault('explicit_exclusion', False)
        sampled.append(neg)

        if idx in near_index_set:
            metadata['sampled_near'] += 1
        else:
            metadata['sampled_far'] += 1

    return sampled, metadata

# -------------------------------------------------------------------------------------------------
# Negative candidate loading
# -------------------------------------------------------------------------------------------------

def _load_negative_candidates(triplets_parquet: str,
                              ) -> Dict[Tuple[int, int], List[Dict[str, Any]]]:
    '''Load all candidate negatives from triplets parquet files.

    Returns a dictionary mapping (anchor_idx, positive_idx) -> list of negative dicts.
    Each negative dict has: negative_idx, negative_code, relation_margin, distance_margin.
    '''
    logger.info('Loading negative candidates from triplets parquet...')

    dataset_files = [str(p) for p in Path(triplets_parquet).glob('**/*.parquet')]
    if not dataset_files:
        logger.warning(f'No parquet files found in {triplets_parquet}')
        return {}

    df = (
        pl.scan_parquet(dataset_files).select(
            'anchor_idx',
            'positive_idx',
            'negative_idx',
            'negative_code',
            'relation_margin',
            'distance_margin',
        ).collect()
    )

    logger.info(f'Loaded {len(df):,} negative candidate rows')

    # Group by (anchor, positive)
    result: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}
    for row in df.iter_rows(named=True):
        key = (row['anchor_idx'], row['positive_idx'])
        if key not in result:
            result[key] = []
        result[key].append(
            {
                'negative_idx': row['negative_idx'],
                'negative_code': row['negative_code'],
                'relation_margin': row['relation_margin'],
                'distance_margin': row['distance_margin'],
            }
        )

    logger.info(f'Grouped into {len(result):,} (anchor, positive) pairs')
    return result

# -------------------------------------------------------------------------------------------------
# Cache utilities
# -------------------------------------------------------------------------------------------------

def _get_final_cache_path(cfg: StreamingConfig) -> Path:
    '''Get the cache file path for streaming query cache.'''

    cache_dict = {
        'descriptions_parquet': str(cfg.descriptions_parquet),
        'relations_parquet': str(cfg.relations_parquet),
        'n_negatives': cfg.n_negatives,
        'seed': cfg.seed,
    }

    config_str = json.dumps(cache_dict, sort_keys=True)
    cache_key = hashlib.sha256(config_str.encode()).hexdigest()[:16]
    descriptions_path = Path(cfg.descriptions_parquet)
    cache_dir = descriptions_path.parent / 'streaming_cache'
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f'streaming_final_{cache_key}.pkl'

def _load_final_cache(cfg: StreamingConfig) -> Optional[List[Dict[str, Any]]]:
    '''Load cached streaming data if available.'''
    cache_path = _get_final_cache_path(cfg)
    if not cache_path.exists():
        return None

    try:
        with open(cache_path, 'rb') as f:
            data = pickle.load(f)
        logger.info(f'Loaded streaming cache from {cache_path}')
        return data
    except Exception as e:
        logger.warning(f'Failed to load cache: {e}')
        return None

def _save_final_cache(data: List[Dict[str, Any]], cfg: StreamingConfig) -> None:
    '''Save streaming data to cache.'''
    cache_path = _get_final_cache_path(cfg)
    try:
        temp_path = cache_path.with_suffix('.tmp')
        with open(temp_path, 'wb') as f:
            pickle.dump(data, f)
        temp_path.replace(cache_path)
        logger.info(f'Saved streaming cache to {cache_path}')
    except Exception as e:
        logger.warning(f'Failed to save cache: {e}')

# -------------------------------------------------------------------------------------------------
# Triplet batch generator
# -------------------------------------------------------------------------------------------------

def create_streaming_generator(cfg: StreamingConfig, sampling_cfg: Optional[SamplingConfig] = None
                               ) -> Iterator[Dict[str, Any]]:
    '''Create a generator that yields triplets for training.

    Uses taxonomy-based stratified positive sampling:
    - Stratum 0 (descendants): for levels 2-5, next-level descendants
    - Stratum 1 (ancestors): for levels 3-6, parent codes up to level 2
    - Stratum 2 (siblings): codes sharing the same parent

    Samples up to 4 positives per stratum, then samples negatives for each.
    '''

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

    # Resolve sampling configuration
    if sampling_cfg is None:
        sampling_cfg = SamplingConfig()
    sampling_strategy = sampling_cfg.strategy
    sans_cfg: SansStaticConfig = sampling_cfg.sans_static

    # Load Phase 1/SANS sampling data if needed
    distance_lookup: Optional[Dict[Tuple[str, str], float]] = None
    excluded_map: Optional[Dict[str, Set[str]]] = None

    requires_distance_lookup = cfg.use_phase1_sampling or sampling_strategy == 'sans_static'

    if requires_distance_lookup:
        logger.info(f'{worker_id} Loading tree distance matrix for sampling strategy...')
        distance_lookup = _load_distance_matrix(
            cfg.distance_matrix_parquet, code_to_idx, idx_to_code
        )

    if cfg.use_phase1_sampling:
        logger.info(f'{worker_id} Loading excluded codes for Phase 1 sampling...')
        excluded_map = _load_excluded_codes(cfg.descriptions_parquet, code_to_idx)

    # Create positive sampler using taxonomy-based stratification
    logger.info(f'{worker_id} Creating positive sampler...')
    positive_sampler = create_positive_sampler(
        descriptions_parquet=cfg.descriptions_parquet,
        relations_parquet=cfg.relations_parquet,
        max_per_stratum=4,
        seed=cfg.seed,
    )

    # Load negative candidates from triplets
    logger.info(f'{worker_id} Loading negative candidates...')
    negative_candidates = _load_negative_candidates(cfg.triplets_parquet)

    # Iterate over anchors and sample positives + negatives
    for anchor_idx in positive_sampler.anchors:
        anchor_code = idx_to_code.get(anchor_idx)
        if anchor_code is None:
            continue

        # Sample positives across strata
        positives = positive_sampler.sample_positives(anchor_idx)
        if not positives:
            continue

        # For each positive, sample negatives
        for positive in positives:
            positive_idx = positive['positive_idx']
            positive_code = positive['positive_code']

            # Get candidate negatives for this (anchor, positive) pair
            key = (anchor_idx, positive_idx)
            candidates = negative_candidates.get(key, [])

            if not candidates:
                # Try reverse lookup or skip
                continue

            # Apply sampling strategy
            sampling_metadata: Optional[Dict[str, Any]] = None

            if sampling_strategy == 'sans_static' and distance_lookup is not None:
                sampled_negatives, sampling_metadata = _sample_negatives_sans_static(
                    anchor_code=anchor_code,
                    candidate_negatives=candidates,
                    n_negatives=cfg.n_negatives,
                    distance_lookup=distance_lookup,
                    sans_cfg=sans_cfg,
                    seed=cfg.seed,
                )
            elif cfg.use_phase1_sampling and distance_lookup is not None and excluded_map is not None:
                sampled_negatives = _sample_negatives_phase1(
                    anchor_code=anchor_code,
                    anchor_idx=anchor_idx,
                    candidate_negatives=candidates,
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
                n_sample = min(cfg.n_negatives, len(candidates))
                sampled_negatives = rng.sample(candidates, n_sample)
                if sampling_strategy == 'sans_static':
                    logger.warning(
                        'SANS static sampling requested but tree distances were unavailable; '
                        'falling back to uniform sampling.'
                    )

            if not sampled_negatives:
                continue

            yield {
                'anchor_idx': anchor_idx,
                'anchor_code': anchor_code,
                'positive_idx': positive_idx,
                'positive_code': positive_code,
                'positive_level': positive['positive_level'],
                'stratum_id': positive['stratum_id'],
                'stratum_wgt': positive['stratum_wgt'],
                'negatives': sampled_negatives,
                'sampling_metadata': sampling_metadata,
            }

# -------------------------------------------------------------------------------------------------
# Streaming dataset generator
# -------------------------------------------------------------------------------------------------

def create_streaming_dataset(
    token_cache: Dict[int, Dict[str, Any]],
    cfg: StreamingConfig,
    sampling_cfg: Optional[SamplingConfig] = None,
) -> Iterator[Dict[str, Any]]:
    '''Create streaming dataset that yields triplets with tokenized embeddings.

    Uses taxonomy-based stratified positive sampling:
    - Samples positives from descendants (stratum 0), ancestors (stratum 1), siblings (stratum 2)
    - Up to 4 positives per stratum
    - Negatives sampled per positive using Phase 1 / SANS / uniform strategies

    Yields anchor-level samples with all positives grouped together.
    '''

    # Get triplets iterator
    triplets_iterator = create_streaming_generator(cfg, sampling_cfg)

    # Buffer triplets by anchor to collect all positives
    current_anchor: Optional[Tuple[int, str]] = None
    buffered_triplets: List[Dict[str, Any]] = []

    def yield_anchor_group(anchor_key: Tuple[int, str],
                           triplets_list: List[Dict[str, Any]]) -> Iterator[Dict[str, Any]]:
        '''Yield grouped triplets for an anchor.'''
        anchor_idx, anchor_code = anchor_key

        try:
            anchor_embedding = {k: v for k, v in token_cache[anchor_idx].items() if k != 'code'}
        except KeyError as e:
            logger.error(f'KeyError accessing token_cache[{anchor_idx}]: {e}')
            raise

        # Build positives list with embeddings
        positives_list = []
        all_negatives: Dict[int, Dict[str, Any]] = {}  # Dedupe negatives by idx
        sampling_metadata = None

        for triplet in triplets_list:
            positive_idx = triplet['positive_idx']
            positive_code = triplet['positive_code']

            # Get positive embedding
            if not isinstance(positive_idx, int):
                positive_idx_int = int(positive_idx)
            else:
                positive_idx_int = positive_idx

            try:
                positive_embedding = {
                    k: v
                    for k, v in token_cache[positive_idx_int].items() if k != 'code'
                }
            except KeyError:
                logger.warning(f'Missing token_cache for positive {positive_idx_int}, skipping')
                continue

            positives_list.append(
                {
                    'positive_idx': positive_idx,
                    'positive_code': positive_code,
                    'positive_level': triplet.get('positive_level', len(positive_code)),
                    'stratum_id': triplet.get('stratum_id', 0),
                    'stratum_wgt': triplet.get('stratum_wgt', 1.0),
                    'positive_embedding': positive_embedding,
                }
            )

            # Collect negatives (dedupe across positives)
            for neg in triplet.get('negatives', []):
                neg_idx = neg['negative_idx']
                if neg_idx not in all_negatives:
                    try:
                        neg_embedding = {
                            k: v
                            for k, v in token_cache[neg_idx].items() if k != 'code'
                        }
                    except KeyError:
                        logger.warning(f'Missing token_cache for negative {neg_idx}, skipping')
                        continue

                    all_negatives[neg_idx] = {
                        'negative_idx': neg_idx,
                        'negative_code': neg['negative_code'],
                        'negative_embedding': neg_embedding,
                        'relation_margin': neg.get('relation_margin', 0),
                        'distance_margin': neg.get('distance_margin', 0),
                        'explicit_exclusion': neg.get('explicit_exclusion', False),
                    }

            # Capture sampling metadata from first triplet
            if sampling_metadata is None:
                sampling_metadata = triplet.get('sampling_metadata')

        if not positives_list:
            return

        result: Dict[str, Any] = {
            'anchor_idx': anchor_idx,
            'anchor_code': anchor_code,
            'anchor_embedding': anchor_embedding,
            'positives': positives_list,
            'negatives': list(all_negatives.values()),
        }

        if sampling_metadata:
            result['sampling_metadata'] = sampling_metadata

        yield result

    # Iterate through triplets and group by anchor
    for triplet in triplets_iterator:
        anchor_key = (triplet['anchor_idx'], triplet['anchor_code'])

        if current_anchor is None:
            current_anchor = anchor_key
            buffered_triplets = [triplet]
        elif current_anchor == anchor_key:
            # Same anchor, add to buffer
            buffered_triplets.append(triplet)
        else:
            # New anchor, yield previous group
            yield from yield_anchor_group(current_anchor, buffered_triplets)
            current_anchor = anchor_key
            buffered_triplets = [triplet]

    # Yield final group
    if current_anchor is not None:
        yield from yield_anchor_group(current_anchor, buffered_triplets)
