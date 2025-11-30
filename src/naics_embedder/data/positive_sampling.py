# -------------------------------------------------------------------------------------------------
# Positive Sampling Helper
#
# Taxonomy-based positive enumeration and stratified sampling for contrastive training.
# Shared by both text_model and graph_model streaming pipelines.
# -------------------------------------------------------------------------------------------------

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import polars as pl

from naics_embedder.utils.utilities import get_indices_codes

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------------------------------
# Taxonomy utilities
# -------------------------------------------------------------------------------------------------

def build_taxonomy(descriptions_parquet: str) -> pl.DataFrame:
    '''Build taxonomy dataframe from descriptions parquet.

    Returns a dataframe with columns code_2, code_3, code_4, code_5, code_6, code
    for each 6-digit NAICS code. Handles merged sectors (31-33, 44-45, 48-49).
    '''
    return (
        pl.read_parquet(descriptions_parquet).filter(pl.col('level').eq(6)).select('code').sort(
            pl.col('code').cast(pl.UInt32)
        ).unique(maintain_order=True).select(
            code_2=pl.col('code').str.slice(0, 2),
            code_3=pl.col('code').str.slice(0, 3),
            code_4=pl.col('code').str.slice(0, 4),
            code_5=pl.col('code').str.slice(0, 5),
            code_6=pl.col('code').str.slice(0, 6),
        ).with_columns(
            # Merge sectors 31-33, 44-45, 48-49
            code_2=pl.when(pl.col('code_2').is_in(['31', '32', '33'])).then(pl.lit('31')).when(
                pl.col('code_2').is_in(['44', '45'])
            ).then(pl.lit('44')).when(pl.col('code_2').is_in(['48', '49'])
                                      ).then(pl.lit('48')).otherwise(pl.col('code_2'))
        ).with_columns(
            code=pl.concat_str(pl.col('code_2'), pl.col('code_6').str.slice(2, 4), separator='')
        )
    )

def build_anchor_list(descriptions_parquet: str) -> pl.DataFrame:
    '''Build list of all possible anchors from descriptions parquet.

    Returns a dataframe with columns: level, anchor (code string).
    '''
    return (
        pl.read_parquet(descriptions_parquet).filter(pl.col('level').ge(2)).select(
            level=pl.col('level'), anchor=pl.col('code')
        ).unique().sort(pl.col('level'),
                        pl.col('anchor').cast(pl.UInt32))
    )

# -------------------------------------------------------------------------------------------------
# Descendants stratum (stratum_id=0)
# -------------------------------------------------------------------------------------------------

def _linear_skip(anchor: str, taxonomy: pl.DataFrame) -> List[str]:
    '''Find the next level of descendants for an anchor.

    For a given anchor, finds the smallest set of distinct descendants
    at the next hierarchical level that have more than one member.
    '''
    lvl = len(anchor)
    anchor_code = f'code_{lvl}'
    codes = [f'code_{i}' for i in range(lvl + 1, 7)]

    for code in codes:
        candidate = taxonomy.filter(pl.col(anchor_code).eq(anchor)
                                    ).get_column(code).unique().to_list()

        if lvl == 5:
            return candidate
        elif len(candidate) > 1:
            return sorted(set(candidate))

    return taxonomy.filter(pl.col(anchor_code).eq(anchor)).get_column('code_6').unique().to_list()

def _build_descendants(anchors: pl.DataFrame, taxonomy: pl.DataFrame) -> pl.DataFrame:
    '''Build descendants stratum for parent anchors (levels 2-5).

    Returns dataframe with columns: level, anchor, positive (list of structs),
    num_positives. Each positive struct has stratum_id=0, positive code, stratum_wgt.
    '''
    parent_anchors = anchors.filter(pl.col('level').lt(6)
                                    ).get_column('anchor').unique().sort().to_list()

    parent_stratum = []
    for anchor in parent_anchors:
        descendants = _linear_skip(anchor, taxonomy)
        if descendants:
            parent_stratum.append({'anchor': anchor, 'stratum': descendants})

    if not parent_stratum:
        # Return empty dataframe with correct schema
        return pl.DataFrame(
            schema={
                'level':
                pl.Int64,
                'anchor':
                pl.Utf8,
                'positive':
                pl.List(
                    pl.Struct(
                        {
                            'stratum_id': pl.Int32,
                            'positive': pl.Utf8,
                            'stratum_wgt': pl.Float64
                        }
                    )
                ),
                'num_positives':
                pl.UInt32,
            }
        )

    return (
        pl.DataFrame(data=parent_stratum, schema={
            'anchor': pl.Utf8,
            'stratum': pl.List(pl.Utf8)
        }).filter(pl.col('stratum').is_not_null()).select(
            level=pl.col('anchor').str.len_chars(),
            anchor=pl.col('anchor'),
            positive=pl.col('stratum'),
            num_positives=pl.col('stratum').list.len(),
        ).explode('positive').with_columns(
            stratum_id=pl.lit(0), stratum_wgt=pl.col('num_positives').cast(pl.Float64).pow(-1)
        ).with_columns(
            positive=pl.struct(pl.col('stratum_id'), pl.col('positive'), pl.col('stratum_wgt'))
        ).group_by('level', 'anchor', maintain_order=True).agg(
            positive=pl.col('positive'), num_positives=pl.col('positive').len()
        )
    )

# -------------------------------------------------------------------------------------------------
# Ancestors stratum (stratum_id=1)
# -------------------------------------------------------------------------------------------------

def _build_ancestors_6(anchors: pl.DataFrame, taxonomy: pl.DataFrame) -> pl.DataFrame:
    '''Build ancestors stratum for 6-digit anchors.

    Returns dataframe with columns: level, anchor, positive (list of structs),
    num_positives. Each positive struct has stratum_id=1, positive code, stratum_wgt=0.25.
    '''
    return (
        anchors.filter(pl.col('level').eq(6)).join(
            taxonomy, left_on='anchor', right_on='code_6', how='inner'
        ).select(
            level=pl.col('level'),
            anchor=pl.col('anchor'),
            code_5=pl.col('code_5'),
            code_4=pl.col('code_4'),
            code_3=pl.col('code_3'),
            code_2=pl.col('code_2'),
        ).unpivot(
            ['code_5', 'code_4', 'code_3', 'code_2'],
            index=['level', 'anchor'],
            variable_name='ancestor_level',
            value_name='ancestor',
        ).with_columns(
            ancestor_level=pl.col('ancestor_level').str.slice(5, 1).cast(pl.Int8).add(-6).mul(-1)
        ).sort('level', 'anchor', 'ancestor_level').with_columns(
            stratum_id=pl.lit(1), positive=pl.col('ancestor'), stratum_wgt=pl.lit(0.25)
        ).with_columns(
            positive=pl.struct(pl.col('stratum_id'), pl.col('positive'), pl.col('stratum_wgt'))
        ).group_by('level', 'anchor', maintain_order=True).agg(
            positive=pl.col('positive'), num_positives=pl.col('positive').len()
        )
    )

def _build_ancestors_level(ancestors_df: pl.DataFrame, level: int) -> pl.DataFrame:
    '''Build ancestors stratum for a specific level (5, 4, or 3) by deriving from higher level.'''
    list_len = level - 2

    return (
        ancestors_df.select(
            level=pl.col('level').sub(1),
            anchor=pl.col('anchor').str.slice(0, level),
            positive=pl.col('positive').list.slice(1, list_len),
            num_positives=pl.col('num_positives').sub(1),
        ).explode('positive').unnest('positive').unique().with_columns(
            positive_level=pl.col('positive').str.len_chars()
        ).sort('level', 'anchor', 'positive_level', descending=[False, False, True]).with_columns(
            stratum_wgt=pl.col('num_positives').cast(pl.Float64).pow(-1)
        ).with_columns(
            positive=pl.struct(pl.col('stratum_id'), pl.col('positive'), pl.col('stratum_wgt'))
        ).group_by('level', 'anchor', maintain_order=True).agg(
            positive=pl.col('positive'), num_positives=pl.col('positive').len()
        )
    )

# -------------------------------------------------------------------------------------------------
# Siblings stratum (stratum_id=2)
# -------------------------------------------------------------------------------------------------

def _build_siblings(relations_parquet: str, anchors: pl.DataFrame) -> pl.DataFrame:
    '''Build siblings stratum from relations parquet.

    Returns dataframe with columns: level, anchor, positive (list of structs),
    num_positives. Each positive struct has stratum_id=2, positive code, stratum_wgt.
    '''
    return (
        pl.read_parquet(relations_parquet).filter(pl.col('relation_id').eq(2)).select(
            anchor=pl.col('code_i'),
            positive=pl.col('code_j'),
            stratum_wgt=pl.col('relation_id').rank(method='dense'),
        ).join(anchors, on='anchor', how='inner').with_columns(
            stratum_id=pl.lit(2), stratum_wgt=pl.lit(0.5).pow(pl.col('stratum_wgt'))
        ).with_columns(stratum_norm=pl.col('stratum_wgt').sum().over('anchor')).with_columns(
            positive=pl.struct(
                pl.col('stratum_id'),
                pl.col('positive'),
                pl.col('stratum_wgt').truediv(pl.col('stratum_norm')),
            )
        ).group_by('level', 'anchor', maintain_order=True).agg(
            positive=pl.col('positive'), num_positives=pl.col('positive').len()
        )
    )

# -------------------------------------------------------------------------------------------------
# Main enumeration function
# -------------------------------------------------------------------------------------------------

def enumerate_positives(
    descriptions_parquet: str = './data/naics_descriptions.parquet',
    relations_parquet: str = './data/naics_relations.parquet',
) -> pl.DataFrame:
    '''Enumerate all possible positives for each anchor across three strata.

    Strata:
        0 - Descendants: for levels 2-5, the next level of descendants
        1 - Ancestors: for levels 3-6, the parent codes up to level 2
        2 - Siblings: codes sharing the same parent (relation_id=2)

    Args:
        descriptions_parquet: Path to descriptions parquet file
        relations_parquet: Path to relations parquet file

    Returns:
        DataFrame with columns:
            anchor_idx, positive_idx: integer indices
            anchor_code, positive_code: string codes
            anchor_level, positive_level: hierarchy levels
            stratum_id: 0=descendants, 1=ancestors, 2=siblings
            stratum_wgt: normalized sampling weight within stratum
    '''
    taxonomy = build_taxonomy(descriptions_parquet)
    anchors = build_anchor_list(descriptions_parquet)
    code_to_idx = get_indices_codes('code_to_idx')

    # Build each stratum
    descendants = _build_descendants(anchors, taxonomy)

    ancestors_6 = _build_ancestors_6(anchors, taxonomy)
    ancestors_5 = _build_ancestors_level(ancestors_6, 5)
    ancestors_4 = _build_ancestors_level(ancestors_5, 4)
    ancestors_3 = _build_ancestors_level(ancestors_4, 3)
    ancestors = pl.concat([ancestors_5, ancestors_4, ancestors_3])

    siblings = _build_siblings(relations_parquet, anchors)

    # Cast to consistent types before concatenation
    def _normalize_schema(df: pl.DataFrame) -> pl.DataFrame:
        if df.height == 0:
            return df
        return df.with_columns(
            level=pl.col('level').cast(pl.Int64),
            num_positives=pl.col('num_positives').cast(pl.UInt32),
        )

    descendants = _normalize_schema(descendants)
    ancestors = _normalize_schema(ancestors)
    siblings = _normalize_schema(siblings)

    # Combine all strata
    return (
        pl.concat([descendants, ancestors, siblings],
                  how='diagonal_relaxed').explode('positive').unnest('positive').select(
                      anchor_idx=pl.col('anchor').replace(code_to_idx).cast(pl.UInt32),
                      positive_idx=pl.col('positive').replace(code_to_idx).cast(pl.UInt32),
                      anchor_code=pl.col('anchor'),
                      positive_code=pl.col('positive'),
                      anchor_level=pl.col('level'),
                      positive_level=pl.col('positive').str.len_chars(),
                      stratum_id=pl.col('stratum_id'),
                      stratum_wgt=pl.col('stratum_wgt'),
                  ).sort('anchor_idx', 'stratum_id', 'positive_idx')
    )

# -------------------------------------------------------------------------------------------------
# Stratified positive sampling
# -------------------------------------------------------------------------------------------------

class PositiveSampler:
    '''Stratified positive sampler for contrastive training.

    Groups positives by anchor and stratum, then samples up to max_per_stratum
    positives from each stratum using the normalized stratum weights.
    '''

    def __init__(
        self,
        positives_df: pl.DataFrame,
        max_per_stratum: int = 4,
        seed: int = 42,
    ):
        '''Initialize sampler.

        Args:
            positives_df: DataFrame from enumerate_positives()
            max_per_stratum: Maximum positives to sample per stratum (default 4)
            seed: Random seed for reproducibility
        '''
        self.max_per_stratum = max_per_stratum
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # Build lookup: anchor_idx -> {stratum_id -> list of (positive_idx, positive_code, weight)}
        self._anchor_strata: Dict[int, Dict[int, List[Dict[str, Any]]]] = {}

        for row in positives_df.iter_rows(named=True):
            anchor_idx = int(row['anchor_idx'])
            stratum_id = int(row['stratum_id'])

            if anchor_idx not in self._anchor_strata:
                self._anchor_strata[anchor_idx] = {}
            if stratum_id not in self._anchor_strata[anchor_idx]:
                self._anchor_strata[anchor_idx][stratum_id] = []

            self._anchor_strata[anchor_idx][stratum_id].append(
                {
                    'positive_idx': int(row['positive_idx']),
                    'positive_code': row['positive_code'],
                    'positive_level': int(row['positive_level']),
                    'stratum_wgt': float(row['stratum_wgt']),
                }
            )

        # Get list of all anchors
        self._anchors = sorted(self._anchor_strata.keys())
        logger.info(f'PositiveSampler initialized with {len(self._anchors)} anchors')

    @property
    def anchors(self) -> List[int]:
        '''Return list of anchor indices.'''
        return self._anchors

    def sample_positives(self, anchor_idx: int) -> List[Dict[str, Any]]:
        '''Sample positives for an anchor across all available strata.

        Args:
            anchor_idx: Anchor index

        Returns:
            List of positive dicts with keys: positive_idx, positive_code,
            positive_level, stratum_id, stratum_wgt
        '''
        if anchor_idx not in self._anchor_strata:
            return []

        sampled = []
        for stratum_id, candidates in self._anchor_strata[anchor_idx].items():
            if not candidates:
                continue

            # Normalize weights within stratum
            weights = np.array([c['stratum_wgt'] for c in candidates])
            total = weights.sum()
            if total > 0:
                probs = weights / total
            else:
                probs = np.ones(len(candidates)) / len(candidates)

            # Sample up to max_per_stratum
            n_sample = min(self.max_per_stratum, len(candidates))
            indices = self.rng.choice(len(candidates), size=n_sample, replace=False, p=probs)

            for idx in indices:
                c = candidates[idx]
                sampled.append(
                    {
                        'positive_idx': c['positive_idx'],
                        'positive_code': c['positive_code'],
                        'positive_level': c['positive_level'],
                        'stratum_id': stratum_id,
                        'stratum_wgt': c['stratum_wgt'],
                    }
                )

        return sampled

    def get_anchor_code(self, anchor_idx: int) -> Optional[str]:
        '''Get anchor code for an anchor index.'''
        idx_to_code_raw = get_indices_codes('idx_to_code')
        if isinstance(idx_to_code_raw, dict):
            idx_to_code: Dict[int, str] = idx_to_code_raw  # type: ignore
            return idx_to_code.get(anchor_idx)
        return None

# -------------------------------------------------------------------------------------------------
# Factory function
# -------------------------------------------------------------------------------------------------

def create_positive_sampler(
    descriptions_parquet: str = './data/naics_descriptions.parquet',
    relations_parquet: str = './data/naics_relations.parquet',
    max_per_stratum: int = 4,
    seed: int = 42,
) -> PositiveSampler:
    '''Create a PositiveSampler from data files.

    Args:
        descriptions_parquet: Path to descriptions parquet
        relations_parquet: Path to relations parquet
        max_per_stratum: Maximum positives per stratum (default 4)
        seed: Random seed

    Returns:
        Initialized PositiveSampler
    '''
    positives_df = enumerate_positives(descriptions_parquet, relations_parquet)
    return PositiveSampler(positives_df, max_per_stratum=max_per_stratum, seed=seed)
