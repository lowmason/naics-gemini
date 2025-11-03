# -------------------------------------------------------------------------------------------------
# Imports and settings
# -------------------------------------------------------------------------------------------------

import logging
import random
from dataclasses import dataclass
from typing import Dict, List, Optional

import polars as pl
from torch.utils.data import IterableDataset

from naics_gemini.data_loader.tokenization_cache import TokenizationCache

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------------------------------

@dataclass
class CurriculumConfig:

    positive_levels: Optional[List[int]] = None
    positive_distance_min: Optional[float] = None
    positive_distance_max: Optional[float] = None
    max_positives: Optional[int] = None

    difficulty_buckets: Optional[List[int]] = None
    bucket_percentages: Optional[Dict[int, float]] = None

    k_negatives: int = 16

    def __post_init__(self):

        if self.difficulty_buckets is None:
            self.difficulty_buckets = [1, 2, 3, 4, 5, 6, 7, 8]

        if self.bucket_percentages is None:
            n = len(self.difficulty_buckets)
            self.bucket_percentages = {b: 1.0 / n for b in self.difficulty_buckets}


# -------------------------------------------------------------------------------------------------
# Class to load and configure streaming dataset for curriculum learning
# -------------------------------------------------------------------------------------------------

class NAICSStreamingDataset(IterableDataset):

    HARDNESS_MAP = {
        8: ('excluded', True, 'unrelated', True),
        7: ('excluded', False, 'unrelated', False, 'distance_diff', 0.5),
        6: ('excluded', False, 'unrelated', False, 'distance_diff', 1.0),
        5: ('excluded', False, 'unrelated', False, 'distance_diff', 2.0),
        4: ('excluded', False, 'unrelated', False, 'distance_diff', (2.5, 3.0)),
        3: ('excluded', False, 'unrelated', False, 'distance_diff', (3.5, 4.0)),
        2: ('excluded', False, 'unrelated', False, 'distance_diff', (4.5, 6.5)),
        1: ('excluded', False, 'unrelated', True),
    }

    def __init__(
        self,
        triplets_path: str,
        token_cache: TokenizationCache,
        curriculum: CurriculumConfig,
        shuffle_buffer_size: int = 10000,
        seed: int = 42
    ):
        super().__init__()

        self.triplets_path = triplets_path
        self.token_cache = token_cache
        self.curriculum = curriculum
        self.shuffle_buffer_size = shuffle_buffer_size
        self.rng = random.Random(seed)

        self._anchor_to_positives = None
        self._positive_to_negatives_by_hardness = None


    def _build_indices(self):

        if self._anchor_to_positives is not None:
            return

        logger.info('Building triplet indices...')

        df = (
            pl
            .scan_parquet(
                self.triplets_path
            )
        )

        # Apply curriculum filters for positive levels
        if self.curriculum.positive_levels:
            df = (
                df
                .filter(
                    pl.col('positive_code')
                      .str.len_chars()
                      .is_in(self.curriculum.positive_levels)
                )
            )

        # Apply curriculum filters for positive min distance
        if self.curriculum.positive_distance_min is not None:

            df = (
                df
                .filter(
                    pl.col('positive_distance')
                      .gt(self.curriculum.positive_distance_min)
                )
            )

        # Apply curriculum filters for positive max distance
        if self.curriculum.positive_distance_max is not None:

            df = (
                df
                .filter(
                    pl.col('positive_distance').lt(self.curriculum.positive_distance_max)
                )
            )

        # Hardness bucketing
        df = (
            df
            .with_columns(
                hardness=pl.when(pl.col('excluded') & pl.col('unrelated')).then(pl.lit(8))
                          .when(pl.col('distance_diff').le(0.5)).then(pl.lit(7))
                          .when(pl.col('distance_diff').le(1.0)).then(pl.lit(6))
                          .when(pl.col('distance_diff').le(2.0)).then(pl.lit(5))
                          .when(pl.col('distance_diff').le(3.0)).then(pl.lit(4))
                          .when(pl.col('distance_diff').le(4.0)).then(pl.lit(3))
                          .when(pl.col('distance_diff').le(6.5)).then(pl.lit(2))
                          .when(~pl.col('excluded') & pl.col('unrelated')).then(pl.lit(1))
                          .otherwise(pl.lit(0))
            )
        )

        # Apply curriculum filters by hardness buckets
        df = (
            df
            .filter(
                pl.col('hardness')
                  .is_in(self.curriculum.difficulty_buckets) #type: ignore
            )
        )

        # Collect to build in-memory indices
        df_collected = (
            df
            .collect()
        )

        # Build anchor to positives mapping
        anchor_to_positives_iter = (
            df_collected
            .select('anchor_code', 'positive_code')
            .unique()
            .iter_rows(named=True)
        )

        # Build anchor to negatives mapping
        anchor_to_negatives_iter = (
            df_collected
            .iter_rows(named=True)
        )

        anchor_to_positives = {}
        for row in anchor_to_positives_iter:
            anchor = row['anchor_code']
            positive = row['positive_code']

            if anchor not in anchor_to_positives:
                anchor_to_positives[anchor] = []

            anchor_to_positives[anchor].append(positive)

        # Limit max positives per anchor if specified
        if self.curriculum.max_positives:
            for anchor in anchor_to_positives:
                positives = anchor_to_positives[anchor]
                if len(positives) > self.curriculum.max_positives:
                    anchor_to_positives[anchor] = self.rng.sample(
                        positives, self.curriculum.max_positives
                    )

        positive_to_negatives = {}
        for row in anchor_to_negatives_iter:
            pos = row['positive_code']
            neg = row['negative_code']
            hardness = row['hardness']

            if pos not in positive_to_negatives:
                positive_to_negatives[pos] = {}

            if hardness not in positive_to_negatives[pos]:
                positive_to_negatives[pos][hardness] = []

            positive_to_negatives[pos][hardness].append(neg)

        self._anchor_to_positives = anchor_to_positives
        self._positive_to_negatives_by_hardness = positive_to_negatives

        total_pairs = sum(len(v) for v in anchor_to_positives.values())

        logger.info(f'Filtered to {len(anchor_to_positives)} anchors, {total_pairs} positive pairs')


    # Sampling negatives according to curriculum
    def _sample_negatives(self, positive_code: str) -> List[str]:

        negatives_by_hardness = self._positive_to_negatives_by_hardness.get(positive_code, {}) #type: ignore

        k = self.curriculum.k_negatives
        target_counts = {
            bucket: int(k * pct)
            for bucket, pct in self.curriculum.bucket_percentages.items() #type: ignore
        }

        remaining = k - sum(target_counts.values())
        if remaining > 0:
            for bucket in sorted(target_counts.keys(), reverse=True):
                if remaining == 0:
                    break
                target_counts[bucket] += 1
                remaining -= 1

        sampled = []
        for bucket in sorted(self.curriculum.difficulty_buckets, reverse=True): #type: ignore
            target = target_counts.get(bucket, 0)
            if target == 0:
                continue

            available = negatives_by_hardness.get(bucket, [])

            if len(available) >= target:
                sampled.extend(self.rng.sample(available, target))

            elif len(available) > 0:
                sampled.extend(available)
                shortage = target - len(available)

                for fallback_bucket in range(bucket - 1, 0, -1):
                    if shortage == 0:
                        break
                    fallback_available = negatives_by_hardness.get(fallback_bucket, [])
                    fallback_available = [n for n in fallback_available if n not in sampled]

                    if len(fallback_available) >= shortage:
                        sampled.extend(self.rng.sample(fallback_available, shortage))
                        shortage = 0

                    elif len(fallback_available) > 0:
                        sampled.extend(fallback_available)
                        shortage -= len(fallback_available)

        if len(sampled) < k:
            all_negatives = []
            for negs in negatives_by_hardness.values():
                all_negatives.extend(negs)
            all_negatives = list(set(all_negatives) - set(sampled))

            if all_negatives:
                needed = min(k - len(sampled), len(all_negatives))
                sampled.extend(self.rng.sample(all_negatives, needed))

        return sampled[:k]


    def __iter__(self):

        self._build_indices()

        all_pairs = []
        for anchor, positives in self._anchor_to_positives.items(): #type: ignore
            for positive in positives:
                all_pairs.append((anchor, positive))

        self.rng.shuffle(all_pairs)

        for anchor_code, positive_code in all_pairs:

            negative_codes = self._sample_negatives(positive_code)

            if len(negative_codes) == 0:
                continue

            anchor_tokens = self.token_cache.get_tokens(anchor_code)
            positive_tokens = self.token_cache.get_tokens(positive_code)
            negative_tokens_list = [self.token_cache.get_tokens(neg) for neg in negative_codes]

            yield {
                'anchor': anchor_tokens,
                'positive': positive_tokens,
                'negatives': negative_tokens_list,
                'anchor_code': anchor_code,
                'positive_code': positive_code,
                'negative_codes': negative_codes
            }
