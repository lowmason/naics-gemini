# -------------------------------------------------------------------------------------------------
# Phase-Aware Negative Sampling for Curriculum Learning (#54, #56)
# -------------------------------------------------------------------------------------------------
'''
Curriculum-aware negative sampling strategies for each training phase.

Phase 1 (Anchoring): Hub-only uniform sampling
  - Filter to high-centrality nodes
  - Use only close relations (relation_id <= 1)
  - Uniform random negatives

Phase 2 (Expansion): Difficulty-weighted sampling
  - Include tail nodes with adaptive weighting
  - Use margin for weighted sampling (higher margin = closer = more likely)
  - Expand to medium-distance relations

Phase 3 (Discrimination): Hard negative mining
  - ANN-based search for nearest false neighbors
  - Optional generative negatives
  - All relations included

Phase 4 (Stabilization): Blended full sampling
  - Mix of cached hard negatives and uniform
  - Full dataset
'''

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
from torch import Tensor

from naics_embedder.graph_model.curriculum.controller import CurriculumPhase
from naics_embedder.graph_model.curriculum.event_bus import (
    CurriculumEvent,
    EventBus,
    get_event_bus,
)

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------------------------------
# Sampling Configuration
# -------------------------------------------------------------------------------------------------

@dataclass
class CurriculumSamplingConfig:
    '''Configuration for curriculum-aware sampling.'''

    # Node difficulty cache paths
    node_scores_path: Optional[str] = None
    difficulty_thresholds_path: Optional[str] = None

    # Phase 1: Hub-only uniform
    phase1_hub_percentile: float = 0.8  # Top 20% by centrality
    phase1_max_relation_id: int = 1  # Child only
    phase1_max_distance: float = 2.0  # Close pairs only

    # Phase 2: Difficulty-weighted
    phase2_hub_percentile: float = 0.0  # Include all nodes
    phase2_max_relation_id: int = 4  # Up to great-grandchild
    phase2_use_margin_weighting: bool = True

    # Phase 3: Hard negative mining
    phase3_hard_negative_ratio: float = 0.5  # 50% hard negatives
    phase3_ann_refresh_epochs: int = 3  # Refresh ANN index every N epochs
    phase3_use_generative: bool = False  # Use generative negatives

    # Phase 4: Blended
    phase4_hard_negative_ratio: float = 0.3  # 30% cached hard negatives
    phase4_uniform_ratio: float = 0.7  # 70% uniform

    # General
    seed: int = 42

# -------------------------------------------------------------------------------------------------
# Negative Sampler Interface
# -------------------------------------------------------------------------------------------------

class NegativeSampler:
    '''
    Base interface for negative sampling strategies.

    Samplers can be swapped at runtime via the EventBus when the
    CurriculumController advances phases.
    '''

    def __init__(
        self,
        config: CurriculumSamplingConfig,
        event_bus: Optional[EventBus] = None,
    ) -> None:
        self.config = config
        self.event_bus = event_bus or get_event_bus()
        self._rng = np.random.default_rng(config.seed)

    def sample_negatives(
        self,
        anchor_idx: int,
        positive_idx: int,
        candidate_pool: List[int],
        n_negatives: int,
        **kwargs: Any,
    ) -> List[int]:
        '''
        Sample negative indices for a given anchor-positive pair.

        Args:
            anchor_idx: Index of anchor node
            positive_idx: Index of positive node
            candidate_pool: List of candidate negative indices
            n_negatives: Number of negatives to sample
            **kwargs: Additional context (margins, embeddings, etc.)

        Returns:
            List of sampled negative indices
        '''
        raise NotImplementedError

    def update_phase(self, phase: CurriculumPhase) -> None:
        '''Update sampler state when phase changes.'''
        pass

# -------------------------------------------------------------------------------------------------
# Phase 1: Hub-Only Uniform Sampling
# -------------------------------------------------------------------------------------------------

class HubUniformSampler(NegativeSampler):
    '''
    Phase 1 sampler: Uniform sampling from hub nodes only.

    Filters candidates to high-centrality nodes and close relations.
    '''

    def __init__(
        self,
        config: CurriculumSamplingConfig,
        node_scores: Optional[Dict[str, Tensor]] = None,
        event_bus: Optional[EventBus] = None,
    ) -> None:
        super().__init__(config, event_bus)
        self._node_scores = node_scores
        self._hub_mask: Optional[Tensor] = None

        if node_scores is not None:
            self._compute_hub_mask()

    def _compute_hub_mask(self) -> None:
        '''Compute mask for hub nodes based on composite centrality.'''
        if self._node_scores is None:
            return

        composite = self._node_scores.get('composite')
        if composite is None:
            return

        threshold = torch.quantile(composite, self.config.phase1_hub_percentile)
        self._hub_mask = composite >= threshold

        n_hubs = int(self._hub_mask.sum().item())
        pct = self.config.phase1_hub_percentile
        logger.info(f'Phase 1 hub mask: {n_hubs} nodes above {pct:.0%} percentile')

    def sample_negatives(
        self,
        anchor_idx: int,
        positive_idx: int,
        candidate_pool: List[int],
        n_negatives: int,
        **kwargs: Any,
    ) -> List[int]:
        '''Uniform sampling from hub candidates only.'''
        # Filter to hubs if mask available
        if self._hub_mask is not None:
            candidate_pool = [
                c for c in candidate_pool if c < len(self._hub_mask) and self._hub_mask[c]
            ]

        # Filter by relation_id if provided
        relation_ids = kwargs.get('relation_ids', {})
        if relation_ids:
            candidate_pool = [
                c for c in candidate_pool
                if relation_ids.get(c, 99) <= self.config.phase1_max_relation_id
            ]

        # Uniform sampling
        if len(candidate_pool) == 0:
            return []

        n_to_sample = min(n_negatives, len(candidate_pool))
        indices = self._rng.choice(len(candidate_pool), size=n_to_sample, replace=False)
        return [candidate_pool[i] for i in indices]

# -------------------------------------------------------------------------------------------------
# Phase 2: Difficulty-Weighted Sampling
# -------------------------------------------------------------------------------------------------

class DifficultyWeightedSampler(NegativeSampler):
    '''
    Phase 2 sampler: Weighted sampling based on margin/difficulty.

    Uses the margin metric (higher = closer = more likely to sample)
    to focus on informative negatives while including tail nodes.
    '''

    def __init__(
        self,
        config: CurriculumSamplingConfig,
        node_scores: Optional[Dict[str, Tensor]] = None,
        event_bus: Optional[EventBus] = None,
    ) -> None:
        super().__init__(config, event_bus)
        self._node_scores = node_scores

    def sample_negatives(
        self,
        anchor_idx: int,
        positive_idx: int,
        candidate_pool: List[int],
        n_negatives: int,
        **kwargs: Any,
    ) -> List[int]:
        '''Weighted sampling based on margin scores.'''
        if len(candidate_pool) == 0:
            return []

        # Get margins for candidates (higher margin = closer = more informative)
        margins = kwargs.get('margins', {})

        if self.config.phase2_use_margin_weighting and margins:
            # Use margin as sampling weight
            weights = np.array([margins.get(c, 0.1) for c in candidate_pool])
            weights = np.clip(weights, 1e-8, None)  # Ensure positive
            weights = weights / weights.sum()  # Normalize

            n_to_sample = min(n_negatives, len(candidate_pool))
            indices = self._rng.choice(
                len(candidate_pool),
                size=n_to_sample,
                replace=False,
                p=weights,
            )
            return [candidate_pool[i] for i in indices]
        else:
            # Fall back to uniform
            n_to_sample = min(n_negatives, len(candidate_pool))
            indices = self._rng.choice(len(candidate_pool), size=n_to_sample, replace=False)
            return [candidate_pool[i] for i in indices]

# -------------------------------------------------------------------------------------------------
# Phase 3: Hard Negative Mining
# -------------------------------------------------------------------------------------------------

class HardNegativeSampler(NegativeSampler):
    '''
    Phase 3 sampler: ANN-based hard negative mining.

    Finds semantically nearest neighbors that are false links using
    approximate nearest neighbor search on current embeddings.
    '''

    def __init__(
        self,
        config: CurriculumSamplingConfig,
        event_bus: Optional[EventBus] = None,
    ) -> None:
        super().__init__(config, event_bus)
        self._embeddings: Optional[Tensor] = None
        self._hard_negative_cache: Dict[int, List[int]] = {}
        self._cache_epoch: int = -1

    def update_embeddings(self, embeddings: Tensor) -> None:
        '''Update embeddings for ANN search.'''
        self._embeddings = embeddings.detach().cpu()

    def refresh_cache(self, epoch: int, positive_pairs: Set[Tuple[int, int]]) -> None:
        '''Refresh hard negative cache using ANN search.'''
        if self._embeddings is None:
            logger.warning('Cannot refresh hard negative cache: no embeddings')
            return

        if epoch - self._cache_epoch < self.config.phase3_ann_refresh_epochs:
            return  # Cache still valid

        logger.info(f'Refreshing hard negative cache at epoch {epoch}')

        # Simple brute-force nearest neighbor (can be replaced with FAISS)
        # For each node, find k nearest neighbors that are NOT positive pairs
        n_nodes = self._embeddings.shape[0]
        k_neighbors = 50  # Find top 50 nearest

        self._hard_negative_cache.clear()

        # Compute pairwise distances in batches
        batch_size = 100
        for i in range(0, n_nodes, batch_size):
            batch_end = min(i + batch_size, n_nodes)
            batch_emb = self._embeddings[i:batch_end]

            # Compute distances to all nodes
            # Using L2 distance in tangent space approximation
            dists = torch.cdist(batch_emb, self._embeddings, p=2)

            for j, node_idx in enumerate(range(i, batch_end)):
                node_dists = dists[j]
                # Sort and get nearest (excluding self)
                sorted_indices = torch.argsort(node_dists)

                hard_negs = []
                for idx in sorted_indices[1:]:  # Skip self
                    idx_int = int(idx.item())
                    # Check it's not a positive pair
                    if (node_idx, idx_int) not in positive_pairs and (
                        idx_int,
                        node_idx,
                    ) not in positive_pairs:
                        hard_negs.append(idx_int)
                        if len(hard_negs) >= k_neighbors:
                            break

                self._hard_negative_cache[node_idx] = hard_negs

        self._cache_epoch = epoch
        logger.info(f'Hard negative cache refreshed: {len(self._hard_negative_cache)} nodes')

    def sample_negatives(
        self,
        anchor_idx: int,
        positive_idx: int,
        candidate_pool: List[int],
        n_negatives: int,
        **kwargs: Any,
    ) -> List[int]:
        '''Sample mix of hard negatives and uniform negatives.'''
        if len(candidate_pool) == 0:
            return []

        # Determine split
        n_hard = int(n_negatives * self.config.phase3_hard_negative_ratio)
        n_uniform = n_negatives - n_hard

        sampled = []

        # Sample hard negatives from cache
        if anchor_idx in self._hard_negative_cache and n_hard > 0:
            hard_candidates = [
                c for c in self._hard_negative_cache[anchor_idx]
                if c in candidate_pool and c != positive_idx
            ]
            n_hard_actual = min(n_hard, len(hard_candidates))
            if n_hard_actual > 0:
                indices = self._rng.choice(len(hard_candidates), size=n_hard_actual, replace=False)
                sampled.extend([hard_candidates[i] for i in indices])
                n_uniform += n_hard - n_hard_actual  # Add remainder to uniform

        # Sample uniform negatives
        remaining = [c for c in candidate_pool if c not in sampled and c != positive_idx]
        if remaining and n_uniform > 0:
            n_uniform_actual = min(n_uniform, len(remaining))
            indices = self._rng.choice(len(remaining), size=n_uniform_actual, replace=False)
            sampled.extend([remaining[i] for i in indices])

        return sampled

# -------------------------------------------------------------------------------------------------
# Phase 4: Blended Sampling
# -------------------------------------------------------------------------------------------------

class BlendedSampler(NegativeSampler):
    '''
    Phase 4 sampler: Blend of cached hard negatives and uniform.

    Uses cached hard negatives from Phase 3 mixed with uniform sampling
    for stabilization and knowledge distillation.
    '''

    def __init__(
        self,
        config: CurriculumSamplingConfig,
        hard_negative_cache: Optional[Dict[int, List[int]]] = None,
        event_bus: Optional[EventBus] = None,
    ) -> None:
        super().__init__(config, event_bus)
        self._hard_negative_cache = hard_negative_cache or {}

    def set_hard_negative_cache(self, cache: Dict[int, List[int]]) -> None:
        '''Set the hard negative cache from Phase 3.'''
        self._hard_negative_cache = cache

    def sample_negatives(
        self,
        anchor_idx: int,
        positive_idx: int,
        candidate_pool: List[int],
        n_negatives: int,
        **kwargs: Any,
    ) -> List[int]:
        '''Sample blend of hard and uniform negatives.'''
        if len(candidate_pool) == 0:
            return []

        n_hard = int(n_negatives * self.config.phase4_hard_negative_ratio)
        n_uniform = n_negatives - n_hard

        sampled = []

        # Sample from cached hard negatives
        if anchor_idx in self._hard_negative_cache and n_hard > 0:
            hard_candidates = [
                c for c in self._hard_negative_cache[anchor_idx]
                if c in candidate_pool and c != positive_idx
            ]
            n_hard_actual = min(n_hard, len(hard_candidates))
            if n_hard_actual > 0:
                indices = self._rng.choice(len(hard_candidates), size=n_hard_actual, replace=False)
                sampled.extend([hard_candidates[i] for i in indices])
                n_uniform += n_hard - n_hard_actual

        # Uniform sampling for remainder
        remaining = [c for c in candidate_pool if c not in sampled and c != positive_idx]
        if remaining and n_uniform > 0:
            n_uniform_actual = min(n_uniform, len(remaining))
            indices = self._rng.choice(len(remaining), size=n_uniform_actual, replace=False)
            sampled.extend([remaining[i] for i in indices])

        return sampled

# -------------------------------------------------------------------------------------------------
# Curriculum Sampler Manager
# -------------------------------------------------------------------------------------------------

class CurriculumSampler:
    '''
    Manager that switches between sampling strategies based on curriculum phase.

    Listens to EventBus for phase transitions and updates the active sampler.
    '''

    def __init__(
        self,
        config: CurriculumSamplingConfig,
        node_scores: Optional[Dict[str, Tensor]] = None,
        event_bus: Optional[EventBus] = None,
    ) -> None:
        self.config = config
        self.event_bus = event_bus or get_event_bus()
        self._node_scores = node_scores

        # Initialize samplers
        self._samplers: Dict[CurriculumPhase, NegativeSampler] = {
            CurriculumPhase.PHASE_1_ANCHORING: HubUniformSampler(config, node_scores, event_bus),
            CurriculumPhase.PHASE_2_EXPANSION:
            DifficultyWeightedSampler(config, node_scores, event_bus),
            CurriculumPhase.PHASE_3_DISCRIMINATION: HardNegativeSampler(config, event_bus),
            CurriculumPhase.PHASE_4_STABILIZATION: BlendedSampler(config, event_bus=event_bus),
        }

        self._current_phase = CurriculumPhase.PHASE_1_ANCHORING
        self._current_sampler = self._samplers[self._current_phase]

        # Subscribe to phase transitions
        self.event_bus.subscribe(CurriculumEvent.PHASE_ADVANCE, self._on_phase_advance)
        self.event_bus.subscribe(CurriculumEvent.PHASE_ROLLBACK, self._on_phase_rollback)

    @property
    def current_sampler(self) -> NegativeSampler:
        '''Get the currently active sampler.'''
        return self._current_sampler

    def _on_phase_advance(self, event: CurriculumEvent, data: Dict[str, Any]) -> None:
        '''Handle phase advancement.'''
        new_phase_name = data.get('new_phase', '')
        try:
            new_phase = CurriculumPhase[new_phase_name]
            self._switch_phase(new_phase)
        except KeyError:
            logger.warning(f'Unknown phase: {new_phase_name}')

    def _on_phase_rollback(self, event: CurriculumEvent, data: Dict[str, Any]) -> None:
        '''Handle phase rollback.'''
        new_phase_name = data.get('new_phase', '')
        try:
            new_phase = CurriculumPhase[new_phase_name]
            self._switch_phase(new_phase)
        except KeyError:
            logger.warning(f'Unknown phase: {new_phase_name}')

    def _switch_phase(self, new_phase: CurriculumPhase) -> None:
        '''Switch to a new sampling phase.'''
        if new_phase == self._current_phase:
            return

        old_phase = self._current_phase
        self._current_phase = new_phase

        if new_phase in self._samplers:
            self._current_sampler = self._samplers[new_phase]

            # Transfer hard negative cache from Phase 3 to Phase 4
            if new_phase == CurriculumPhase.PHASE_4_STABILIZATION:
                phase3_sampler = self._samplers[CurriculumPhase.PHASE_3_DISCRIMINATION]
                if isinstance(phase3_sampler, HardNegativeSampler):
                    phase4_sampler = self._samplers[CurriculumPhase.PHASE_4_STABILIZATION]
                    if isinstance(phase4_sampler, BlendedSampler):
                        phase4_sampler.set_hard_negative_cache(phase3_sampler._hard_negative_cache)

            logger.info(f'Sampler switched: {old_phase.name} -> {new_phase.name}')

    def sample_negatives(
        self,
        anchor_idx: int,
        positive_idx: int,
        candidate_pool: List[int],
        n_negatives: int,
        **kwargs: Any,
    ) -> List[int]:
        """Sample negatives using the current phase's sampler."""
        return self._current_sampler.sample_negatives(
            anchor_idx, positive_idx, candidate_pool, n_negatives, **kwargs
        )

    def update_embeddings(self, embeddings: Tensor) -> None:
        '''Update embeddings for hard negative mining.'''
        phase3_sampler = self._samplers.get(CurriculumPhase.PHASE_3_DISCRIMINATION)
        if isinstance(phase3_sampler, HardNegativeSampler):
            phase3_sampler.update_embeddings(embeddings)

    def refresh_hard_negatives(
        self,
        epoch: int,
        positive_pairs: Set[Tuple[int, int]],
    ) -> None:
        '''Refresh hard negative cache.'''
        phase3_sampler = self._samplers.get(CurriculumPhase.PHASE_3_DISCRIMINATION)
        if isinstance(phase3_sampler, HardNegativeSampler):
            phase3_sampler.refresh_cache(epoch, positive_pairs)

    def get_phase_filter_params(self) -> Dict[str, Any]:
        '''Get filtering parameters for the current phase.'''
        if self._current_phase == CurriculumPhase.PHASE_1_ANCHORING:
            return {
                'max_relation_id': self.config.phase1_max_relation_id,
                'max_distance': self.config.phase1_max_distance,
                'hub_percentile': self.config.phase1_hub_percentile,
            }
        elif self._current_phase == CurriculumPhase.PHASE_2_EXPANSION:
            return {
                'max_relation_id': self.config.phase2_max_relation_id,
                'hub_percentile': self.config.phase2_hub_percentile,
                'use_margin_weighting': self.config.phase2_use_margin_weighting,
            }
        elif self._current_phase == CurriculumPhase.PHASE_3_DISCRIMINATION:
            return {
                'max_relation_id': 99,  # All relations
                'hard_negative_ratio': self.config.phase3_hard_negative_ratio,
            }
        elif self._current_phase == CurriculumPhase.PHASE_4_STABILIZATION:
            return {
                'max_relation_id': 99,  # All relations
                'hard_negative_ratio': self.config.phase4_hard_negative_ratio,
                'uniform_ratio': self.config.phase4_uniform_ratio,
            }
        return {}
