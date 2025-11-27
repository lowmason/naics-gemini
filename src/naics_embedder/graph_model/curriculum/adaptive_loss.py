# -------------------------------------------------------------------------------------------------
# Adaptive Loss Strategies for Curriculum Learning (#55)
# -------------------------------------------------------------------------------------------------
'''
Phase-aware loss functions with adaptive margins for curriculum learning.

Phase 1: Fixed loose margins
Phase 2: Model-Aware Contrastive Learning (MACL) with adaptive margins
Phase 3: Adversarial contrastive loss
Phase 4: Knowledge distillation loss

Adaptive Margin Formula (MACL):
    margin_t = margin_base + alpha * (1 - confidence(anchor, positive))

where confidence is derived from hyperbolic distance in the previous epoch.
Harder samples (low confidence) get larger margins.
'''

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from naics_embedder.graph_model.curriculum.controller import CurriculumPhase
from naics_embedder.graph_model.curriculum.event_bus import (
    CurriculumEvent,
    EventBus,
    get_event_bus,
)
from naics_embedder.text_model.hyperbolic import LorentzOps

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------------------------------
# Loss Configuration
# -------------------------------------------------------------------------------------------------

@dataclass
class AdaptiveLossConfig:
    '''Configuration for adaptive loss strategies.'''

    # Base margins
    base_margin: float = 1.0
    min_margin: float = 0.5
    max_margin: float = 3.0

    # MACL parameters (Phase 2)
    macl_alpha: float = 0.5  # Scaling factor for confidence-based adjustment
    macl_confidence_smoothing: float = 0.1  # EMA smoothing for confidence estimates

    # Temperature
    temperature_start: float = 0.1
    temperature_end: float = 0.05

    # Adversarial loss parameters (Phase 3)
    adversarial_epsilon: float = 0.1  # Perturbation magnitude

    # Distillation parameters (Phase 4)
    distillation_temperature: float = 2.0
    distillation_alpha: float = 0.5  # Balance between hard and soft targets

# -------------------------------------------------------------------------------------------------
# Confidence Tracker
# -------------------------------------------------------------------------------------------------

class ConfidenceTracker:
    '''
    Tracks per-node confidence estimates based on hyperbolic distances.

    Confidence = 1 - normalized_distance(anchor, positive)
    High confidence = positive is close to anchor (easy sample)
    Low confidence = positive is far from anchor (hard sample)
    '''

    def __init__(
        self,
        num_nodes: int,
        smoothing: float = 0.1,
        device: Optional[torch.device] = None,
    ) -> None:
        self.num_nodes = num_nodes
        self.smoothing = smoothing
        self.device = device or torch.device('cpu')

        # Initialize confidence to 0.5 (neutral)
        self._confidence = torch.full((num_nodes, ), 0.5, dtype=torch.float32, device=self.device)
        self._update_count = torch.zeros(num_nodes, dtype=torch.long, device=self.device)

    def update(
        self,
        anchor_indices: Tensor,
        positive_indices: Tensor,
        distances: Tensor,
        max_distance: float = 10.0,
    ) -> None:
        '''
        Update confidence estimates based on observed distances.

        Args:
            anchor_indices: Anchor node indices
            positive_indices: Positive node indices
            distances: Hyperbolic distances between anchors and positives
            max_distance: Maximum distance for normalization
        '''
        # Normalize distances to [0, 1]
        normalized = torch.clamp(distances / max_distance, 0.0, 1.0)

        # Confidence = 1 - normalized_distance
        new_confidence = 1.0 - normalized

        # EMA update for each anchor
        for anchor_idx, conf in zip(anchor_indices, new_confidence):
            idx = int(anchor_idx.item())
            if idx < self.num_nodes:
                old_conf = self._confidence[idx]
                self._confidence[idx] = self.smoothing * conf + (1 - self.smoothing) * old_conf
                self._update_count[idx] += 1

    def get_confidence(self, indices: Tensor) -> Tensor:
        '''Get confidence values for given indices.'''
        return self._confidence[indices]

    def get_all_confidence(self) -> Tensor:
        '''Get all confidence values.'''
        return self._confidence.clone()

# -------------------------------------------------------------------------------------------------
# Adaptive Triplet Loss
# -------------------------------------------------------------------------------------------------

def compute_adaptive_margin(
    anchor_confidence: Tensor,
    node_difficulty: Optional[Tensor],
    config: AdaptiveLossConfig,
) -> Tensor:
    '''
    Compute adaptive margins based on confidence and node difficulty.

    MACL formula: margin = base + alpha * (1 - confidence)

    Args:
        anchor_confidence: Confidence scores for anchors [0, 1]
        node_difficulty: Optional difficulty scores (higher = easier)
        config: Loss configuration

    Returns:
        Per-sample margins
    '''
    # Base adjustment from confidence
    # Low confidence -> higher margin (harder samples need more separation)
    confidence_adjustment = config.macl_alpha * (1.0 - anchor_confidence)

    margin = config.base_margin + confidence_adjustment

    # Optional adjustment from node difficulty
    if node_difficulty is not None:
        # Lower difficulty (tail nodes) -> slightly higher margin
        difficulty_adjustment = 0.2 * (1.0 - node_difficulty)
        margin = margin + difficulty_adjustment

    # Clamp to valid range
    margin = torch.clamp(margin, config.min_margin, config.max_margin)

    return margin

def adaptive_triplet_loss(
    embeddings: Tensor,
    anchors: Tensor,
    positives: Tensor,
    negatives: Tensor,
    config: AdaptiveLossConfig,
    confidence_tracker: Optional[ConfidenceTracker] = None,
    node_difficulty: Optional[Tensor] = None,
    temperature: float = 1.0,
    c: float = 1.0,
) -> Tuple[Tensor, Dict[str, Tensor]]:
    '''
    Adaptive triplet loss with MACL-style margins.

    Args:
        embeddings: Node embeddings [N, D]
        anchors: Anchor indices [B]
        positives: Positive indices [B]
        negatives: Negative indices [B, K]
        config: Loss configuration
        confidence_tracker: Optional confidence tracker for MACL
        node_difficulty: Optional per-node difficulty scores
        temperature: Temperature for loss scaling
        c: Hyperbolic curvature

    Returns:
        Tuple of (loss, metrics_dict)
    '''
    batch_size = anchors.size(0)
    k_negatives = negatives.size(1)

    # Get embeddings
    anchor_emb = embeddings[anchors]  # [B, D]
    positive_emb = embeddings[positives]  # [B, D]
    negative_emb = embeddings[negatives.reshape(-1)].view(batch_size, k_negatives, -1)

    # Compute positive distances
    d_pos = LorentzOps.lorentz_distance(anchor_emb, positive_emb, c=c)  # [B]

    # Compute negative distances
    anchor_exp = anchor_emb.unsqueeze(1).expand(-1, k_negatives, -1)
    d_neg = LorentzOps.lorentz_distance(
        anchor_exp.reshape(-1, anchor_exp.size(-1)),
        negative_emb.reshape(-1, negative_emb.size(-1)),
        c=c,
    ).view(batch_size, k_negatives)  # [B, K]

    # Compute adaptive margins
    if confidence_tracker is not None:
        anchor_confidence = confidence_tracker.get_confidence(anchors)
    else:
        anchor_confidence = torch.full_like(d_pos, 0.5)

    if node_difficulty is not None:
        anchor_difficulty = node_difficulty[anchors]
    else:
        anchor_difficulty = None

    margins = compute_adaptive_margin(anchor_confidence, anchor_difficulty, config)

    # Update confidence tracker with new distances
    if confidence_tracker is not None:
        confidence_tracker.update(anchors, positives, d_pos.detach())

    # Triplet loss: max(0, d_pos - d_neg + margin)
    # Broadcast margin to [B, 1] for comparison with [B, K] negatives
    loss_per_neg = F.relu(d_pos.unsqueeze(1) - d_neg + margins.unsqueeze(1))

    # Apply temperature scaling
    loss_scaled = loss_per_neg / temperature

    # Mean over negatives and batch
    loss = loss_scaled.mean()

    # Compute metrics
    metrics = {
        'avg_positive_dist': d_pos.mean(),
        'avg_negative_dist': d_neg.mean(),
        'avg_margin': margins.mean(),
        'margin_std': margins.std(),
        'avg_confidence': anchor_confidence.mean(),
        'triplet_accuracy': (d_pos.unsqueeze(1) < d_neg).float().mean(),
    }

    return loss, metrics

# -------------------------------------------------------------------------------------------------
# Phase-Specific Loss Functions
# -------------------------------------------------------------------------------------------------

class CurriculumLossModule(nn.Module):
    '''
    Loss module that adapts based on curriculum phase.

    Phase 1: Fixed margin triplet loss
    Phase 2: MACL with adaptive margins
    Phase 3: Adversarial contrastive loss
    Phase 4: Distillation loss
    '''

    def __init__(
        self,
        config: AdaptiveLossConfig,
        num_nodes: int,
        event_bus: Optional[EventBus] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.event_bus = event_bus or get_event_bus()

        # Confidence tracker for MACL
        self.confidence_tracker = ConfidenceTracker(
            num_nodes, smoothing=config.macl_confidence_smoothing
        )

        # Current phase
        self._current_phase = CurriculumPhase.PHASE_1_ANCHORING
        self._current_temperature = config.temperature_start

        # Teacher model for distillation (set externally)
        self._teacher_embeddings: Optional[Tensor] = None

        # Node difficulty scores (set externally)
        self._node_difficulty: Optional[Tensor] = None

        # Subscribe to phase transitions
        self.event_bus.subscribe(CurriculumEvent.PHASE_ADVANCE, self._on_phase_advance)
        self.event_bus.subscribe(CurriculumEvent.TEMPERATURE_UPDATE, self._on_temperature_update)

    def _on_phase_advance(self, event: CurriculumEvent, data: Dict[str, Any]) -> None:
        '''Handle phase advancement.'''
        new_phase_name = data.get('new_phase', '')
        try:
            self._current_phase = CurriculumPhase[new_phase_name]
            logger.info(f'Loss module switched to {self._current_phase.name}')
        except KeyError:
            pass

    def _on_temperature_update(self, event: CurriculumEvent, data: Dict[str, Any]) -> None:
        '''Handle temperature updates.'''
        self._current_temperature = data.get('temperature', self._current_temperature)

    def set_node_difficulty(self, difficulty: Tensor) -> None:
        '''Set per-node difficulty scores.'''
        self._node_difficulty = difficulty

    def set_teacher_embeddings(self, embeddings: Tensor) -> None:
        '''Set teacher embeddings for distillation.'''
        self._teacher_embeddings = embeddings.detach()

    def forward(
        self,
        embeddings: Tensor,
        anchors: Tensor,
        positives: Tensor,
        negatives: Tensor,
        c: float = 1.0,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        '''
        Compute loss based on current curriculum phase.

        Args:
            embeddings: Node embeddings [N, D]
            anchors: Anchor indices [B]
            positives: Positive indices [B]
            negatives: Negative indices [B, K]
            c: Hyperbolic curvature

        Returns:
            Tuple of (loss, metrics_dict)
        '''
        if self._current_phase == CurriculumPhase.PHASE_1_ANCHORING:
            return self._phase1_loss(embeddings, anchors, positives, negatives, c)

        elif self._current_phase == CurriculumPhase.PHASE_2_EXPANSION:
            return self._phase2_loss(embeddings, anchors, positives, negatives, c)

        elif self._current_phase == CurriculumPhase.PHASE_3_DISCRIMINATION:
            return self._phase3_loss(embeddings, anchors, positives, negatives, c)

        elif self._current_phase == CurriculumPhase.PHASE_4_STABILIZATION:
            return self._phase4_loss(embeddings, anchors, positives, negatives, c)

        else:
            # Default to Phase 1
            return self._phase1_loss(embeddings, anchors, positives, negatives, c)

    def _phase1_loss(
        self,
        embeddings: Tensor,
        anchors: Tensor,
        positives: Tensor,
        negatives: Tensor,
        c: float,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        '''Phase 1: Fixed margin triplet loss.'''
        batch_size = anchors.size(0)
        k_negatives = negatives.size(1)

        anchor_emb = embeddings[anchors]
        positive_emb = embeddings[positives]
        negative_emb = embeddings[negatives.reshape(-1)].view(batch_size, k_negatives, -1)

        d_pos = LorentzOps.lorentz_distance(anchor_emb, positive_emb, c=c)

        anchor_exp = anchor_emb.unsqueeze(1).expand(-1, k_negatives, -1)
        d_neg = LorentzOps.lorentz_distance(
            anchor_exp.reshape(-1, anchor_exp.size(-1)),
            negative_emb.reshape(-1, negative_emb.size(-1)),
            c=c,
        ).view(batch_size, k_negatives)

        # Fixed margin
        loss = F.relu(d_pos.unsqueeze(1) - d_neg + self.config.base_margin)
        loss = (loss / self._current_temperature).mean()

        metrics = {
            'avg_positive_dist': d_pos.mean(),
            'avg_negative_dist': d_neg.mean(),
            'margin': torch.tensor(self.config.base_margin),
        }

        return loss, metrics

    def _phase2_loss(
        self,
        embeddings: Tensor,
        anchors: Tensor,
        positives: Tensor,
        negatives: Tensor,
        c: float,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        '''Phase 2: MACL with adaptive margins.'''
        return adaptive_triplet_loss(
            embeddings,
            anchors,
            positives,
            negatives,
            self.config,
            confidence_tracker=self.confidence_tracker,
            node_difficulty=self._node_difficulty,
            temperature=self._current_temperature,
            c=c,
        )

    def _phase3_loss(
        self,
        embeddings: Tensor,
        anchors: Tensor,
        positives: Tensor,
        negatives: Tensor,
        c: float,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        '''Phase 3: Adversarial contrastive loss with hard negatives.'''
        # Start with adaptive loss
        base_loss, metrics = adaptive_triplet_loss(
            embeddings,
            anchors,
            positives,
            negatives,
            self.config,
            confidence_tracker=self.confidence_tracker,
            node_difficulty=self._node_difficulty,
            temperature=self._current_temperature,
            c=c,
        )

        # Add adversarial perturbation term
        # Encourage robustness to small perturbations
        if self.training:
            # Compute gradient w.r.t. positive embeddings
            positive_emb = embeddings[positives]
            anchor_emb = embeddings[anchors]

            # Perturb positives slightly toward negatives
            with torch.no_grad():
                neg_mean = embeddings[negatives].mean(dim=1)
                direction = neg_mean - positive_emb
                direction = direction / (direction.norm(dim=-1, keepdim=True) + 1e-8)
                perturbed = positive_emb + self.config.adversarial_epsilon * direction

            # Loss on perturbed positives
            d_pos_perturbed = LorentzOps.lorentz_distance(anchor_emb, perturbed, c=c)
            adversarial_loss = F.relu(d_pos_perturbed - self.config.base_margin).mean()

            loss = base_loss + 0.1 * adversarial_loss
            metrics['adversarial_loss'] = adversarial_loss
        else:
            loss = base_loss

        return loss, metrics

    def _phase4_loss(
        self,
        embeddings: Tensor,
        anchors: Tensor,
        positives: Tensor,
        negatives: Tensor,
        c: float,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        '''Phase 4: Knowledge distillation loss.'''
        # Base triplet loss
        base_loss, metrics = adaptive_triplet_loss(
            embeddings,
            anchors,
            positives,
            negatives,
            self.config,
            confidence_tracker=self.confidence_tracker,
            node_difficulty=self._node_difficulty,
            temperature=self._current_temperature,
            c=c,
        )

        # Add distillation loss if teacher available
        if self._teacher_embeddings is not None:
            # Soft target: match distance distributions from teacher
            teacher_anchor = self._teacher_embeddings[anchors]
            teacher_positive = self._teacher_embeddings[positives]

            student_anchor = embeddings[anchors]
            student_positive = embeddings[positives]

            # Distance matching loss
            teacher_dist = LorentzOps.lorentz_distance(teacher_anchor, teacher_positive, c=c)
            student_dist = LorentzOps.lorentz_distance(student_anchor, student_positive, c=c)

            # Soft distillation: match distances with temperature
            distill_loss = F.mse_loss(
                student_dist / self.config.distillation_temperature,
                teacher_dist / self.config.distillation_temperature,
            )

            loss = (
                1 - self.config.distillation_alpha
            ) * base_loss + self.config.distillation_alpha * distill_loss
            metrics['distillation_loss'] = distill_loss
        else:
            loss = base_loss

        return loss, metrics
