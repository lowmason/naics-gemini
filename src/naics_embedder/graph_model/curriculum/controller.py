# -------------------------------------------------------------------------------------------------
# Curriculum Controller for HGCN (#52)
# -------------------------------------------------------------------------------------------------
'''
Central state machine for metric-driven curriculum learning.

Implements the 4-phase dynamic curriculum:
  Phase 1 (Anchoring): Hub nodes, uniform sampling, high curvature
  Phase 2 (Expansion): Tail nodes, adaptive margins
  Phase 3 (Discrimination): Hard negative mining
  Phase 4 (Stabilization): Full graph, knowledge distillation

The controller monitors metrics and decides when to:
  - STAY: Continue in current phase
  - ADVANCE: Move to next phase
  - ROLLBACK: Return to previous phase (overfitting detected)
'''

import json
import logging
from collections import deque
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

from naics_embedder.graph_model.curriculum.event_bus import (
    CurriculumEvent,
    EventBus,
    get_event_bus,
)

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------------------------------
# Phase and Action Enums
# -------------------------------------------------------------------------------------------------

class CurriculumPhase(Enum):
    '''Training phases for curriculum learning.'''

    PHASE_1_ANCHORING = 1  # Hub nodes, easy relations, high curvature
    PHASE_2_EXPANSION = 2  # Tail nodes, adaptive margins
    PHASE_3_DISCRIMINATION = 3  # Hard negative mining
    PHASE_4_STABILIZATION = 4  # Full graph, distillation
    FINISHED = 5  # Training complete

class NextAction(Enum):
    '''Actions the controller can take after evaluating metrics.'''

    STAY = auto()  # Continue in current phase
    ADVANCE = auto()  # Move to next phase
    ROLLBACK = auto()  # Return to previous phase

# -------------------------------------------------------------------------------------------------
# Controller Configuration
# -------------------------------------------------------------------------------------------------

@dataclass
class ControllerConfig:
    '''Configuration for the curriculum controller.'''

    # Plateau detection
    plateau_patience: int = 5  # Steps with loss change < threshold
    plateau_threshold: float = 1e-4  # Minimum loss change to not be plateau

    # Generalization gap
    gap_tolerance: float = 0.5  # Max allowed val_loss - train_loss

    # Uniformity (representation collapse detection)
    uniformity_threshold: float = -2.0  # Below this indicates collapse

    # Phase advancement thresholds
    phase1_min_epochs: int = 2  # Minimum epochs before advancing from P1
    phase2_min_epochs: int = 3
    phase3_min_epochs: int = 2
    phase4_min_epochs: int = 1

    # Metric improvement thresholds for advancement
    mrr_improvement_threshold: float = 0.05  # 5% improvement to advance

    # Rollback settings
    max_rollbacks: int = 3  # Maximum rollbacks before forcing advance
    rollback_cooldown: int = 2  # Epochs to wait after rollback

    # Curvature settings per phase
    phase1_curvature: float = 2.0  # High curvature for Phase 1
    phase2_curvature: Optional[float] = None  # Learnable in Phase 2
    phase3_curvature: Optional[float] = None  # Learnable
    phase4_curvature: Optional[float] = None  # Learnable

# -------------------------------------------------------------------------------------------------
# Metrics History
# -------------------------------------------------------------------------------------------------

@dataclass
class MetricsSnapshot:
    '''Snapshot of training metrics at a given step.'''

    epoch: int
    step: int
    train_loss: float
    val_loss: float
    generalization_gap: float
    loss_velocity: float = 0.0  # d(loss)/dt
    loss_acceleration: float = 0.0  # d²(loss)/dt²
    mrr: float = 0.0
    hits_at_10: float = 0.0
    uniformity: float = 0.0
    curvature: float = 1.0
    phase: CurriculumPhase = CurriculumPhase.PHASE_1_ANCHORING

# -------------------------------------------------------------------------------------------------
# Curriculum Controller
# -------------------------------------------------------------------------------------------------

class CurriculumController:
    '''
    Metric-driven state machine for curriculum learning.

    Monitors training metrics and decides phase transitions based on:
      - Loss velocity (plateau detection)
      - Generalization gap (overfitting detection)
      - Uniformity (collapse detection)
      - MRR/Hits@10 (performance thresholds)
    '''

    def __init__(
        self,
        config: Optional[ControllerConfig] = None,
        event_bus: Optional[EventBus] = None,
    ) -> None:
        self.config = config or ControllerConfig()
        self.event_bus = event_bus or get_event_bus()

        # State
        self._current_phase = CurriculumPhase.PHASE_1_ANCHORING
        self._phase_epoch_count: Dict[CurriculumPhase, int] = {
            phase: 0
            for phase in CurriculumPhase
        }
        self._total_epochs = 0
        self._patience_counter = 0
        self._rollback_count = 0
        self._rollback_cooldown_remaining = 0

        # Metrics tracking
        self._best_metric: float = float('inf')
        self._best_checkpoint_path: Optional[str] = None
        self._metrics_history: Deque[MetricsSnapshot] = deque(maxlen=100)
        self._recent_losses: Deque[float] = deque(maxlen=10)

        # Phase-specific state
        self._phase_start_metrics: Dict[CurriculumPhase, Optional[MetricsSnapshot]] = {
            phase: None
            for phase in CurriculumPhase
        }

        logger.info(f'CurriculumController initialized in {self._current_phase.name}')

    @property
    def current_phase(self) -> CurriculumPhase:
        '''Get the current curriculum phase.'''
        return self._current_phase

    @property
    def is_finished(self) -> bool:
        '''Check if training is complete.'''
        return self._current_phase == CurriculumPhase.FINISHED

    @property
    def total_epochs(self) -> int:
        '''Get total epochs trained across all phases.'''
        return self._total_epochs

    @property
    def phase_epochs(self) -> int:
        '''Get epochs in current phase.'''
        return self._phase_epoch_count[self._current_phase]

    def step(self, metrics: Dict[str, float]) -> NextAction:
        '''
        Process metrics and determine next action.

        Args:
            metrics: Dictionary with keys:
                - train_loss: Training loss
                - val_loss: Validation loss
                - mrr: Mean Reciprocal Rank (optional)
                - hits_at_10: Hits@10 metric (optional)
                - uniformity: Embedding uniformity (optional)
                - curvature: Current curvature (optional)

        Returns:
            NextAction indicating what the training loop should do
        '''
        if self.is_finished:
            return NextAction.STAY

        # Extract metrics
        train_loss = metrics.get('train_loss', 0.0)
        val_loss = metrics.get('val_loss', train_loss)
        mrr = metrics.get('mrr', 0.0)
        hits_at_10 = metrics.get('hits_at_10', 0.0)
        uniformity = metrics.get('uniformity', 0.0)
        curvature = metrics.get('curvature', 1.0)

        # Compute derived metrics
        gen_gap = val_loss - train_loss
        loss_velocity = self._compute_loss_velocity(val_loss)
        loss_acceleration = self._compute_loss_acceleration()

        # Create snapshot
        snapshot = MetricsSnapshot(
            epoch=self._total_epochs,
            step=len(self._metrics_history),
            train_loss=train_loss,
            val_loss=val_loss,
            generalization_gap=gen_gap,
            loss_velocity=loss_velocity,
            loss_acceleration=loss_acceleration,
            mrr=mrr,
            hits_at_10=hits_at_10,
            uniformity=uniformity,
            curvature=curvature,
            phase=self._current_phase,
        )
        self._metrics_history.append(snapshot)
        self._recent_losses.append(val_loss)

        # Record phase start if needed
        if self._phase_start_metrics[self._current_phase] is None:
            self._phase_start_metrics[self._current_phase] = snapshot

        # Update counters
        self._total_epochs += 1
        self._phase_epoch_count[self._current_phase] += 1

        # Decrement rollback cooldown
        if self._rollback_cooldown_remaining > 0:
            self._rollback_cooldown_remaining -= 1

        # Determine action
        action = self._evaluate_metrics(snapshot)

        # Execute action
        if action == NextAction.ADVANCE:
            self._advance_phase(snapshot)
        elif action == NextAction.ROLLBACK:
            self._rollback_phase(snapshot)

        # Publish metrics event
        self.event_bus.publish(
            CurriculumEvent.METRICS_LOGGED,
            {
                'snapshot': snapshot,
                'action': action.name,
                'phase': self._current_phase.name,
            },
        )

        return action

    def _compute_loss_velocity(self, current_loss: float) -> float:
        '''Compute first derivative of loss (velocity).'''
        if len(self._recent_losses) < 2:
            return 0.0
        prev_loss = self._recent_losses[-1] if self._recent_losses else current_loss
        return current_loss - prev_loss

    def _compute_loss_acceleration(self) -> float:
        '''Compute second derivative of loss (acceleration).'''
        if len(self._recent_losses) < 3:
            return 0.0
        losses = list(self._recent_losses)[-3:]
        v1 = losses[1] - losses[0]
        v2 = losses[2] - losses[1]
        return v2 - v1

    def _evaluate_metrics(self, snapshot: MetricsSnapshot) -> NextAction:
        '''
        Evaluate metrics and determine action.

        Decision tree:
          1. Check for overfitting (high gen gap) -> ROLLBACK
          2. Check for collapse (low uniformity) -> signal but STAY
          3. Check for plateau -> potentially ADVANCE
          4. Check phase-specific advancement criteria -> ADVANCE
          5. Default -> STAY
        '''
        # Check minimum epochs in phase
        min_epochs = self._get_min_epochs_for_phase()
        if self.phase_epochs < min_epochs:
            return NextAction.STAY

        # Check for overfitting (rollback condition)
        if self._should_rollback(snapshot):
            return NextAction.ROLLBACK

        # Check for plateau (advancement condition)
        if self._is_plateau():
            self._patience_counter += 1
            if self._patience_counter >= self.config.plateau_patience:
                if self._meets_advancement_criteria(snapshot):
                    return NextAction.ADVANCE
        else:
            self._patience_counter = 0

        # Check phase-specific advancement
        if self._meets_advancement_criteria(snapshot):
            return NextAction.ADVANCE

        return NextAction.STAY

    def _should_rollback(self, snapshot: MetricsSnapshot) -> bool:
        '''Check if we should rollback due to overfitting.'''
        # Don't rollback if in cooldown
        if self._rollback_cooldown_remaining > 0:
            return False

        # Don't rollback if we've hit max rollbacks
        if self._rollback_count >= self.config.max_rollbacks:
            return False

        # Don't rollback from Phase 1
        if self._current_phase == CurriculumPhase.PHASE_1_ANCHORING:
            return False

        # Check generalization gap
        if snapshot.generalization_gap > self.config.gap_tolerance:
            logger.warning(
                f'High generalization gap detected: {snapshot.generalization_gap:.4f} '
                f'> {self.config.gap_tolerance}'
            )
            self.event_bus.publish(
                CurriculumEvent.OVERFITTING_DETECTED,
                {
                    'gap': snapshot.generalization_gap,
                    'threshold': self.config.gap_tolerance,
                },
            )
            return True

        return False

    def _is_plateau(self) -> bool:
        '''Check if loss has plateaued.'''
        if len(self._recent_losses) < self.config.plateau_patience:
            return False

        recent = list(self._recent_losses)[-self.config.plateau_patience:]
        max_change = max(recent) - min(recent)

        is_plateau = max_change < self.config.plateau_threshold

        if is_plateau:
            self.event_bus.publish(
                CurriculumEvent.PLATEAU_DETECTED,
                {
                    'max_change': max_change,
                    'threshold': self.config.plateau_threshold,
                },
            )

        return is_plateau

    def _meets_advancement_criteria(self, snapshot: MetricsSnapshot) -> bool:
        '''Check if phase-specific advancement criteria are met.'''
        phase_start = self._phase_start_metrics[self._current_phase]
        if phase_start is None:
            return False

        # Phase-specific criteria
        if self._current_phase == CurriculumPhase.PHASE_1_ANCHORING:
            # Advance when loss stabilizes and basic structure is learned
            return snapshot.val_loss < phase_start.val_loss * 0.8

        elif self._current_phase == CurriculumPhase.PHASE_2_EXPANSION:
            # Advance when MRR improves significantly
            if snapshot.mrr > 0 and phase_start.mrr > 0:
                improvement = (snapshot.mrr - phase_start.mrr) / phase_start.mrr
                return improvement >= self.config.mrr_improvement_threshold
            return snapshot.val_loss < phase_start.val_loss * 0.9

        elif self._current_phase == CurriculumPhase.PHASE_3_DISCRIMINATION:
            # Advance when MRR plateaus (hard negatives exhausted)
            return self._is_plateau() and self.phase_epochs >= self.config.phase3_min_epochs

        elif self._current_phase == CurriculumPhase.PHASE_4_STABILIZATION:
            # Finish when distillation stabilizes
            return self._is_plateau() and self.phase_epochs >= self.config.phase4_min_epochs

        return False

    def _get_min_epochs_for_phase(self) -> int:
        '''Get minimum epochs required for current phase.'''
        return {
            CurriculumPhase.PHASE_1_ANCHORING: self.config.phase1_min_epochs,
            CurriculumPhase.PHASE_2_EXPANSION: self.config.phase2_min_epochs,
            CurriculumPhase.PHASE_3_DISCRIMINATION: self.config.phase3_min_epochs,
            CurriculumPhase.PHASE_4_STABILIZATION: self.config.phase4_min_epochs,
            CurriculumPhase.FINISHED: 0,
        }.get(self._current_phase, 1)

    def _advance_phase(self, snapshot: MetricsSnapshot) -> None:
        '''Advance to the next curriculum phase.'''
        old_phase = self._current_phase

        # Determine next phase
        phase_order = [
            CurriculumPhase.PHASE_1_ANCHORING,
            CurriculumPhase.PHASE_2_EXPANSION,
            CurriculumPhase.PHASE_3_DISCRIMINATION,
            CurriculumPhase.PHASE_4_STABILIZATION,
            CurriculumPhase.FINISHED,
        ]

        current_idx = phase_order.index(self._current_phase)
        if current_idx < len(phase_order) - 1:
            self._current_phase = phase_order[current_idx + 1]

        # Reset patience counter
        self._patience_counter = 0

        # Update best metric
        if snapshot.val_loss < self._best_metric:
            self._best_metric = snapshot.val_loss

        logger.info(f'Phase transition: {old_phase.name} -> {self._current_phase.name}')

        # Publish event
        self.event_bus.publish(
            CurriculumEvent.PHASE_ADVANCE,
            {
                'old_phase': old_phase.name,
                'new_phase': self._current_phase.name,
                'epoch': self._total_epochs,
                'metrics': {
                    'val_loss': snapshot.val_loss,
                    'mrr': snapshot.mrr,
                },
            },
        )

    def _rollback_phase(self, snapshot: MetricsSnapshot) -> None:
        '''Rollback to the previous curriculum phase.'''
        old_phase = self._current_phase

        # Determine previous phase
        phase_order = [
            CurriculumPhase.PHASE_1_ANCHORING,
            CurriculumPhase.PHASE_2_EXPANSION,
            CurriculumPhase.PHASE_3_DISCRIMINATION,
            CurriculumPhase.PHASE_4_STABILIZATION,
        ]

        current_idx = phase_order.index(self._current_phase)
        if current_idx > 0:
            self._current_phase = phase_order[current_idx - 1]

        # Update rollback tracking
        self._rollback_count += 1
        self._rollback_cooldown_remaining = self.config.rollback_cooldown
        self._patience_counter = 0

        logger.warning(
            f'Phase rollback: {old_phase.name} -> {self._current_phase.name} '
            f'(rollback {self._rollback_count}/{self.config.max_rollbacks})'
        )

        # Publish event
        self.event_bus.publish(
            CurriculumEvent.PHASE_ROLLBACK,
            {
                'old_phase': old_phase.name,
                'new_phase': self._current_phase.name,
                'epoch': self._total_epochs,
                'reason': 'high_generalization_gap',
                'gap': snapshot.generalization_gap,
            },
        )

        # Request checkpoint restore
        self.event_bus.publish(
            CurriculumEvent.CHECKPOINT_RESTORE,
            {
                'checkpoint_path': self._best_checkpoint_path,
            },
        )

    def save_checkpoint_path(self, path: str) -> None:
        '''Record the path to the best checkpoint for potential rollback.'''
        self._best_checkpoint_path = path
        self.event_bus.publish(
            CurriculumEvent.CHECKPOINT_SAVE,
            {
                'checkpoint_path': path,
            },
        )

    def get_phase_config(self) -> Dict[str, Any]:
        '''Get configuration parameters for the current phase.'''
        base_config = {
            'phase': self._current_phase.name,
            'phase_number': self._current_phase.value,
            'epoch': self._total_epochs,
            'phase_epoch': self.phase_epochs,
        }

        # Phase-specific parameters
        if self._current_phase == CurriculumPhase.PHASE_1_ANCHORING:
            return {
                **base_config,
                'curvature': self.config.phase1_curvature,
                'curvature_learnable': False,
                'sampling_strategy': 'hub_uniform',
                'max_relation_id': 1,  # Child only
                'margin_strategy': 'fixed',
            }

        elif self._current_phase == CurriculumPhase.PHASE_2_EXPANSION:
            return {
                **base_config,
                'curvature': self.config.phase2_curvature,
                'curvature_learnable': True,
                'sampling_strategy': 'difficulty_weighted',
                'max_relation_id': 4,  # Up to great-grandchild
                'margin_strategy': 'adaptive',
            }

        elif self._current_phase == CurriculumPhase.PHASE_3_DISCRIMINATION:
            return {
                **base_config,
                'curvature': self.config.phase3_curvature,
                'curvature_learnable': True,
                'sampling_strategy': 'hard_negative',
                'max_relation_id': 99,  # All relations
                'margin_strategy': 'adversarial',
            }

        elif self._current_phase == CurriculumPhase.PHASE_4_STABILIZATION:
            return {
                **base_config,
                'curvature': self.config.phase4_curvature,
                'curvature_learnable': True,
                'sampling_strategy': 'full_blended',
                'max_relation_id': 99,  # All relations
                'margin_strategy': 'distillation',
            }

        return base_config

    def get_history(self) -> List[Dict[str, Any]]:
        '''Get metrics history as list of dicts.'''
        return [
            {
                'epoch': s.epoch,
                'step': s.step,
                'train_loss': s.train_loss,
                'val_loss': s.val_loss,
                'generalization_gap': s.generalization_gap,
                'loss_velocity': s.loss_velocity,
                'mrr': s.mrr,
                'phase': s.phase.name,
            } for s in self._metrics_history
        ]

    def save_history(self, path: str) -> None:
        '''Save metrics history to JSON file.'''
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(self.get_history(), f, indent=2)

        logger.info(f'Saved curriculum history to {path}')

    def load_state(self, state: Dict[str, Any]) -> None:
        '''Load controller state from dict (for checkpoint restoration).'''
        self._current_phase = CurriculumPhase[state['current_phase']]
        self._total_epochs = state['total_epochs']
        self._patience_counter = state['patience_counter']
        self._rollback_count = state['rollback_count']
        self._best_metric = state['best_metric']

    def get_state(self) -> Dict[str, Any]:
        '''Get controller state as dict (for checkpointing).'''
        return {
            'current_phase': self._current_phase.name,
            'total_epochs': self._total_epochs,
            'patience_counter': self._patience_counter,
            'rollback_count': self._rollback_count,
            'best_metric': self._best_metric,
            'phase_epoch_counts': {
                p.name: c
                for p, c in self._phase_epoch_count.items()
            },
        }
