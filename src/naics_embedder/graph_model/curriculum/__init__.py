# -------------------------------------------------------------------------------------------------
# Curriculum Learning Module for HGCN
# -------------------------------------------------------------------------------------------------
'''
Metric-driven curriculum learning for Hyperbolic Graph Convolutional Networks.

This module implements a 4-phase dynamic curriculum:
  Phase 1 (Anchoring): Hub nodes with uniform sampling, high curvature
  Phase 2 (Expansion): Tail nodes with adaptive margins
  Phase 3 (Discrimination): Hard negative mining
  Phase 4 (Stabilization): Full graph with knowledge distillation

Distance/Difficulty Metrics:
  - distance: Tree distance (lower = closer = easier)
  - relation_id: Relation type ID (lower = closer = easier)
  - margin: Composite metric (HIGHER = closer, for weighted sampling)
      Formula: margin = 1 / (1/3 * relation_id + 2/3 * distance)
'''

from naics_embedder.graph_model.curriculum.adaptive_loss import (
    AdaptiveLossConfig,
    ConfidenceTracker,
    CurriculumLossModule,
    adaptive_triplet_loss,
    compute_adaptive_margin,
)
from naics_embedder.graph_model.curriculum.controller import (
    ControllerConfig,
    CurriculumController,
    CurriculumPhase,
    NextAction,
)
from naics_embedder.graph_model.curriculum.event_bus import (
    CurriculumEvent,
    EventBus,
    get_event_bus,
    reset_event_bus,
)
from naics_embedder.graph_model.curriculum.monitoring import (
    CurriculumAnalyzer,
    PhaseMetrics,
    generate_report,
    plot_curriculum_training,
)
from naics_embedder.graph_model.curriculum.preprocess_curriculum import (
    compute_difficulty_thresholds,
    compute_node_scores,
    compute_relation_cardinality,
    load_distance_thresholds,
    load_node_scores,
    load_relation_types,
    preprocess_curriculum_data,
)
from naics_embedder.graph_model.curriculum.sampling import (
    BlendedSampler,
    CurriculumSampler,
    CurriculumSamplingConfig,
    DifficultyWeightedSampler,
    HardNegativeSampler,
    HubUniformSampler,
    NegativeSampler,
)

__all__ = [
    # Controller
    'ControllerConfig',
    'CurriculumController',
    'CurriculumPhase',
    'NextAction',
    # Event Bus
    'CurriculumEvent',
    'EventBus',
    'get_event_bus',
    'reset_event_bus',
    # Preprocessing
    'compute_difficulty_thresholds',
    'compute_node_scores',
    'compute_relation_cardinality',
    'load_distance_thresholds',
    'load_node_scores',
    'load_relation_types',
    'preprocess_curriculum_data',
    # Sampling
    'BlendedSampler',
    'CurriculumSampler',
    'CurriculumSamplingConfig',
    'DifficultyWeightedSampler',
    'HardNegativeSampler',
    'HubUniformSampler',
    'NegativeSampler',
    # Adaptive Loss
    'AdaptiveLossConfig',
    'ConfidenceTracker',
    'CurriculumLossModule',
    'adaptive_triplet_loss',
    'compute_adaptive_margin',
    # Monitoring
    'CurriculumAnalyzer',
    'PhaseMetrics',
    'generate_report',
    'plot_curriculum_training',
]
