'''
Graph model module for NAICS embeddings.

This module provides:
  - HGCN: Hyperbolic Graph Convolutional Network for refining embeddings
  - Curriculum learning: 4-phase metric-driven training
  - Evaluation metrics for hyperbolic embeddings
  - Data loading utilities for graph training

Curriculum Learning Phases:
  Phase 1 (Anchoring): Hub nodes, uniform sampling, high curvature
  Phase 2 (Expansion): Tail nodes, adaptive margins (MACL)
  Phase 3 (Discrimination): Hard negative mining
  Phase 4 (Stabilization): Full graph, knowledge distillation
'''

from naics_embedder.graph_model.curriculum import (
    # Loss
    AdaptiveLossConfig,
    # Controller
    ControllerConfig,
    # Monitoring
    CurriculumAnalyzer,
    CurriculumController,
    # Event Bus
    CurriculumEvent,
    CurriculumLossModule,
    CurriculumPhase,
    # Sampling
    CurriculumSampler,
    CurriculumSamplingConfig,
    EventBus,
    NextAction,
    generate_report,
    get_event_bus,
    # Preprocessing
    preprocess_curriculum_data,
)
from naics_embedder.graph_model.evaluation import compute_validation_metrics
from naics_embedder.graph_model.hgcn import (
    HGCN,
    HGCNLightningModule,
    HyperbolicConvolution,
    load_edge_index,
    load_embeddings,
)
from naics_embedder.graph_model.hgcn import (
    main as train_hgcn,
)

__all__ = [
    # HGCN
    'HGCN',
    'HGCNLightningModule',
    'HyperbolicConvolution',
    'load_edge_index',
    'load_embeddings',
    'train_hgcn',
    # Evaluation
    'compute_validation_metrics',
    # Curriculum Controller
    'ControllerConfig',
    'CurriculumController',
    'CurriculumPhase',
    'NextAction',
    # Event Bus
    'CurriculumEvent',
    'EventBus',
    'get_event_bus',
    # Preprocessing
    'preprocess_curriculum_data',
    # Sampling
    'CurriculumSampler',
    'CurriculumSamplingConfig',
    # Loss
    'AdaptiveLossConfig',
    'CurriculumLossModule',
    # Monitoring
    'CurriculumAnalyzer',
    'generate_report',
]
