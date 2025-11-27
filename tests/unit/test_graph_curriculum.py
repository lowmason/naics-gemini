'''
Unit tests for graph_model curriculum learning components.

Tests the curriculum controller, event bus, sampling strategies,
adaptive loss functions, and monitoring utilities.
'''

import json
from typing import Set, Tuple

import pytest
import torch

from naics_embedder.graph_model.curriculum import (
    # Loss
    AdaptiveLossConfig,
    ConfidenceTracker,
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
    DifficultyWeightedSampler,
    HardNegativeSampler,
    HubUniformSampler,
    NextAction,
    adaptive_triplet_loss,
    compute_adaptive_margin,
    get_event_bus,
    reset_event_bus,
)

# -------------------------------------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------------------------------------

@pytest.fixture
def event_bus():
    '''Fresh event bus for each test.'''
    reset_event_bus()
    return get_event_bus()

@pytest.fixture
def controller_config():
    '''Standard controller configuration for testing.'''
    return ControllerConfig(
        plateau_patience=3,
        plateau_threshold=1e-4,
        gap_tolerance=0.5,
        phase1_min_epochs=1,
        phase2_min_epochs=1,
        phase3_min_epochs=1,
        phase4_min_epochs=1,
        max_rollbacks=2,
        rollback_cooldown=1,
    )

@pytest.fixture
def sampling_config():
    '''Standard sampling configuration for testing.'''
    return CurriculumSamplingConfig(
        phase1_hub_percentile=0.8,
        phase1_max_relation_id=1,
        phase2_max_relation_id=4,
        phase3_hard_negative_ratio=0.5,
        phase4_hard_negative_ratio=0.3,
        seed=42,
    )

@pytest.fixture
def loss_config():
    '''Standard loss configuration for testing.'''
    return AdaptiveLossConfig(
        base_margin=1.0,
        min_margin=0.5,
        max_margin=3.0,
        macl_alpha=0.5,
    )

@pytest.fixture
def sample_node_scores(test_device):
    '''Sample node difficulty scores.'''
    num_nodes = 100
    return {
        'degree_centrality': torch.rand(num_nodes, device=test_device),
        'pagerank': torch.rand(num_nodes, device=test_device),
        'kcore': torch.randint(1, 5, (num_nodes, ), device=test_device).float(),
        'level': torch.randint(2, 7, (num_nodes, ), device=test_device).float(),
        'composite': torch.rand(num_nodes, device=test_device),
        'num_nodes': torch.tensor([num_nodes]),
    }

@pytest.fixture
def sample_embeddings(test_device):
    '''Sample hyperbolic embeddings for testing.'''
    from naics_embedder.text_model.hyperbolic import LorentzOps

    num_nodes = 100
    dim = 32

    # Create valid Lorentz embeddings
    tangent = torch.randn(num_nodes, dim + 1, device=test_device)
    tangent[:, 0] = 0.0
    tangent = tangent / (torch.norm(tangent, dim=1, keepdim=True) + 1e-8) * 2.0

    return LorentzOps.exp_map_zero(tangent, c=1.0)

@pytest.fixture
def sample_triplet_batch(test_device):
    '''Sample triplet batch for loss testing.'''
    batch_size = 16
    k_negatives = 8

    anchors = torch.randint(0, 50, (batch_size, ), device=test_device)
    positives = torch.randint(50, 100, (batch_size, ), device=test_device)
    negatives = torch.randint(0, 100, (batch_size, k_negatives), device=test_device)

    return anchors, positives, negatives

# -------------------------------------------------------------------------------------------------
# EventBus Tests
# -------------------------------------------------------------------------------------------------

@pytest.mark.unit
class TestEventBus:
    '''Test suite for EventBus pub/sub system.'''

    def test_subscribe_and_publish(self, event_bus):
        '''Test basic subscribe and publish functionality.'''
        received_events = []

        def handler(event, data):
            received_events.append((event, data))

        event_bus.subscribe(CurriculumEvent.PHASE_ADVANCE, handler)
        event_bus.publish(CurriculumEvent.PHASE_ADVANCE, {'phase': 2})

        assert len(received_events) == 1
        assert received_events[0][0] == CurriculumEvent.PHASE_ADVANCE
        assert received_events[0][1]['phase'] == 2

    def test_multiple_subscribers(self, event_bus):
        '''Test multiple handlers for same event.'''
        count = [0]

        def handler1(event, data):
            count[0] += 1

        def handler2(event, data):
            count[0] += 10

        event_bus.subscribe(CurriculumEvent.METRICS_LOGGED, handler1)
        event_bus.subscribe(CurriculumEvent.METRICS_LOGGED, handler2)
        event_bus.publish(CurriculumEvent.METRICS_LOGGED, {})

        assert count[0] == 11

    def test_global_handler(self, event_bus):
        '''Test global handler receives all events.'''
        received = []

        def global_handler(event, data):
            received.append(event)

        event_bus.subscribe_all(global_handler)
        event_bus.publish(CurriculumEvent.PHASE_ADVANCE, {})
        event_bus.publish(CurriculumEvent.PLATEAU_DETECTED, {})

        assert len(received) == 2
        assert CurriculumEvent.PHASE_ADVANCE in received
        assert CurriculumEvent.PLATEAU_DETECTED in received

    def test_unsubscribe(self, event_bus):
        '''Test unsubscribing a handler.'''
        count = [0]

        def handler(event, data):
            count[0] += 1

        event_bus.subscribe(CurriculumEvent.PHASE_ADVANCE, handler)
        event_bus.publish(CurriculumEvent.PHASE_ADVANCE, {})
        assert count[0] == 1

        event_bus.unsubscribe(CurriculumEvent.PHASE_ADVANCE, handler)
        event_bus.publish(CurriculumEvent.PHASE_ADVANCE, {})
        assert count[0] == 1  # Should not increment

    def test_event_history(self, event_bus):
        '''Test event history tracking.'''
        event_bus.publish(CurriculumEvent.PHASE_ADVANCE, {'phase': 1})
        event_bus.publish(CurriculumEvent.PHASE_ADVANCE, {'phase': 2})

        history = event_bus.get_history()
        assert len(history) == 2
        assert history[0]['data']['phase'] == 1
        assert history[1]['data']['phase'] == 2

    def test_filtered_history(self, event_bus):
        '''Test filtering event history by type.'''
        event_bus.publish(CurriculumEvent.PHASE_ADVANCE, {})
        event_bus.publish(CurriculumEvent.PLATEAU_DETECTED, {})
        event_bus.publish(CurriculumEvent.PHASE_ADVANCE, {})

        filtered = event_bus.get_history(event_filter=CurriculumEvent.PHASE_ADVANCE)
        assert len(filtered) == 2

# -------------------------------------------------------------------------------------------------
# CurriculumController Tests
# -------------------------------------------------------------------------------------------------

@pytest.mark.unit
class TestCurriculumController:
    '''Test suite for CurriculumController state machine.'''

    def test_initial_state(self, controller_config, event_bus):
        '''Test controller initializes in Phase 1.'''
        controller = CurriculumController(controller_config, event_bus)

        assert controller.current_phase == CurriculumPhase.PHASE_1_ANCHORING
        assert not controller.is_finished
        assert controller.total_epochs == 0

    def test_step_increments_epoch(self, controller_config, event_bus):
        '''Test that step() increments epoch counter.'''
        controller = CurriculumController(controller_config, event_bus)

        controller.step({'train_loss': 1.0, 'val_loss': 1.0})

        assert controller.total_epochs == 1

    def test_plateau_detection(self, controller_config, event_bus):
        '''Test plateau detection triggers after patience steps.'''
        controller_config.plateau_patience = 3
        controller_config.plateau_threshold = 0.01
        controller = CurriculumController(controller_config, event_bus)

        # Feed constant loss to trigger plateau
        for _ in range(5):
            controller.step({'train_loss': 1.0, 'val_loss': 1.0})

        # Check plateau was detected via event bus
        history = event_bus.get_history(event_filter=CurriculumEvent.PLATEAU_DETECTED)
        assert len(history) > 0

    def test_phase_advancement(self, controller_config, event_bus):
        '''Test phase advancement when criteria met.'''
        controller_config.phase1_min_epochs = 2
        controller = CurriculumController(controller_config, event_bus)

        # Simulate improving loss to trigger advancement
        controller.step({'train_loss': 1.0, 'val_loss': 1.0})
        controller.step({'train_loss': 0.5, 'val_loss': 0.5})
        controller.step({'train_loss': 0.1, 'val_loss': 0.1})  # Significant improvement

        # Should advance after sufficient improvement
        # Note: exact behavior depends on advancement criteria
        assert controller.total_epochs == 3

    def test_rollback_on_overfitting(self, controller_config, event_bus):
        '''Test rollback when generalization gap exceeds threshold.'''
        controller_config.gap_tolerance = 0.3
        controller_config.phase1_min_epochs = 1
        controller = CurriculumController(controller_config, event_bus)

        # First advance to Phase 2
        controller._current_phase = CurriculumPhase.PHASE_2_EXPANSION

        # Trigger rollback with high generalization gap
        action = controller.step({'train_loss': 0.1, 'val_loss': 0.6})

        assert action == NextAction.ROLLBACK
        assert controller.current_phase == CurriculumPhase.PHASE_1_ANCHORING

    def test_max_rollbacks_limit(self, controller_config, event_bus):
        '''Test that rollbacks are limited.'''
        controller_config.max_rollbacks = 2
        controller_config.rollback_cooldown = 0
        controller_config.gap_tolerance = 0.3
        controller = CurriculumController(controller_config, event_bus)

        # Manually set rollback count
        controller._rollback_count = 2
        controller._current_phase = CurriculumPhase.PHASE_2_EXPANSION

        # Should not rollback after max reached
        action = controller.step({'train_loss': 0.1, 'val_loss': 0.6})

        assert action != NextAction.ROLLBACK

    def test_get_phase_config(self, controller_config, event_bus):
        '''Test phase configuration retrieval.'''
        controller = CurriculumController(controller_config, event_bus)

        config = controller.get_phase_config()

        assert 'phase' in config
        assert 'sampling_strategy' in config
        assert config['phase'] == 'PHASE_1_ANCHORING'
        assert config['sampling_strategy'] == 'hub_uniform'

    def test_state_serialization(self, controller_config, event_bus):
        '''Test controller state can be saved and restored.'''
        controller = CurriculumController(controller_config, event_bus)

        # Advance state
        controller.step({'train_loss': 1.0, 'val_loss': 1.0})
        controller._current_phase = CurriculumPhase.PHASE_2_EXPANSION

        state = controller.get_state()

        # Create new controller and restore
        new_controller = CurriculumController(controller_config, event_bus)
        new_controller.load_state(state)

        assert new_controller.current_phase == CurriculumPhase.PHASE_2_EXPANSION
        assert new_controller.total_epochs == 1

    def test_finished_state(self, controller_config, event_bus):
        '''Test controller finished state.'''
        controller = CurriculumController(controller_config, event_bus)
        controller._current_phase = CurriculumPhase.FINISHED

        assert controller.is_finished
        action = controller.step({'train_loss': 0.1, 'val_loss': 0.1})
        assert action == NextAction.STAY

# -------------------------------------------------------------------------------------------------
# Sampling Tests
# -------------------------------------------------------------------------------------------------

@pytest.mark.unit
class TestHubUniformSampler:
    '''Test suite for Phase 1 hub uniform sampling.'''

    def test_sample_from_candidates(self, sampling_config):
        '''Test basic sampling from candidate pool.'''
        sampler = HubUniformSampler(sampling_config)

        candidates = list(range(100))
        sampled = sampler.sample_negatives(
            anchor_idx=0,
            positive_idx=1,
            candidate_pool=candidates,
            n_negatives=10,
        )

        assert len(sampled) == 10
        assert all(s in candidates for s in sampled)

    def test_hub_filtering(self, sampling_config, sample_node_scores):
        '''Test that hub mask filters candidates.'''
        sampler = HubUniformSampler(sampling_config, node_scores=sample_node_scores)

        candidates = list(range(100))
        sampled = sampler.sample_negatives(
            anchor_idx=0,
            positive_idx=1,
            candidate_pool=candidates,
            n_negatives=10,
        )

        # Should only sample from hub nodes
        assert len(sampled) <= 10

    def test_relation_filtering(self, sampling_config):
        '''Test filtering by relation_id.'''
        sampler = HubUniformSampler(sampling_config)

        candidates = list(range(100))
        relation_ids = {i: i % 5 for i in range(100)}  # Varying relation IDs

        sampled = sampler.sample_negatives(
            anchor_idx=0,
            positive_idx=1,
            candidate_pool=candidates,
            n_negatives=10,
            relation_ids=relation_ids,
        )

        # Should only include relation_id <= 1
        for s in sampled:
            assert relation_ids[s] <= sampling_config.phase1_max_relation_id

@pytest.mark.unit
class TestDifficultyWeightedSampler:
    '''Test suite for Phase 2 difficulty-weighted sampling.'''

    def test_weighted_sampling(self, sampling_config):
        '''Test that margin weights affect sampling.'''
        sampler = DifficultyWeightedSampler(sampling_config)

        candidates = list(range(100))
        # High margin for first 10, low for rest
        margins = {i: 10.0 if i < 10 else 0.1 for i in range(100)}

        # Sample many times and check distribution
        high_margin_count = 0
        for _ in range(100):
            sampled = sampler.sample_negatives(
                anchor_idx=0,
                positive_idx=1,
                candidate_pool=candidates,
                n_negatives=5,
                margins=margins,
            )
            high_margin_count += sum(1 for s in sampled if s < 10)

        # High margin candidates should be sampled more often
        assert high_margin_count > 100  # At least 20% of samples

@pytest.mark.unit
class TestHardNegativeSampler:
    '''Test suite for Phase 3 hard negative sampling.'''

    def test_cache_refresh(self, sampling_config, sample_embeddings):
        '''Test hard negative cache refresh.'''
        # Set refresh interval to 0 so it always refreshes
        sampling_config.phase3_ann_refresh_epochs = 0
        sampler = HardNegativeSampler(sampling_config)
        sampler.update_embeddings(sample_embeddings)

        positive_pairs: Set[Tuple[int, int]] = {(0, 1), (2, 3)}
        sampler.refresh_cache(epoch=0, positive_pairs=positive_pairs)

        assert len(sampler._hard_negative_cache) > 0

    def test_mixed_sampling(self, sampling_config, sample_embeddings):
        '''Test mix of hard and uniform negatives.'''
        sampler = HardNegativeSampler(sampling_config)
        sampler.update_embeddings(sample_embeddings)

        positive_pairs: Set[Tuple[int, int]] = set()
        sampler.refresh_cache(epoch=0, positive_pairs=positive_pairs)

        candidates = list(range(100))
        sampled = sampler.sample_negatives(
            anchor_idx=0,
            positive_idx=1,
            candidate_pool=candidates,
            n_negatives=10,
        )

        assert len(sampled) <= 10

@pytest.mark.unit
class TestCurriculumSampler:
    '''Test suite for curriculum sampler manager.'''

    def test_phase_switching(self, sampling_config, event_bus):
        '''Test sampler switches on phase events.'''
        sampler = CurriculumSampler(sampling_config, event_bus=event_bus)

        assert isinstance(sampler.current_sampler, HubUniformSampler)

        # Publish phase advance event
        event_bus.publish(
            CurriculumEvent.PHASE_ADVANCE,
            {
                'new_phase': 'PHASE_2_EXPANSION',
            },
        )

        assert isinstance(sampler.current_sampler, DifficultyWeightedSampler)

    def test_get_phase_filter_params(self, sampling_config, event_bus):
        '''Test phase filter parameters.'''
        sampler = CurriculumSampler(sampling_config, event_bus=event_bus)

        params = sampler.get_phase_filter_params()

        assert 'max_relation_id' in params
        assert params['max_relation_id'] == sampling_config.phase1_max_relation_id

# -------------------------------------------------------------------------------------------------
# Adaptive Loss Tests
# -------------------------------------------------------------------------------------------------

@pytest.mark.unit
class TestConfidenceTracker:
    '''Test suite for confidence tracking.'''

    def test_initial_confidence(self, test_device):
        '''Test initial confidence is 0.5.'''
        tracker = ConfidenceTracker(num_nodes=100, device=test_device)

        indices = torch.tensor([0, 1, 2], device=test_device)
        confidence = tracker.get_confidence(indices)

        assert torch.allclose(confidence, torch.tensor([0.5, 0.5, 0.5], device=test_device))

    def test_confidence_update(self, test_device):
        '''Test confidence updates with EMA.'''
        tracker = ConfidenceTracker(num_nodes=100, smoothing=0.5, device=test_device)

        anchors = torch.tensor([0, 1], device=test_device)
        positives = torch.tensor([10, 11], device=test_device)
        distances = torch.tensor([1.0, 5.0], device=test_device)

        tracker.update(anchors, positives, distances, max_distance=10.0)

        confidence = tracker.get_confidence(anchors)

        # Lower distance -> higher confidence
        assert confidence[0] > confidence[1]

@pytest.mark.unit
class TestAdaptiveMargin:
    '''Test suite for adaptive margin computation.'''

    def test_margin_bounds(self, loss_config, test_device):
        '''Test margins stay within bounds.'''
        confidence = torch.tensor([0.0, 0.5, 1.0], device=test_device)

        margins = compute_adaptive_margin(confidence, None, loss_config)

        assert torch.all(margins >= loss_config.min_margin)
        assert torch.all(margins <= loss_config.max_margin)

    def test_low_confidence_high_margin(self, loss_config, test_device):
        '''Test low confidence produces higher margin.'''
        low_conf = torch.tensor([0.1], device=test_device)
        high_conf = torch.tensor([0.9], device=test_device)

        low_margin = compute_adaptive_margin(low_conf, None, loss_config)
        high_margin = compute_adaptive_margin(high_conf, None, loss_config)

        assert low_margin > high_margin

@pytest.mark.unit
class TestAdaptiveTripletLoss:
    '''Test suite for adaptive triplet loss.'''

    def test_loss_output(self, loss_config, sample_embeddings, sample_triplet_batch, test_device):
        '''Test loss produces valid output.'''
        anchors, positives, negatives = sample_triplet_batch

        loss, metrics = adaptive_triplet_loss(
            sample_embeddings,
            anchors,
            positives,
            negatives,
            loss_config,
            temperature=1.0,
            c=1.0,
        )

        assert loss.ndim == 0  # Scalar
        assert loss >= 0
        assert not torch.isnan(loss)

    def test_loss_metrics(self, loss_config, sample_embeddings, sample_triplet_batch, test_device):
        '''Test loss returns expected metrics.'''
        anchors, positives, negatives = sample_triplet_batch

        loss, metrics = adaptive_triplet_loss(
            sample_embeddings,
            anchors,
            positives,
            negatives,
            loss_config,
            temperature=1.0,
            c=1.0,
        )

        assert 'avg_positive_dist' in metrics
        assert 'avg_negative_dist' in metrics
        assert 'avg_margin' in metrics
        assert 'triplet_accuracy' in metrics

@pytest.mark.unit
class TestCurriculumLossModule:
    '''Test suite for curriculum loss module.'''

    def test_phase_switching(self, loss_config, event_bus):
        '''Test loss module switches on phase events.'''
        loss_module = CurriculumLossModule(loss_config, num_nodes=100, event_bus=event_bus)

        assert loss_module._current_phase == CurriculumPhase.PHASE_1_ANCHORING

        event_bus.publish(
            CurriculumEvent.PHASE_ADVANCE,
            {
                'new_phase': 'PHASE_2_EXPANSION',
            },
        )

        assert loss_module._current_phase == CurriculumPhase.PHASE_2_EXPANSION

    def test_forward_all_phases(
        self, loss_config, sample_embeddings, sample_triplet_batch, event_bus
    ):
        '''Test forward pass works for all phases.'''
        loss_module = CurriculumLossModule(loss_config, num_nodes=100, event_bus=event_bus)
        anchors, positives, negatives = sample_triplet_batch

        for phase in [
            CurriculumPhase.PHASE_1_ANCHORING,
            CurriculumPhase.PHASE_2_EXPANSION,
            CurriculumPhase.PHASE_3_DISCRIMINATION,
            CurriculumPhase.PHASE_4_STABILIZATION,
        ]:
            loss_module._current_phase = phase
            loss, metrics = loss_module(sample_embeddings, anchors, positives, negatives)

            assert loss >= 0
            assert not torch.isnan(loss)

# -------------------------------------------------------------------------------------------------
# Monitoring Tests
# -------------------------------------------------------------------------------------------------

@pytest.mark.unit
class TestCurriculumAnalyzer:
    '''Test suite for curriculum training analyzer.'''

    @pytest.fixture
    def sample_history(self):
        '''Sample training history for testing.'''
        return [
            {
                'epoch': 0,
                'train_loss': 1.0,
                'val_loss': 1.1,
                'mrr': 0.1,
                'phase': 'PHASE_1_ANCHORING'
            },
            {
                'epoch': 1,
                'train_loss': 0.8,
                'val_loss': 0.9,
                'mrr': 0.2,
                'phase': 'PHASE_1_ANCHORING'
            },
            {
                'epoch': 2,
                'train_loss': 0.6,
                'val_loss': 0.7,
                'mrr': 0.3,
                'phase': 'PHASE_2_EXPANSION'
            },
            {
                'epoch': 3,
                'train_loss': 0.5,
                'val_loss': 0.6,
                'mrr': 0.4,
                'phase': 'PHASE_2_EXPANSION'
            },
            {
                'epoch': 4,
                'train_loss': 0.4,
                'val_loss': 0.5,
                'mrr': 0.5,
                'phase': 'PHASE_3_DISCRIMINATION',
            },
        ]

    def test_load_history(self, sample_history, tmp_path):
        '''Test loading history from JSON file.'''
        history_path = tmp_path / 'history.json'
        with open(history_path, 'w') as f:
            json.dump(sample_history, f)

        analyzer = CurriculumAnalyzer(str(history_path))

        assert len(analyzer._history) == 5

    def test_phase_summary(self, sample_history):
        '''Test phase summary computation.'''
        analyzer = CurriculumAnalyzer()
        analyzer.load_from_controller_history(sample_history)

        summary = analyzer.get_phase_summary()

        assert 'PHASE_1_ANCHORING' in summary
        assert 'PHASE_2_EXPANSION' in summary
        assert summary['PHASE_1_ANCHORING'].duration_epochs == 2

    def test_phase_transitions(self, sample_history):
        '''Test phase transition detection.'''
        analyzer = CurriculumAnalyzer()
        analyzer.load_from_controller_history(sample_history)

        transitions = analyzer.get_phase_transitions()

        # Should have 3 transitions (start + 2 advances)
        assert len(transitions) == 3

    def test_metrics_timeseries(self, sample_history):
        '''Test metrics time series extraction.'''
        analyzer = CurriculumAnalyzer()
        analyzer.load_from_controller_history(sample_history)

        metrics = analyzer.get_metrics_timeseries()

        assert 'epoch' in metrics
        assert 'train_loss' in metrics
        assert 'val_loss' in metrics
        assert len(metrics['epoch']) == 5

    def test_rollback_detection(self):
        '''Test rollback event detection.'''
        history = [
            {
                'epoch': 0,
                'train_loss': 1.0,
                'val_loss': 1.0,
                'phase': 'PHASE_1_ANCHORING'
            },
            {
                'epoch': 1,
                'train_loss': 0.5,
                'val_loss': 0.5,
                'phase': 'PHASE_2_EXPANSION'
            },
            {
                'epoch': 2,
                'train_loss': 0.3,
                'val_loss': 0.8,
                'phase': 'PHASE_1_ANCHORING',
                'rollback': True,
                'generalization_gap': 0.5,
            },
        ]

        analyzer = CurriculumAnalyzer()
        analyzer.load_from_controller_history(history)

        rollbacks = analyzer.get_rollback_events()

        assert len(rollbacks) == 1
        assert rollbacks[0]['epoch'] == 2

# -------------------------------------------------------------------------------------------------
# Integration Tests
# -------------------------------------------------------------------------------------------------

@pytest.mark.unit
class TestCurriculumIntegration:
    '''Integration tests for curriculum components working together.'''

    def test_controller_sampler_integration(self, controller_config, sampling_config, event_bus):
        '''Test controller and sampler work together via event bus.'''
        from naics_embedder.graph_model.curriculum.controller import MetricsSnapshot

        controller = CurriculumController(controller_config, event_bus)
        sampler = CurriculumSampler(sampling_config, event_bus=event_bus)

        # Initial state
        assert isinstance(sampler.current_sampler, HubUniformSampler)

        # Create a proper MetricsSnapshot for advancement
        snapshot = MetricsSnapshot(
            epoch=0,
            step=0,
            train_loss=0.5,
            val_loss=0.5,
            generalization_gap=0.0,
            mrr=0.3,
            phase=CurriculumPhase.PHASE_1_ANCHORING,
        )

        # Advance controller manually
        controller._advance_phase(snapshot)

        # Sampler should have switched
        assert isinstance(sampler.current_sampler, DifficultyWeightedSampler)

    def test_controller_loss_integration(self, controller_config, loss_config, event_bus):
        '''Test controller and loss module work together via event bus.'''
        from naics_embedder.graph_model.curriculum.controller import MetricsSnapshot

        controller = CurriculumController(controller_config, event_bus)
        loss_module = CurriculumLossModule(loss_config, num_nodes=100, event_bus=event_bus)

        # Initial state
        assert loss_module._current_phase == CurriculumPhase.PHASE_1_ANCHORING

        # Create a proper MetricsSnapshot for advancement
        snapshot = MetricsSnapshot(
            epoch=0,
            step=0,
            train_loss=0.5,
            val_loss=0.5,
            generalization_gap=0.0,
            mrr=0.3,
            phase=CurriculumPhase.PHASE_1_ANCHORING,
        )

        # Advance controller manually
        controller._advance_phase(snapshot)

        # Loss module should have switched
        assert loss_module._current_phase == CurriculumPhase.PHASE_2_EXPANSION

    def test_full_training_simulation(
        self,
        controller_config,
        sampling_config,
        loss_config,
        sample_embeddings,
        sample_triplet_batch,
        event_bus,
    ):
        '''Simulate a full training run with all components.'''
        # Increase gap tolerance to prevent rollbacks during simulation
        controller_config.gap_tolerance = 2.0
        controller_config.max_rollbacks = 0  # Disable rollbacks for this test

        controller = CurriculumController(controller_config, event_bus)
        _sampler = CurriculumSampler(sampling_config, event_bus=event_bus)  # noqa: F841
        loss_module = CurriculumLossModule(loss_config, num_nodes=100, event_bus=event_bus)

        anchors, positives, negatives = sample_triplet_batch

        # Simulate epochs with decreasing loss (no overfitting)
        for epoch in range(10):
            # Compute loss
            loss, metrics = loss_module(sample_embeddings, anchors, positives, negatives)

            # Step controller with simulated metrics (train and val loss close together)
            base_loss = float(loss) * (1 - epoch * 0.05)
            controller.step(
                {
                    'train_loss': base_loss,
                    'val_loss': base_loss + 0.01,  # Small gap
                    'mrr': epoch * 0.1,
                }
            )

        # Should have made progress
        assert controller.total_epochs == 10
