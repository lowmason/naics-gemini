import json
from types import SimpleNamespace
from typing import Callable, Tuple

import pytest
import torch
from torch import nn

from naics_embedder.graph_model.hgcn import (
    CurriculumState,
    HGCNLightningModule,
    HyperbolicConvolution,
)
from naics_embedder.text_model.hyperbolic import LorentzOps, check_lorentz_manifold_validity
from naics_embedder.utils.config import GraphConfig

def _make_lorentz_embeddings(num_nodes: int, dim: int, device: torch.device) -> torch.Tensor:
    tangent = torch.randn(num_nodes, dim, device=device)
    tangent[:, 0] = 0.0  # Time component 0 in tangent space
    tangent = tangent / (torch.norm(tangent, dim=1, keepdim=True) + 1e-8) * 1.5
    return LorentzOps.exp_map_zero(tangent, c=1.0)

@pytest.fixture
def graph_inputs(tmp_path, test_device) -> Callable[..., Tuple[HGCNLightningModule, torch.Tensor]]:

    def _builder(**overrides):
        cache_dir = tmp_path / 'curriculum'
        cache_dir.mkdir()
        (cache_dir / 'difficulty_thresholds.json').write_text(
            json.dumps(
                {
                    'phase1_max_relation': 1,
                    'phase1_max_distance': 1.5,
                    'phase2_max_relation': 3,
                    'phase2_max_distance': 2.5,
                }
            )
        )

        cfg_kwargs = {
            'tangent_dim': overrides.pop('tangent_dim', 4),
            'n_hgcn_layers': overrides.pop('n_hgcn_layers', 2),
            'dropout': overrides.pop('dropout', 0.0),
            'learnable_curvature': overrides.pop('learnable_curvature', False),
            'learnable_loss_weights': overrides.pop('learnable_loss_weights', False),
            'curriculum_enabled': overrides.pop('curriculum_enabled', True),
            'curriculum_cache_dir': str(cache_dir),
            'distance_matrix_parquet': None,
            'relations_parquet': str(tmp_path / 'missing_relations.parquet'),
            'full_eval_frequency': 10,
            'num_epochs': 6,
            'hard_negative_start_epoch': overrides.pop('hard_negative_start_epoch', 3),
            'curriculum_warmup_epochs': overrides.pop('curriculum_warmup_epochs', 1),
            'edge_attention_hidden_dim': 8,
            'ndcg_k_values': [5],
            'output_dir': str(tmp_path / 'outputs'),
        }
        cfg_kwargs.update(overrides)
        cfg = GraphConfig(**cfg_kwargs)

        num_nodes = 6
        embeddings = _make_lorentz_embeddings(num_nodes, cfg.tangent_dim, test_device)
        levels = torch.randint(2, 7, (num_nodes, ), device=test_device)

        edge_index = torch.tensor(
            [
                [0, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 0],
                [1, 2, 3, 4, 5, 0, 0, 1, 2, 3, 4, 5],
            ],
            dtype=torch.long,
            device=test_device,
        )
        edge_types = torch.zeros(edge_index.size(1), dtype=torch.long, device=test_device)
        edge_weights = torch.ones(edge_index.size(1), dtype=torch.float32, device=test_device)
        edge_meta = {'edge_type_count': 1, 'sibling_type_id': None}

        lit_module = HGCNLightningModule(
            cfg, embeddings.clone(), levels, edge_index, edge_types, edge_weights, edge_meta
        )
        return lit_module, embeddings

    return _builder

@pytest.mark.unit
def test_hyperbolic_convolution_preserves_manifold(test_device):
    conv = HyperbolicConvolution(
        dim=4,
        dropout=0.0,
        learnable_curvature=False,
        edge_type_count=1,
    ).to(test_device)
    x_hyp = _make_lorentz_embeddings(4, 4, test_device)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long, device=test_device)
    edge_types = torch.zeros(edge_index.size(1), dtype=torch.long, device=test_device)
    edge_weights = torch.ones(edge_index.size(1), dtype=torch.float32, device=test_device)

    out = conv(x_hyp, edge_index, edge_types, edge_weights)
    is_valid, _, violations = check_lorentz_manifold_validity(out, curvature=1.0)

    assert out.shape == x_hyp.shape
    assert is_valid
    assert float(violations.max()) < 1e-3

@pytest.mark.unit
def test_edge_aware_attention_weights_sum_to_one(test_device):
    conv = HyperbolicConvolution(
        dim=3,
        dropout=0.0,
        learnable_curvature=False,
        edge_type_count=2,
        sibling_type_id=None,
    ).to(test_device)

    conv.edge_type_emb.weight.data.zero_()
    for layer in conv.attn_mlp:
        if isinstance(layer, nn.Linear):
            layer.weight.data.zero_()
            layer.bias.data.zero_()

    num_edges = 4
    x_i = torch.zeros(num_edges, 3, device=test_device)
    x_j = torch.ones(num_edges, 3, device=test_device)
    edge_type = torch.tensor([0, 0, 1, 1], dtype=torch.long, device=test_device)
    edge_weight = torch.ones(num_edges, dtype=torch.float32, device=test_device)
    index = torch.tensor([0, 0, 1, 1], dtype=torch.long, device=test_device)

    message = conv.message(x_i, x_j, edge_type, edge_weight, index)
    alpha = message[:, 0]  # Each feature identical, so first dim recovers attention weight

    group0 = alpha[index == 0].sum()
    group1 = alpha[index == 1].sum()
    assert torch.isclose(group0, torch.tensor(1.0, device=test_device))
    assert torch.isclose(group1, torch.tensor(1.0, device=test_device))

@pytest.mark.unit
def test_gradient_flow_through_hyperbolic_ops(test_device):
    conv = HyperbolicConvolution(
        dim=4,
        dropout=0.0,
        learnable_curvature=False,
        edge_type_count=1,
    ).to(test_device)
    x_hyp = _make_lorentz_embeddings(4, 4, test_device).requires_grad_(True)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long, device=test_device)
    edge_types = torch.zeros(edge_index.size(1), dtype=torch.long, device=test_device)
    edge_weights = torch.ones(edge_index.size(1), dtype=torch.float32, device=test_device)

    out = conv(x_hyp, edge_index, edge_types, edge_weights)
    loss = out.pow(2).sum()
    loss.backward()

    assert x_hyp.grad is not None
    assert torch.all(torch.isfinite(x_hyp.grad))

@pytest.mark.unit
def test_curriculum_phase_transitions(graph_inputs):
    module, _ = graph_inputs()

    module.curriculum_enabled = False
    module.trainer = SimpleNamespace(current_epoch=0)
    static_state = module._get_curriculum_state(batch_idx=0)
    assert static_state.name == 'static'

    module.curriculum_enabled = True
    module.trainer.current_epoch = 0
    warmup = module._get_curriculum_state(batch_idx=0)
    assert warmup.name == 'warmup'
    assert warmup.margin_scale == pytest.approx(module.cfg.curriculum_margin_warmup_scale)

    module.trainer.current_epoch = module.cfg.curriculum_warmup_epochs
    expansion = module._get_curriculum_state(batch_idx=0)
    assert expansion.name == 'expansion'
    assert expansion.use_hard_negatives is False

    module.trainer.current_epoch = module.cfg.hard_negative_start_epoch
    discrimination = module._get_curriculum_state(batch_idx=0)
    assert discrimination.name == 'discrimination'
    assert discrimination.use_hard_negatives is True

@pytest.mark.unit
def test_adaptive_margin_respects_bounds(graph_inputs):
    module, embeddings = graph_inputs(use_adaptive_margin=True)
    anchors = torch.tensor([0, 1, 2], dtype=torch.long)
    positives = torch.tensor([1, 2, 3], dtype=torch.long)
    curriculum = CurriculumState(name='discrimination', margin_scale=0.8, temperature=0.9)

    margins = module._compute_adaptive_margin(embeddings, anchors, positives, curriculum)
    assert margins.shape == (anchors.size(0), )

    assert torch.all(margins >= module.cfg.adaptive_margin_min * curriculum.margin_scale - 1e-6)
    assert torch.all(margins <= module.cfg.adaptive_margin_max * curriculum.margin_scale + 1e-6)

@pytest.mark.unit
def test_phase_specific_negative_filtering(graph_inputs):
    module, _ = graph_inputs()

    negatives = torch.tensor([[9, 8, 7], [6, 5, 4]], dtype=torch.long)
    neg_rel = torch.tensor([[0, 3, 4], [2, 2, 2]], dtype=torch.long)
    neg_dist = torch.tensor([[0.5, 2.5, 0.2], [3.0, 0.5, 0.9]], dtype=torch.float32)
    curriculum = CurriculumState(name='warmup', max_relation=1, max_distance=1.0)

    filtered = module._filter_negatives_by_curriculum(negatives, neg_rel, neg_dist, curriculum)

    assert filtered.shape == negatives.shape
    # First row retains only index 0 (others padded with last valid entry)
    assert torch.equal(filtered[0], torch.tensor([9, 9, 9]))
    # Second row has no valid entries, so it should fall back to original row
    assert torch.equal(filtered[1], negatives[1])

@pytest.mark.unit
def test_hard_negative_mining_returns_weights(graph_inputs):
    module, embeddings = graph_inputs()
    anchors = torch.tensor([0, 1], dtype=torch.long)
    negatives = torch.tensor([[2, 3, 4, 5], [3, 4, 5, 0]], dtype=torch.long)
    curriculum = CurriculumState(name='discrimination', use_hard_negatives=True)

    sampled, weights = module._mine_hard_negatives(embeddings, anchors, negatives, curriculum)

    assert sampled.shape == negatives.shape
    assert weights is not None
    assert weights.shape == negatives.shape
    assert torch.allclose(weights.sum(dim=1), torch.ones(weights.size(0)))

@pytest.mark.unit
def test_prepare_negatives_returns_weights_for_hard_phase(graph_inputs):
    module, embeddings = graph_inputs()
    anchors = torch.tensor([0], dtype=torch.long)
    positives = torch.tensor([1], dtype=torch.long)
    negatives = torch.tensor([[2, 3, 4]], dtype=torch.long)
    rel_ids = torch.tensor([[0, 1, 2]], dtype=torch.long)
    dists = torch.tensor([[0.5, 1.0, 3.0]], dtype=torch.float32)
    zeros = torch.zeros_like(dists)

    discrimination = CurriculumState(name='discrimination', use_hard_negatives=True)
    filtered_negatives, weights = module._prepare_negatives(
        embeddings, anchors, positives, negatives, rel_ids, dists, zeros, zeros, discrimination
    )

    assert filtered_negatives.shape == negatives.shape
    assert weights is not None
    assert torch.allclose(weights.sum(dim=1), torch.ones(weights.size(0)))

    warmup = CurriculumState(
        name='warmup', use_hard_negatives=False, max_relation=1, max_distance=0.6
    )
    warmup_negatives, warmup_weights = module._prepare_negatives(
        embeddings, anchors, positives, negatives, rel_ids, dists, zeros, zeros, warmup
    )

    assert warmup_weights is None
    assert torch.equal(warmup_negatives[0, :2], torch.tensor([2, 2]))

@pytest.mark.unit
def test_convolution_handles_self_loops_only(test_device):
    conv = HyperbolicConvolution(
        dim=3,
        dropout=0.0,
        learnable_curvature=False,
        edge_type_count=1,
    ).to(test_device)
    x_hyp = _make_lorentz_embeddings(2, 3, test_device)
    edge_index = torch.tensor([[0, 1], [0, 1]], dtype=torch.long, device=test_device)
    edge_types = torch.zeros(edge_index.size(1), dtype=torch.long, device=test_device)
    edge_weights = torch.ones(edge_index.size(1), dtype=torch.float32, device=test_device)

    out = conv(x_hyp, edge_index, edge_types, edge_weights)
    assert torch.isfinite(out).all()
