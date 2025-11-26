'''
Unit tests for router-guided negative mining utilities.
'''

import torch

from naics_embedder.text_model.hard_negative_mining import RouterGuidedNegativeMiner

def test_router_miner_kl_prefers_matching_distribution():
    '''KL-divergence metric should favor negatives with similar gate probs.'''

    miner = RouterGuidedNegativeMiner(metric='kl_divergence')

    anchor_gate_probs = torch.tensor([[0.7, 0.3]])
    negative_gate_probs = torch.tensor(
        [[
            [0.7, 0.3],  # Matching distribution
            [0.2, 0.8],  # Divergent distribution
        ]]
    )

    candidate_negatives = torch.tensor([[
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ]])

    router_negatives, scores = miner.mine_router_hard_negatives(
        anchor_gate_probs=anchor_gate_probs,
        negative_gate_probs=negative_gate_probs,
        candidate_negatives=candidate_negatives,
        k=1,
        return_scores=True,
    )

    assert router_negatives.shape == (1, 1, 3)
    assert scores is not None
    assert scores.shape == (1, 1)
    # First negative has the matching distribution, so it should be selected
    assert torch.allclose(router_negatives[0, 0], candidate_negatives[0, 0])
    # Confusion score for matching distribution should be higher
    full_scores = miner.compute_confusion_scores(anchor_gate_probs, negative_gate_probs)
    assert full_scores[0, 0] > full_scores[0, 1]

def test_router_miner_cosine_prefers_high_similarity():
    '''Cosine metric should select the highest-similarity gate distribution.'''

    miner = RouterGuidedNegativeMiner(metric='cosine_similarity')

    anchor_gate_probs = torch.tensor([[0.2, 0.2, 0.6]])
    negative_gate_probs = torch.tensor(
        [
            [
                [0.2, 0.2, 0.6],  # Identical
                [0.6, 0.2, 0.2],  # Less similar
                [0.1, 0.8, 0.1],  # Least similar
            ]
        ]
    )

    candidate_negatives = torch.arange(3 * 4, dtype=torch.float32).view(1, 3, 4)

    router_negatives, scores = miner.mine_router_hard_negatives(
        anchor_gate_probs=anchor_gate_probs,
        negative_gate_probs=negative_gate_probs,
        candidate_negatives=candidate_negatives,
        k=1,
        return_scores=True,
    )

    assert router_negatives.shape == (1, 1, 4)
    assert scores is not None
    # First candidate matches anchor distribution, should be chosen
    assert torch.allclose(router_negatives[0, 0], candidate_negatives[0, 0])
    # Score ordering should follow similarity
    full_scores = miner.compute_confusion_scores(anchor_gate_probs, negative_gate_probs)
    assert full_scores[0, 0] >= full_scores[0, 1]
    assert full_scores[0, 1] >= full_scores[0, 2]

def test_confusion_scores_shape_consistency():
    '''Confusion scores should align with batch and candidate dimensions.'''

    miner = RouterGuidedNegativeMiner(metric='kl_divergence')

    batch_size = 4
    num_experts = 3
    num_negatives = 5
    anchor_gate_probs = torch.softmax(torch.randn(batch_size, num_experts), dim=1)
    negative_gate_probs = torch.softmax(torch.randn(batch_size, num_negatives, num_experts), dim=2)

    scores = miner.compute_confusion_scores(anchor_gate_probs, negative_gate_probs)

    assert scores.shape == (batch_size, num_negatives)
    # With KL-based metric, higher confusion corresponds to more similar distributions
    best_indices = scores.argmax(dim=1)
    assert torch.all(best_indices < num_negatives)
