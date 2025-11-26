# -------------------------------------------------------------------------------------------------
# Imports and settings
# -------------------------------------------------------------------------------------------------

import logging
from typing import Dict

import torch

from naics_embedder.text_model.hyperbolic import LorentzOps

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------------------------------------
# Validation Metrics
# -------------------------------------------------------------------------------------------------


def compute_validation_metrics(
    emb: torch.Tensor,
    anchors: torch.Tensor,
    positives: torch.Tensor,
    negatives: torch.Tensor,
    c: float = 1.0,
    top_k: int = 1,
) -> Dict[str, float]:
    '''
    Compute validation metrics for hyperbolic embeddings.

    Args:
        emb: Embeddings tensor of shape (N, embedding_dim+1)
        anchors: Anchor indices, shape (batch_size,)
        positives: Positive indices, shape (batch_size,)
        negatives: Negative indices, shape (batch_size, k_negatives)
        c: Curvature parameter (default: 1.0)
        top_k: Number of top negatives to consider for ranking (default: 1)

    Returns:
        Dictionary with validation metrics:
        - avg_positive_dist: Average distance to positive samples
        - avg_negative_dist: Average distance to negative samples
        - distance_spread: Standard deviation of distances
        - relation_accuracy: Accuracy of positive being closer than negatives
        - mean_positive_rank: Mean rank of positive sample among negatives
    '''
    batch_size = anchors.size(0)
    k_negatives = negatives.size(1)

    # Get embeddings
    anchor_emb = emb[anchors]  # (batch_size, embedding_dim+1)
    positive_emb = emb[positives]  # (batch_size, embedding_dim+1)

    # Compute positive distances
    positive_dist = LorentzOps.lorentz_distance(anchor_emb, positive_emb, c=c)
    avg_positive_dist = positive_dist.mean().item()

    # Compute negative distances
    # Reshape negatives for batched computation
    negative_emb = emb[negatives.view(-1)]  # (batch_size * k_negatives, embedding_dim+1)
    # (batch_size, k_negatives, embedding_dim+1)
    negative_emb = negative_emb.view(batch_size, k_negatives, -1)

    # Expand anchor for broadcasting
    anchor_emb.unsqueeze(1)  # (batch_size, 1, embedding_dim+1)

    # Compute distances using batched operations
    # Use LorentzOps for consistency
    negative_distances = []
    for i in range(batch_size):
        anchor = anchor_emb[i]  # (embedding_dim+1,)
        negs = negative_emb[i]  # (k_negatives, embedding_dim+1)
        dists = torch.stack(
            [
                LorentzOps.lorentz_distance(anchor.unsqueeze(0), neg.unsqueeze(0), c=c)[0]
                for neg in negs
            ]
        )
        negative_distances.append(dists)

    negative_dist = torch.stack(negative_distances)  # (batch_size, k_negatives)
    avg_negative_dist = negative_dist.mean().item()

    # Distance spread (coefficient of variation)
    all_distances = torch.cat([positive_dist, negative_dist.view(-1)])
    distance_spread = all_distances.std().item() / (all_distances.mean().item() + 1e-8)

    # Relation accuracy: positive should be closer than all negatives
    positive_vs_negatives = positive_dist.unsqueeze(1) < negative_dist  # (batch_size, k_negatives)
    relation_accuracy = positive_vs_negatives.all(dim=1).float().mean().item()

    # Mean positive rank: rank of positive distance among all distances
    # (anchor-positive + anchor-negatives)
    # Lower rank is better (rank 0 = closest)
    all_dists_per_anchor = torch.cat(
        [positive_dist.unsqueeze(1), negative_dist], dim=1
    )  # (batch_size, 1 + k_negatives)

    # Get ranks (0 = smallest distance, higher = larger distance)
    _, ranks = torch.sort(all_dists_per_anchor, dim=1)
    # Position of positive (should be 0 if closest)
    positive_ranks = ranks[:, 0].float().mean().item()

    return {
        'avg_positive_dist': avg_positive_dist,
        'avg_negative_dist': avg_negative_dist,
        'distance_spread': distance_spread,
        'relation_accuracy': relation_accuracy,
        'mean_positive_rank': positive_ranks,
    }
