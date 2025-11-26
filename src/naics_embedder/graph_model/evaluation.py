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
    *,
    as_tensors: bool = False,
) -> Dict[str, float]:
    '''Compute validation metrics for hyperbolic embeddings.

    Args:
        emb: Embeddings tensor of shape ``(N, embedding_dim+1)``.
        anchors: Anchor indices, shape ``(batch_size,)``.
        positives: Positive indices, shape ``(batch_size,)``.
        negatives: Negative indices, shape ``(batch_size, k_negatives)``.
        c: Curvature parameter (default: 1.0).
        top_k: Number of top negatives to consider for auxiliary accuracy.
        as_tensors: Return torch scalars instead of Python floats (for Lightning logging).

    Returns:
        Mapping of metric names to values (either tensors or floats).
    '''
    batch_size, k_negatives = negatives.shape
    effective_top_k = max(1, min(top_k, k_negatives))

    anchor_emb = emb[anchors]
    positive_emb = emb[positives]

    positive_dist = LorentzOps.lorentz_distance(anchor_emb, positive_emb, c=c)

    negative_emb = emb[negatives.reshape(-1)].view(batch_size, k_negatives, -1)
    anchor_expanded = anchor_emb.unsqueeze(1).expand(-1, k_negatives, -1)
    negative_dist = LorentzOps.lorentz_distance(
        anchor_expanded.reshape(-1, anchor_expanded.size(-1)),
        negative_emb.reshape(-1, negative_emb.size(-1)),
        c=c,
    ).view(batch_size, k_negatives)

    avg_positive_dist = positive_dist.mean()
    avg_negative_dist = negative_dist.mean()

    all_distances = torch.cat([positive_dist, negative_dist.reshape(-1)], dim=0)
    distance_spread = torch.div(all_distances.std(), all_distances.mean().clamp_min(1e-8))

    relation_accuracy = (positive_dist.unsqueeze(1) < negative_dist).all(dim=1).float().mean()

    closest_negatives = torch.topk(negative_dist, k=effective_top_k, dim=1, largest=False).values
    top_k_relation_accuracy = (
        positive_dist.unsqueeze(1) < closest_negatives
    ).all(dim=1).float().mean()

    all_dists_per_anchor = torch.cat([positive_dist.unsqueeze(1), negative_dist], dim=1)
    order = torch.argsort(all_dists_per_anchor, dim=1)
    positive_rank_tensor = torch.argmax((order == 0).int(), dim=1)
    mean_positive_rank = positive_rank_tensor.float().mean()

    metrics = {
        'avg_positive_dist': avg_positive_dist,
        'avg_negative_dist': avg_negative_dist,
        'distance_spread': distance_spread,
        'relation_accuracy': relation_accuracy,
        'top_k_relation_accuracy': top_k_relation_accuracy,
        'mean_positive_rank': mean_positive_rank,
    }

    if as_tensors:
        return metrics

    return {k: float(v.detach().cpu()) for k, v in metrics.items()}
