# -------------------------------------------------------------------------------------------------
# Imports and settings
# -------------------------------------------------------------------------------------------------

import logging
from typing import Optional, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from naics_embedder.model.hyperbolic import LorentzDistance

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------------------------------------
# Hyperbolic InfoNCE Loss
# -------------------------------------------------------------------------------------------------

class HyperbolicInfoNCELoss(nn.Module):
    '''
    Hyperbolic InfoNCE loss operating directly on Lorentz-model embeddings.
    
    The encoder now returns hyperbolic embeddings directly, so this loss function
    works with them without additional projection.
    '''
    
    def __init__(
        self,
        embedding_dim: int,
        temperature: float = 0.07,
        curvature: float = 1.0
    ):
        super().__init__()
        
        self.temperature = temperature
        self.curvature = curvature
        
        # Use shared Lorentz distance computation
        self.lorentz_distance = LorentzDistance(curvature)
    
    def forward(
        self,
        anchor_emb: torch.Tensor,
        positive_emb: torch.Tensor,
        negative_embs: torch.Tensor,
        batch_size: int,
        k_negatives: int,
        false_negative_mask: Optional[torch.Tensor] = None 
    ) -> torch.Tensor:
        '''
        Compute Hyperbolic InfoNCE loss.
        
        Args:
            anchor_emb: Anchor hyperbolic embeddings (batch_size, embedding_dim+1)
            positive_emb: Positive hyperbolic embeddings (batch_size, embedding_dim+1)
            negative_embs: Negative hyperbolic embeddings (batch_size * k_negatives, embedding_dim+1)
            batch_size: Batch size
            k_negatives: Number of negatives per anchor
            false_negative_mask: Optional mask for false negatives (batch_size, k_negatives)
        
        Returns:
            Loss scalar
        '''
        # Embeddings are already in hyperbolic space (Lorentz model)
        anchor_hyp = anchor_emb
        positive_hyp = positive_emb
        negative_hyp = negative_embs
        
        pos_distances = self.lorentz_distance(anchor_hyp, positive_hyp)
        
        # Compute negative distances using batched operations
        # Reshape negative_hyp from (batch_size * k_negatives, embedding_dim+1)
        # to (batch_size, k_negatives, embedding_dim+1)
        negative_hyp_reshaped = negative_hyp.view(batch_size, k_negatives, -1)
        
        # Use batched forward to compute all anchor-negative distances at once
        # anchor_hyp: (batch_size, embedding_dim+1) -> (batch_size, 1, embedding_dim+1)
        #   via broadcasting
        # negative_hyp_reshaped: (batch_size, k_negatives, embedding_dim+1)
        # Result: (batch_size, k_negatives)
        neg_distances = self.lorentz_distance.batched_forward(
            anchor_hyp, 
            negative_hyp_reshaped
        )
        
        pos_similarities = -pos_distances / self.temperature
        neg_similarities = -neg_distances / self.temperature

        if false_negative_mask is not None:
            neg_similarities = neg_similarities.masked_fill(
                false_negative_mask,
                -torch.finfo(neg_similarities.dtype).max
            )
        
        logits = torch.cat([
            pos_similarities.unsqueeze(1),
            neg_similarities
        ], dim=1)
        
        labels = torch.zeros(batch_size, dtype=torch.long, device=logits.device)
        
        loss = F.cross_entropy(logits, labels)
        
        return loss


# -------------------------------------------------------------------------------------------------
# Hierarchy Preservation Loss
# -------------------------------------------------------------------------------------------------

class HierarchyPreservationLoss(nn.Module):
    """
    Loss component that encourages embedding distances to match tree distances.
    This directly optimizes hierarchy preservation by penalizing deviations from
    ground truth tree structure.
    """
    
    def __init__(
        self,
        tree_distances: torch.Tensor,
        code_to_idx: Dict[str, int],
        weight: float = 0.1,
        min_distance: float = 0.1
    ):
        super().__init__()
        # Register as buffer so it moves with model to correct device
        self.register_buffer('tree_distances', tree_distances)
        self.code_to_idx = code_to_idx
        self.weight = weight
        self.min_distance = min_distance
        
    def forward(
        self,
        embeddings: torch.Tensor,
        codes: List[str],
        lorentz_distance_fn
    ) -> torch.Tensor:
        """
        Compute hierarchy preservation loss.
        
        Args:
            embeddings: Hyperbolic embeddings (N, D+1)
            codes: List of NAICS codes corresponding to embeddings
            lorentz_distance_fn: Function to compute Lorentz distances
        
        Returns:
            Loss scalar
        """
        # Get indices for codes that exist in ground truth
        valid_indices = []
        valid_codes = []
        for i, code in enumerate(codes):
            if code in self.code_to_idx:
                valid_indices.append(i)
                valid_codes.append(code)
        
        if len(valid_indices) < 2:
            return torch.tensor(0.0, device=embeddings.device)
        
        # Get embeddings for valid codes
        valid_embeddings = embeddings[valid_indices]
        
        # Get ground truth distance matrix indices
        gt_indices = torch.tensor([
            self.code_to_idx[code] for code in valid_codes
        ], device=embeddings.device)
        
        # Get ground truth distances for these codes
        gt_dists = self.tree_distances[gt_indices][:, gt_indices]
        
        # Compute embedding distances
        N = valid_embeddings.shape[0]
        emb_dists = torch.zeros((N, N), device=embeddings.device)
        
        for i in range(N):
            for j in range(i+1, N):
                dist = lorentz_distance_fn(
                    valid_embeddings[i:i+1],
                    valid_embeddings[j:j+1]
                )
                emb_dists[i, j] = dist
                emb_dists[j, i] = dist
        
        # Get upper triangular values (excluding diagonal)
        triu_indices = torch.triu_indices(N, N, offset=1, device=embeddings.device)
        emb_dists_flat = emb_dists[triu_indices[0], triu_indices[1]]
        gt_dists_flat = gt_dists[triu_indices[0], triu_indices[1]]
        
        # Filter out pairs with very small tree distances
        valid_mask = gt_dists_flat >= self.min_distance
        if valid_mask.sum() < 2:
            return torch.tensor(0.0, device=embeddings.device)
        
        emb_dists_filtered = emb_dists_flat[valid_mask]
        gt_dists_filtered = gt_dists_flat[valid_mask]
        
        # Normalize distances to similar scales for stable training
        emb_mean = emb_dists_filtered.mean()
        gt_mean = gt_dists_filtered.mean()
        
        emb_dists_norm = emb_dists_filtered / (emb_mean + 1e-8)
        gt_dists_norm = gt_dists_filtered / (gt_mean + 1e-8)
        
        # MSE loss between normalized distances
        mse_loss = torch.mean((emb_dists_norm - gt_dists_norm) ** 2)
        
        return self.weight * mse_loss