# -------------------------------------------------------------------------------------------------
# Phase 2: Lorentzian Hard Negative Mining (HNM)
# -------------------------------------------------------------------------------------------------

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn

from naics_embedder.text_model.hyperbolic import LorentzDistance

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------------------------------------
# Lorentzian Hard Negative Mining
# -------------------------------------------------------------------------------------------------

class LorentzianHardNegativeMiner(nn.Module):
    '''
    Phase 2 Hard Negative Mining: Select negatives that are geometrically close
    in the learned hyperbolic space using Lorentzian distance.
    
    For each anchor, selects the top-k negatives with the smallest Lorentzian distance.
    '''
    
    def __init__(
        self,
        curvature: float = 1.0,
        safety_epsilon: float = 1e-5
    ):
        '''
        Initialize hard negative miner.
        
        Args:
            curvature: Hyperbolic curvature parameter c
            safety_epsilon: Small epsilon for safety checks to prevent NaN
        '''
        super().__init__()
        self.curvature = curvature
        self.safety_epsilon = safety_epsilon
        
        # Use shared Lorentz distance computation
        self.lorentz_distance = LorentzDistance(curvature)
    
    def compute_lorentz_norm(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Compute Lorentz norm: ||x||_L = sqrt(⟨x, x⟩_L)
        
        For a point on the hyperboloid: ⟨x, x⟩_L = -1/c
        The norm is the hyperbolic radius: sqrt(x0^2 - 1/c)
        
        Args:
            x: Hyperbolic embeddings (batch_size, embedding_dim+1)
        
        Returns:
            Lorentz norms (batch_size,)
        '''
        # Lorentz inner product with itself
        time_coord = x[:, 0]  # x₀
        spatial_coords = x[:, 1:]  # x₁...xₙ
        
        spatial_norm_sq = torch.sum(spatial_coords ** 2, dim=1)
        time_norm_sq = time_coord ** 2
        
        # For valid hyperboloid points: spatial_norm_sq - time_norm_sq = -1/c
        # Lorentz norm (hyperbolic radius) = sqrt(time_norm_sq - spatial_norm_sq) = sqrt(x0^2 - ||x_spatial||^2)
        # But we can also use: ||x||_L = sqrt(⟨x, x⟩_L + 2/c) = sqrt(-1/c + 2/c) = sqrt(1/c)
        # Actually, the hyperbolic radius is: r = sqrt(x0^2 - 1/c)
        c = self.curvature
        lorentz_norm_sq = time_norm_sq - spatial_norm_sq  # Should be 1/c for valid points
        lorentz_norm = torch.sqrt(torch.clamp(lorentz_norm_sq, min=1e-8))
        
        return lorentz_norm
    
    def check_lorentz_inner_product_safety(
        self,
        u: torch.Tensor,
        v: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        Safety check: Ensure ⟨u, v⟩_L < -1 to prevent NaN in gradients.
        
        Args:
            u: First point on hyperboloid (batch_size, embedding_dim+1)
            v: Second point on hyperboloid (batch_size, k, embedding_dim+1) or (batch_size, embedding_dim+1)
        
        Returns:
            Tuple of (safe_dot_product, is_valid)
            - safe_dot_product: Clamped Lorentz inner product
            - is_valid: Boolean tensor indicating if all pairs are valid
        '''
        # Compute Lorentz inner product
        if v.dim() == 3:
            # Batched case: u (batch_size, D+1), v (batch_size, k, D+1)
            uv = u.unsqueeze(1) * v  # (batch_size, k, D+1)
            dot_product = torch.sum(uv[:, :, 1:], dim=2) - uv[:, :, 0]  # (batch_size, k)
        else:
            # Pairwise case: u (batch_size, D+1), v (batch_size, D+1)
            uv = u * v  # (batch_size, D+1)
            dot_product = torch.sum(uv[:, 1:], dim=1) - uv[:, 0]  # (batch_size,)
        
        # Safety check: ⟨u, v⟩_L must be < -1 for valid arccosh
        # Clamp to ensure: dot_product <= -1 - epsilon
        safe_dot_product = torch.clamp(
            dot_product,
            max=-1.0 - self.safety_epsilon
        )
        
        # Check validity: all pairs should satisfy constraint
        is_valid = torch.all(dot_product < -1.0 - self.safety_epsilon)
        
        return safe_dot_product, is_valid
    
    def mine_hard_negatives(
        self,
        anchor_emb: torch.Tensor,
        candidate_negatives: torch.Tensor,
        k: int,
        return_distances: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        '''
        Mine top-k hardest negatives for each anchor based on Lorentzian distance.
        
        Args:
            anchor_emb: Anchor embeddings (batch_size, embedding_dim+1)
            candidate_negatives: Candidate negative embeddings 
                (batch_size, num_candidates, embedding_dim+1) or
                (batch_size * num_candidates, embedding_dim+1)
            k: Number of hard negatives to select (top-k)
            return_distances: If True, also return the distances
        
        Returns:
            Tuple of:
            - hard_negatives: Selected hard negatives (batch_size, k, embedding_dim+1)
            - distances: Optional distances (batch_size, k) if return_distances=True
        '''
        batch_size = anchor_emb.shape[0]
        
        # Reshape candidate_negatives if needed
        if candidate_negatives.dim() == 2:
            # Flattened: (batch_size * num_candidates, embedding_dim+1)
            num_candidates = candidate_negatives.shape[0] // batch_size
            candidate_negatives = candidate_negatives.view(
                batch_size, num_candidates, -1
            )
        elif candidate_negatives.dim() == 3:
            # Already batched: (batch_size, num_candidates, embedding_dim+1)
            num_candidates = candidate_negatives.shape[1]
        else:
            raise ValueError(
                f"Invalid candidate_negatives shape: {candidate_negatives.shape}"
            )
        
        # Safety check: ensure Lorentz inner products are valid
        safe_dot, is_valid = self.check_lorentz_inner_product_safety(
            anchor_emb,
            candidate_negatives
        )
        
        if not is_valid:
            logger.warning(
                f"Some anchor-negative pairs violate Lorentz constraint "
                f"(⟨u, v⟩_L < -1). Clamping applied."
            )
        
        # Compute Lorentzian distances for all anchor-candidate pairs
        # anchor_emb: (batch_size, embedding_dim+1)
        # candidate_negatives: (batch_size, num_candidates, embedding_dim+1)
        distances = self.lorentz_distance.batched_forward(
            anchor_emb,
            candidate_negatives
        )  # (batch_size, num_candidates)
        
        # Select top-k hardest negatives (smallest distances = hardest)
        # Use topk to get indices and values
        k_actual = min(k, num_candidates)
        topk_distances, topk_indices = torch.topk(
            distances,
            k=k_actual,
            dim=1,
            largest=False  # Smallest distances = hardest negatives
        )
        
        # Gather selected negatives
        # Create index tensor for gathering
        batch_indices = torch.arange(
            batch_size,
            device=anchor_emb.device
        ).unsqueeze(1).expand(-1, k_actual)  # (batch_size, k)
        
        hard_negatives = candidate_negatives[batch_indices, topk_indices]
        
        if return_distances:
            return hard_negatives, topk_distances
        else:
            return hard_negatives, None


# -------------------------------------------------------------------------------------------------
# Router-Guided Negative Mining
# -------------------------------------------------------------------------------------------------

class RouterGuidedNegativeMiner(nn.Module):
    '''
    Router-Guided Negative Mining: Select negatives that confuse the Gating Network.
    
    Prevents "Expert Collapse" where a single expert handles all easy negatives by
    mining negatives where the router assigns high probability to the same experts as the anchor.
    
    Uses KL-Divergence or Cosine Similarity to measure confusion between gate distributions.
    '''
    
    def __init__(
        self,
        metric: str = 'kl_divergence',
        temperature: float = 1.0
    ):
        '''
        Initialize router-guided negative miner.
        
        Args:
            metric: Confusion metric to use ('kl_divergence' or 'cosine_similarity')
            temperature: Temperature for KL-divergence computation (higher = more uniform)
        '''
        super().__init__()
        self.metric = metric
        self.temperature = temperature
        
        if metric not in ['kl_divergence', 'cosine_similarity']:
            raise ValueError(f"metric must be 'kl_divergence' or 'cosine_similarity', got {metric}")
    
    def compute_kl_divergence(
        self,
        anchor_gate_probs: torch.Tensor,
        negative_gate_probs: torch.Tensor
    ) -> torch.Tensor:
        '''
        Compute KL-divergence between anchor and negative gate distributions.
        
        KL(P_anchor || P_negative) = sum(P_anchor * log(P_anchor / P_negative))
        
        Lower KL-divergence means similar distributions (router assigns similar expert probabilities).
        We want negatives with similar gate distributions (low KL-divergence) to confuse the router,
        as they make the router think the negative is similar to the anchor.
        
        Args:
            anchor_gate_probs: Anchor gate probabilities (batch_size, num_experts)
            negative_gate_probs: Negative gate probabilities (batch_size, k_negatives, num_experts)
        
        Returns:
            KL-divergence scores (batch_size, k_negatives) - lower = more confusion
        '''
        # Add small epsilon for numerical stability
        eps = 1e-8
        
        # Normalize to ensure valid probability distributions
        anchor_probs = anchor_gate_probs + eps
        anchor_probs = anchor_probs / anchor_probs.sum(dim=1, keepdim=True)
        
        negative_probs = negative_gate_probs + eps
        negative_probs = negative_probs / negative_probs.sum(dim=2, keepdim=True)
        
        # Expand anchor_probs for broadcasting: (batch_size, 1, num_experts)
        anchor_probs_expanded = anchor_probs.unsqueeze(1)  # (batch_size, 1, num_experts)
        
        # Compute KL-divergence: sum(P_anchor * log(P_anchor / P_negative))
        log_ratio = torch.log(anchor_probs_expanded + eps) - torch.log(negative_probs + eps)
        kl_div = (anchor_probs_expanded * log_ratio).sum(dim=2)  # (batch_size, k_negatives)
        
        return kl_div
    
    def compute_cosine_similarity(
        self,
        anchor_gate_probs: torch.Tensor,
        negative_gate_probs: torch.Tensor
    ) -> torch.Tensor:
        '''
        Compute cosine similarity between anchor and negative gate distributions.
        
        Higher cosine similarity means more confusion (similar distributions).
        
        Args:
            anchor_gate_probs: Anchor gate probabilities (batch_size, num_experts)
            negative_gate_probs: Negative gate probabilities (batch_size, k_negatives, num_experts)
        
        Returns:
            Cosine similarity scores (batch_size, k_negatives)
        '''
        # Normalize to unit vectors
        anchor_norm = torch.norm(anchor_gate_probs, dim=1, keepdim=True)  # (batch_size, 1)
        anchor_normalized = anchor_gate_probs / (anchor_norm + 1e-8)
        
        negative_norm = torch.norm(negative_gate_probs, dim=2, keepdim=True)  # (batch_size, k_negatives, 1)
        negative_normalized = negative_gate_probs / (negative_norm + 1e-8)
        
        # Expand anchor for broadcasting: (batch_size, 1, num_experts)
        anchor_expanded = anchor_normalized.unsqueeze(1)
        
        # Compute cosine similarity: dot product of normalized vectors
        cosine_sim = (anchor_expanded * negative_normalized).sum(dim=2)  # (batch_size, k_negatives)
        
        return cosine_sim
    
    def compute_confusion_scores(
        self,
        anchor_gate_probs: torch.Tensor,
        negative_gate_probs: torch.Tensor
    ) -> torch.Tensor:
        '''
        Compute confusion scores between anchor and negative gate distributions.
        
        Args:
            anchor_gate_probs: Anchor gate probabilities (batch_size, num_experts)
            negative_gate_probs: Negative gate probabilities (batch_size, k_negatives, num_experts)
        
        Returns:
            Confusion scores (batch_size, k_negatives)
            - For KL-divergence: lower scores = more confusion (similar distributions)
            - For cosine similarity: higher scores = more confusion (similar distributions)
        '''
        if self.metric == 'kl_divergence':
            scores = self.compute_kl_divergence(anchor_gate_probs, negative_gate_probs)
            # Lower KL-divergence = more confusion, so we negate for consistency
            # (we want to select negatives with high confusion = low KL-divergence)
            return -scores
        elif self.metric == 'cosine_similarity':
            scores = self.compute_cosine_similarity(anchor_gate_probs, negative_gate_probs)
            # Higher cosine similarity = more confusion
            return scores
        else:
            raise ValueError(f"Unknown metric: {self.metric}")
    
    def mine_router_hard_negatives(
        self,
        anchor_gate_probs: torch.Tensor,
        negative_gate_probs: torch.Tensor,
        candidate_negatives: torch.Tensor,
        k: int,
        return_scores: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        '''
        Mine top-k router-hard negatives that confuse the gating network.
        
        Args:
            anchor_gate_probs: Anchor gate probabilities (batch_size, num_experts)
            negative_gate_probs: Negative gate probabilities (batch_size, k_negatives, num_experts)
            candidate_negatives: Candidate negative embeddings (batch_size, k_negatives, embedding_dim+1)
            k: Number of router-hard negatives to select
            return_scores: If True, also return confusion scores
        
        Returns:
            Tuple of:
            - router_hard_negatives: Selected router-hard negatives (batch_size, k, embedding_dim+1)
            - scores: Optional confusion scores (batch_size, k) if return_scores=True
        '''
        batch_size = anchor_gate_probs.shape[0]
        num_candidates = negative_gate_probs.shape[1]
        
        # Compute confusion scores
        confusion_scores = self.compute_confusion_scores(
            anchor_gate_probs,
            negative_gate_probs
        )  # (batch_size, k_negatives)
        
        # Select top-k negatives with highest confusion (most similar gate distributions)
        k_actual = min(k, num_candidates)
        topk_scores, topk_indices = torch.topk(
            confusion_scores,
            k=k_actual,
            dim=1,
            largest=True  # Higher confusion = better
        )
        
        # Gather selected negatives
        batch_indices = torch.arange(
            batch_size,
            device=anchor_gate_probs.device
        ).unsqueeze(1).expand(-1, k_actual)  # (batch_size, k)
        
        router_hard_negatives = candidate_negatives[batch_indices, topk_indices]
        
        if return_scores:
            return router_hard_negatives, topk_scores
        else:
            return router_hard_negatives, None


# -------------------------------------------------------------------------------------------------
# Norm-Adaptive Margin
# -------------------------------------------------------------------------------------------------

class NormAdaptiveMargin(nn.Module):
    '''
    Norm-adaptive margin for triplet loss that decays as anchor's norm increases.
    
    Formula: m(a) = m_0 * sech(||a||_L)
    
    This ensures that anchors near the leaf boundary (large norm) have smaller margins,
    making the loss more adaptive to the hyperbolic geometry.
    '''
    
    def __init__(
        self,
        base_margin: float = 0.5,
        curvature: float = 1.0
    ):
        '''
        Initialize norm-adaptive margin.
        
        Args:
            base_margin: Base margin m_0
            curvature: Hyperbolic curvature parameter c
        '''
        super().__init__()
        self.base_margin = base_margin
        self.curvature = curvature
        
        # Use Lorentz distance for computing norms
        self.lorentz_distance = LorentzDistance(curvature)
    
    def compute_lorentz_norm(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Compute Lorentz norm (hyperbolic radius) for embeddings.
        
        Args:
            x: Hyperbolic embeddings (batch_size, embedding_dim+1)
        
        Returns:
            Lorentz norms (batch_size,)
        '''
        time_coord = x[:, 0]  # x₀
        spatial_coords = x[:, 1:]  # x₁...xₙ
        
        spatial_norm_sq = torch.sum(spatial_coords ** 2, dim=1)
        time_norm_sq = time_coord ** 2
        
        # Hyperbolic radius: r = sqrt(x0^2 - ||x_spatial||^2)
        # For valid hyperboloid: x0^2 - ||x_spatial||^2 = 1/c
        c = self.curvature
        lorentz_norm_sq = time_norm_sq - spatial_norm_sq
        lorentz_norm = torch.sqrt(torch.clamp(lorentz_norm_sq, min=1e-8))
        
        return lorentz_norm
    
    def forward(self, anchor_emb: torch.Tensor) -> torch.Tensor:
        '''
        Compute norm-adaptive margin for each anchor.
        
        Args:
            anchor_emb: Anchor embeddings (batch_size, embedding_dim+1)
        
        Returns:
            Adaptive margins (batch_size,)
        '''
        # Compute Lorentz norm for each anchor
        lorentz_norms = self.compute_lorentz_norm(anchor_emb)  # (batch_size,)
        
        # Compute sech(||a||_L) = 1 / cosh(||a||_L)
        # sech is numerically stable: 1 / cosh(x)
        cosh_norms = torch.cosh(lorentz_norms)
        sech_norms = 1.0 / (cosh_norms + 1e-8)  # Add small epsilon for stability
        
        # Adaptive margin: m(a) = m_0 * sech(||a||_L)
        adaptive_margins = self.base_margin * sech_norms
        
        return adaptive_margins

