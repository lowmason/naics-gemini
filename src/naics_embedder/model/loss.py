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


# -------------------------------------------------------------------------------------------------
# Rank Order Preservation Loss
# -------------------------------------------------------------------------------------------------

class RankOrderPreservationLoss(nn.Module):
    """
    Loss component that explicitly optimizes for rank order preservation (Spearman correlation).
    
    This loss penalizes violations of rank order: if code A is closer to code B than to code C
    in the tree (ground truth), then the embedding distance A-B should be smaller than A-C.
    
    This directly optimizes for Spearman correlation by ensuring relative distance ordering
    matches ground truth ordering.
    """
    
    def __init__(
        self,
        tree_distances: torch.Tensor,
        code_to_idx: Dict[str, int],
        weight: float = 0.1,
        min_distance: float = 0.1,
        margin: float = 0.1
    ):
        super().__init__()
        # Register as buffer so it moves with model to correct device
        self.register_buffer('tree_distances', tree_distances)
        self.code_to_idx = code_to_idx
        self.weight = weight
        self.min_distance = min_distance
        self.margin = margin  # Margin for ranking loss
        
    def forward(
        self,
        embeddings: torch.Tensor,
        codes: List[str],
        lorentz_distance_fn
    ) -> torch.Tensor:
        """
        Compute rank order preservation loss.
        
        For each anchor code, we check if the relative ordering of distances to other codes
        matches the ground truth ordering. We penalize violations using a margin-based ranking loss.
        
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
        
        if len(valid_indices) < 3:  # Need at least 3 codes for ranking
            return torch.tensor(0.0, device=embeddings.device)
        
        # Get embeddings for valid codes
        valid_embeddings = embeddings[valid_indices]
        N = valid_embeddings.shape[0]
        
        # Get ground truth distance matrix indices
        gt_indices = torch.tensor([
            self.code_to_idx[code] for code in valid_codes
        ], device=embeddings.device)
        
        # Get ground truth distances for these codes
        gt_dists = self.tree_distances[gt_indices][:, gt_indices]
        
        # Compute embedding distances (efficiently using vectorized operations)
        # For each pair (i, j), compute distance
        emb_dists = torch.zeros((N, N), device=embeddings.device)
        
        # Use vectorized computation if possible, otherwise loop
        for i in range(N):
            for j in range(i+1, N):
                dist = lorentz_distance_fn(
                    valid_embeddings[i:i+1],
                    valid_embeddings[j:j+1]
                )
                emb_dists[i, j] = dist
                emb_dists[j, i] = dist
        
        # Ranking loss: for each anchor i, compare all pairs (j, k) where j != k != i
        # If gt_dists[i, j] < gt_dists[i, k], then emb_dists[i, j] should be < emb_dists[i, k] + margin
        total_loss = torch.tensor(0.0, device=embeddings.device)
        num_violations = 0
        
        for anchor_idx in range(N):
            # Get distances from anchor to all other codes
            anchor_gt_dists = gt_dists[anchor_idx]  # (N,)
            anchor_emb_dists = emb_dists[anchor_idx]  # (N,)
            
            # Filter out pairs with very small tree distances
            valid_mask = anchor_gt_dists >= self.min_distance
            valid_mask[anchor_idx] = False  # Exclude self
            
            if valid_mask.sum() < 2:
                continue
            
            # Get valid indices
            valid_j = torch.where(valid_mask)[0]
            
            # For each pair (j, k) where j < k and both are valid
            for idx_j, j in enumerate(valid_j):
                for k in valid_j[idx_j+1:]:
                    gt_dist_j = anchor_gt_dists[j]
                    gt_dist_k = anchor_gt_dists[k]
                    emb_dist_j = anchor_emb_dists[j]
                    emb_dist_k = anchor_emb_dists[k]
                    
                    # Check if rank order is violated
                    if gt_dist_j < gt_dist_k:
                        # j should be closer than k
                        # Penalize if emb_dist_j >= emb_dist_k (violation)
                        violation = torch.clamp(emb_dist_j - emb_dist_k + self.margin, min=0.0)
                        total_loss = total_loss + violation
                        if violation > 0:
                            num_violations += 1
                    elif gt_dist_k < gt_dist_j:
                        # k should be closer than j
                        # Penalize if emb_dist_k >= emb_dist_j (violation)
                        violation = torch.clamp(emb_dist_k - emb_dist_j + self.margin, min=0.0)
                        total_loss = total_loss + violation
                        if violation > 0:
                            num_violations += 1
                    # If gt_dist_j == gt_dist_k, no constraint (ties are allowed)
        
        # Average loss over all violations
        if num_violations > 0:
            avg_loss = total_loss / num_violations
        else:
            avg_loss = torch.tensor(0.0, device=embeddings.device)
        
        return self.weight * avg_loss


# -------------------------------------------------------------------------------------------------
# LambdaRank Loss (Global Ranking)
# -------------------------------------------------------------------------------------------------

class LambdaRankLoss(nn.Module):
    """
    LambdaRank loss for global ranking optimization.
    
    Unlike pairwise ranking loss, LambdaRank considers the full ranking list
    (1 positive + k negatives) for each anchor and directly optimizes for NDCG.
    
    Key advantages:
    1. Position-aware: Top positions matter more (via NDCG)
    2. Global optimization: Considers entire ranking list, not just pairs
    3. Direct NDCG optimization: Gradients scaled by NDCG change from swapping
    
    This is particularly effective for contrastive learning where we have
    1 positive and k negatives (e.g., 24-32 negatives) per anchor.
    """
    
    def __init__(
        self,
        tree_distances: torch.Tensor,
        code_to_idx: Dict[str, int],
        weight: float = 0.15,
        sigma: float = 1.0,
        ndcg_k: int = 10
    ):
        super().__init__()
        # Register as buffer so it moves with model to correct device
        self.register_buffer('tree_distances', tree_distances)
        self.code_to_idx = code_to_idx
        self.weight = weight
        self.sigma = sigma  # Smoothing parameter for RankNet probability
        self.ndcg_k = ndcg_k  # Top-k for NDCG computation
        
    def _compute_ndcg(
        self,
        relevance_scores: torch.Tensor,
        distances: torch.Tensor,
        k: Optional[int] = None
    ) -> torch.Tensor:
        """
        Compute NDCG (Normalized Discounted Cumulative Gain).
        
        Args:
            relevance_scores: Ground truth relevance (higher = more relevant) (N,)
            distances: Predicted distances (lower = more relevant) (N,)
            k: Top-k for NDCG (None = all)
            
        Returns:
            NDCG score (scalar)
        """
        if k is None:
            k = len(relevance_scores)
        k = min(k, len(relevance_scores))
        
        # Sort by distances (ascending) to get ranking
        _, sorted_indices = torch.sort(distances, descending=False)
        
        # Get relevance scores in ranked order
        sorted_relevance = relevance_scores[sorted_indices[:k]]
        
        # Compute DCG: sum of (relevance / log2(position + 1))
        positions = torch.arange(1, k + 1, dtype=torch.float32, device=distances.device)
        dcg = torch.sum(sorted_relevance / torch.log2(positions + 1))
        
        # Compute ideal DCG (IDCG): sort relevance descending
        ideal_relevance, _ = torch.sort(relevance_scores, descending=True)
        ideal_dcg = torch.sum(ideal_relevance[:k] / torch.log2(positions + 1))
        
        # NDCG = DCG / IDCG
        if ideal_dcg > 0:
            ndcg = dcg / ideal_dcg
        else:
            ndcg = torch.tensor(0.0, device=distances.device)
        
        return ndcg
    
    def _compute_lambdas(
        self,
        relevance_scores: torch.Tensor,
        distances: torch.Tensor,
        ndcg_k: int
    ) -> torch.Tensor:
        """
        Compute LambdaRank lambdas (gradients).
        
        Lambda for pair (i, j) = |delta_NDCG| * (1 / (1 + exp(sigma * (s_i - s_j))))
        where delta_NDCG is the change in NDCG from swapping i and j.
        
        Args:
            relevance_scores: Ground truth relevance (N,)
            distances: Predicted distances (N,)
            ndcg_k: Top-k for NDCG computation
            
        Returns:
            Lambdas for each pair (N, N)
        """
        N = len(relevance_scores)
        lambdas = torch.zeros((N, N), device=distances.device)
        
        # Current NDCG
        current_ndcg = self._compute_ndcg(relevance_scores, distances, ndcg_k)
        
        for i in range(N):
            for j in range(i + 1, N):
                # Swap i and j in ranking
                swapped_distances = distances.clone()
                swapped_distances[i], swapped_distances[j] = distances[j], distances[i]
                
                # Compute NDCG after swap
                swapped_ndcg = self._compute_ndcg(relevance_scores, swapped_distances, ndcg_k)
                
                # Delta NDCG
                delta_ndcg = torch.abs(swapped_ndcg - current_ndcg)
                
                # RankNet probability: P(i > j) = 1 / (1 + exp(sigma * (s_i - s_j)))
                # where s_i = -distance_i (higher score = more relevant)
                score_i = -distances[i]
                score_j = -distances[j]
                prob_i_beats_j = 1.0 / (1.0 + torch.exp(self.sigma * (score_i - score_j)))
                
                # Lambda: gradient scaled by delta_NDCG
                if relevance_scores[i] > relevance_scores[j]:
                    # i should be ranked higher than j
                    lambda_ij = delta_ndcg * (1.0 - prob_i_beats_j)
                    lambdas[i, j] = lambda_ij
                    lambdas[j, i] = -lambda_ij
                elif relevance_scores[j] > relevance_scores[i]:
                    # j should be ranked higher than i
                    lambda_ij = delta_ndcg * (1.0 - (1.0 - prob_i_beats_j))
                    lambdas[i, j] = -lambda_ij
                    lambdas[j, i] = lambda_ij
        
        return lambdas
    
    def forward(
        self,
        anchor_emb: torch.Tensor,
        positive_emb: torch.Tensor,
        negative_embs: torch.Tensor,
        anchor_codes: List[str],
        positive_codes: List[str],
        negative_codes: List[List[str]],
        lorentz_distance_fn,
        batch_size: int,
        k_negatives: int
    ) -> torch.Tensor:
        """
        Compute LambdaRank loss.
        
        For each anchor, creates a ranking list: [positive, negative_1, ..., negative_k]
        and optimizes for NDCG using LambdaRank gradients.
        
        Args:
            anchor_emb: Anchor embeddings (batch_size, embedding_dim+1)
            positive_emb: Positive embeddings (batch_size, embedding_dim+1)
            negative_embs: Negative embeddings (batch_size * k_negatives, embedding_dim+1)
            anchor_codes: List of anchor NAICS codes (batch_size,)
            positive_codes: List of positive NAICS codes (batch_size,)
            negative_codes: List of lists of negative codes (batch_size, k_negatives)
            lorentz_distance_fn: Function to compute Lorentz distances
            batch_size: Batch size
            k_negatives: Number of negatives per anchor
            
        Returns:
            Loss scalar
        """
        # Reshape negatives: (batch_size * k_negatives, D+1) -> (batch_size, k_negatives, D+1)
        negative_embs_reshaped = negative_embs.view(batch_size, k_negatives, -1)
        
        total_loss = torch.tensor(0.0, device=anchor_emb.device)
        num_valid_anchors = 0
        
        for b in range(batch_size):
            anchor_code = anchor_codes[b]
            positive_code = positive_codes[b]
            negative_codes_b = negative_codes[b]
            
            # Skip if anchor or positive not in ground truth
            if anchor_code not in self.code_to_idx or positive_code not in self.code_to_idx:
                continue
            
            # Get embeddings for this anchor
            anchor_emb_b = anchor_emb[b:b+1]  # (1, D+1)
            positive_emb_b = positive_emb[b:b+1]  # (1, D+1)
            negative_embs_b = negative_embs_reshaped[b]  # (k_negatives, D+1)
            
            # Compute distances from anchor to positive and negatives
            pos_dist = lorentz_distance_fn(anchor_emb_b, positive_emb_b).squeeze()  # scalar
            
            # Compute distances to all negatives
            neg_dists = []
            valid_neg_indices = []
            valid_neg_codes = []
            
            for i, neg_code in enumerate(negative_codes_b):
                if neg_code in self.code_to_idx:
                    neg_emb = negative_embs_b[i:i+1]  # (1, D+1)
                    neg_dist = lorentz_distance_fn(anchor_emb_b, neg_emb).squeeze()  # scalar
                    neg_dists.append(neg_dist)
                    valid_neg_indices.append(i)
                    valid_neg_codes.append(neg_code)
            
            if len(valid_neg_indices) < 1:
                continue
            
            # Create ranking list: [positive, negative_1, ..., negative_k]
            # Distances: [pos_dist, neg_dist_1, ..., neg_dist_k]
            all_distances = torch.cat([pos_dist.unsqueeze(0), torch.stack(neg_dists)])
            
            # Compute relevance scores based on tree distances
            # Lower tree distance = higher relevance
            anchor_idx = self.code_to_idx[anchor_code]
            pos_idx = self.code_to_idx[positive_code]
            
            # Get tree distance for positive
            pos_tree_dist = self.tree_distances[anchor_idx, pos_idx]
            
            # Get tree distances for negatives
            neg_tree_dists = []
            for neg_code in valid_neg_codes:
                neg_idx = self.code_to_idx[neg_code]
                neg_tree_dist = self.tree_distances[anchor_idx, neg_idx]
                neg_tree_dists.append(neg_tree_dist)
            
            all_tree_dists = torch.cat([
                pos_tree_dist.unsqueeze(0),
                torch.stack(neg_tree_dists)
            ])
            
            # Convert tree distances to relevance scores
            # Lower distance = higher relevance
            # Use inverse distance as relevance (add small epsilon to avoid division by zero)
            max_dist = all_tree_dists.max()
            relevance_scores = (max_dist - all_tree_dists + 1e-6) / (max_dist + 1e-6)
            
            # Compute lambdas
            lambdas = self._compute_lambdas(
                relevance_scores,
                all_distances,
                self.ndcg_k
            )
            
            # LambdaRank loss: sum of lambdas * distance differences
            # For each pair (i, j), loss contribution = lambda_ij * (distance_i - distance_j)
            loss_contrib = torch.tensor(0.0, device=anchor_emb.device)
            N = len(all_distances)
            
            for i in range(N):
                for j in range(i + 1, N):
                    if lambdas[i, j] != 0:
                        # Loss = lambda * (distance_i - distance_j)
                        # This encourages correct ranking based on NDCG gradients
                        loss_contrib += lambdas[i, j] * (all_distances[i] - all_distances[j])
            
            total_loss += loss_contrib
            num_valid_anchors += 1
        
        # Average over valid anchors
        if num_valid_anchors > 0:
            avg_loss = total_loss / num_valid_anchors
        else:
            avg_loss = torch.tensor(0.0, device=anchor_emb.device)
        
        return self.weight * avg_loss