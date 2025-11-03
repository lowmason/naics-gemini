# -------------------------------------------------------------------------------------------------
# Imports and settings
# -------------------------------------------------------------------------------------------------

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------------------------------------
# Hyperbolic projection 
# -------------------------------------------------------------------------------------------------

class HyperbolicProjection(nn.Module):
    
    def __init__(self, input_dim: int, curvature: float = 1.0):
        super().__init__()
        
        self.input_dim = input_dim
        self.c = curvature
        
        self.projection = nn.Linear(input_dim, input_dim + 1)
    
    def exp_map_zero(self, v: torch.Tensor) -> torch.Tensor:
        
        sqrt_c = torch.sqrt(torch.tensor(self.c, device=v.device))
        norm_v = torch.norm(v, p=2, dim=1, keepdim=True)
        norm_v = torch.clamp(norm_v, min=1e-8)
        
        sinh_term = torch.sinh(norm_v / sqrt_c)
        
        x0 = torch.cosh(norm_v / sqrt_c)
        x_rest = (sinh_term * v) / norm_v
        
        return torch.cat([x0, x_rest], dim=1)
    
    def forward(self, euclidean_embedding: torch.Tensor) -> torch.Tensor:
        
        tangent_vec = self.projection(euclidean_embedding)
        
        hyperbolic_embedding = self.exp_map_zero(tangent_vec)
        
        return hyperbolic_embedding


# -------------------------------------------------------------------------------------------------
# Lorentz hyperboloid distance
# -------------------------------------------------------------------------------------------------

class LorentzDistance(nn.Module):
    
    def __init__(self, curvature: float = 1.0):
        super().__init__()
        self.c = curvature
    
    def lorentz_dot(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        
        uv = u * v
        return torch.sum(uv[:, 1:], dim=1) - uv[:, 0]
    
    def forward(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        
        dot_product = self.lorentz_dot(u, v)
        
        clamped_dot = torch.clamp(dot_product, max=-1.0 - 1e-5)
        
        sqrt_c = torch.sqrt(torch.tensor(self.c, device=u.device))
        dist = sqrt_c * torch.acosh(-clamped_dot)
        
        return dist


# -------------------------------------------------------------------------------------------------
# Hyperbolic InfoNCE Loss
# -------------------------------------------------------------------------------------------------

class HyperbolicInfoNCELoss(nn.Module):
    
    def __init__(
        self,
        embedding_dim: int,
        temperature: float = 0.07,
        curvature: float = 1.0
    ):
        super().__init__()
        
        self.temperature = temperature
        
        self.hyperbolic_proj = HyperbolicProjection(embedding_dim, curvature)
        self.lorentz_distance = LorentzDistance(curvature)
    
    def forward(
        self,
        anchor_emb: torch.Tensor,
        positive_emb: torch.Tensor,
        negative_embs: torch.Tensor,
        batch_size: int,
        k_negatives: int
    ) -> torch.Tensor:
        
        anchor_hyp = self.hyperbolic_proj(anchor_emb)
        positive_hyp = self.hyperbolic_proj(positive_emb)
        negative_hyp = self.hyperbolic_proj(negative_embs)
        
        pos_distances = self.lorentz_distance(anchor_hyp, positive_hyp)
        
        neg_distances = []
        for i in range(batch_size):
            anchor_repeated = anchor_hyp[i:i+1].repeat(k_negatives, 1)
            neg_batch = negative_hyp[i * k_negatives:(i + 1) * k_negatives]
            neg_dist = self.lorentz_distance(anchor_repeated, neg_batch)
            neg_distances.append(neg_dist)
        
        neg_distances = torch.stack(neg_distances)
        
        pos_similarities = -pos_distances / self.temperature
        neg_similarities = -neg_distances / self.temperature
        
        logits = torch.cat([
            pos_similarities.unsqueeze(1),
            neg_similarities
        ], dim=1)
        
        labels = torch.zeros(batch_size, dtype=torch.long, device=logits.device)
        
        loss = F.cross_entropy(logits, labels)
        
        return loss
