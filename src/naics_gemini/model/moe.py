# -------------------------------------------------------------------------------------------------
# Imports and settings
# -------------------------------------------------------------------------------------------------

import logging
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------------------------------------
# Class to enable Mixture of Experts (MoE) layer with Top-2 gating
# -------------------------------------------------------------------------------------------------

class MixtureOfExpertsLayer(nn.Module):
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_experts: int = 4,
        top_k: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.top_k = top_k
        
        self.gate = nn.Linear(input_dim, num_experts)
        
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, input_dim)
            )
            for _ in range(num_experts)
        ])
        
        self.load_balancing_loss_coef = 0.01
    
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        
        batch_size = x.shape[0]
        
        gate_logits = self.gate(x)
        gate_probs = F.softmax(gate_logits, dim=-1)
        
        top_k_probs, top_k_indices = torch.topk(gate_probs, self.top_k, dim=-1)
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        
        output = torch.zeros_like(x)
        
        for i in range(self.top_k):
            expert_indices = top_k_indices[:, i]
            expert_weights = top_k_probs[:, i].unsqueeze(-1)
            
            for expert_id in range(self.num_experts):
                mask = (expert_indices == expert_id)
                if mask.any():
                    expert_input = x[mask]
                    expert_output = self.experts[expert_id](expert_input)
                    output[mask] += expert_weights[mask] * expert_output
        
        importance = gate_probs.sum(dim=0)
        importance = importance / importance.sum()
        
        load = torch.zeros(self.num_experts, device=x.device)
        for i in range(self.top_k):
            for expert_id in range(self.num_experts):
                mask = (top_k_indices[:, i] == expert_id)
                load[expert_id] += mask.float().sum()
        load = load / (batch_size * self.top_k)
        
        load_balancing_loss = self.num_experts * (importance * load).sum()
        
        return output, load_balancing_loss
