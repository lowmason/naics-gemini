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
# Mixture of Experts (MoE) Layer
# -------------------------------------------------------------------------------------------------

class MixtureOfExperts(nn.Module):

    '''
    Mixture of Experts layer with top-k gating and load balancing.
    
    Each expert is a simple feedforward network. The gating network decides
    which experts to use for each input, and load balancing ensures experts
    are utilized evenly.
    '''
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 1024,
        num_experts: int = 4,
        top_k: int = 2
    ):
        
        '''
        Initialize Mixture of Experts layer.
        
        Args:
            input_dim: Input dimension (e.g., 768 * 4 for 4 channels)
            hidden_dim: Hidden dimension for each expert
            num_experts: Number of expert networks
            top_k: Number of experts to select for each input
        '''

        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.top_k = top_k
        
        # Gating network: decides which experts to use
        self.gate = nn.Linear(input_dim, num_experts)
        
        # Expert networks: simple 2-layer MLPs
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, input_dim)
            )
            for _ in range(num_experts)
        ])
        
        logger.info(
            'MoE initialized:\n'
            f'  • {num_experts} experts\n'
            f'  • input_dim={input_dim}\n'
            f'  • hidden_dim={hidden_dim}\n'
            f'  • top_k={top_k}\n'
        )
    
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        '''
        Forward pass through MoE layer.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
        
        Returns:
            Tuple of:
                - Output tensor of shape (batch_size, input_dim)
                - Gating probabilities of shape (batch_size, num_experts)
                - Top-k indices of shape (batch_size, top_k)
        '''
        
        # Compute gating scores for all experts
        gate_logits = self.gate(x)  # (batch_size, num_experts)
        gate_probs = F.softmax(gate_logits, dim=1) # (batch_size, num_experts)
        
        # Select top-k experts
        top_k_logits, top_k_indices = torch.topk(gate_logits, self.top_k, dim=1)
        
        # Compute gating weights with softmax over top-k
        top_k_gates = F.softmax(top_k_logits, dim=1)  # (batch_size, top_k)
        
        # ----- The local loss calculation is REMOVED -----
        
        # Initialize output
        output = torch.zeros_like(x)
        
        # Process each expert
        for i in range(self.num_experts):
            # Find which batch items use this expert
            expert_mask = (top_k_indices == i).any(dim=1)  # (batch_size,)
            
            if expert_mask.any():
                # Get inputs for this expert
                expert_input = x[expert_mask]
                
                # Forward through expert
                expert_output = self.experts[i](expert_input)
                
                # Get gating weights for this expert
                # Find position of expert i in top_k for each batch item
                expert_positions = (top_k_indices[expert_mask] == i).float()
                expert_gates = (
                    top_k_gates[expert_mask] * expert_positions
                ).sum(dim=1, keepdim=True)
                
                # Weight expert output by gate
                weighted_output = expert_output * expert_gates
                
                # Add to output
                output[expert_mask] += weighted_output
        
        # Return materials for global loss calculation
        return output, gate_probs, top_k_indices


# -------------------------------------------------------------------------------------------------
# Helper function for creating MoE
# -------------------------------------------------------------------------------------------------

def create_moe_layer(
    input_dim: int,
    hidden_dim: int = 1024,
    num_experts: int = 4,
    top_k: int = 2
) -> MixtureOfExperts:
    
    '''
    Factory function to create a MoE layer.
    
    Args:
        input_dim: Input dimension
        hidden_dim: Hidden dimension for experts
        num_experts: Number of experts
        top_k: Number of experts to activate
    
    Returns:
        MixtureOfExperts module
    '''

    return MixtureOfExperts(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_experts=num_experts,
        top_k=top_k
    )