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
        self, input_dim: int, hidden_dim: int = 1024, num_experts: int = 4, top_k: int = 2
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
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(hidden_dim, input_dim),
                )
                for _ in range(num_experts)
            ]
        )

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
        gate_probs = F.softmax(gate_logits, dim=1)  # (batch_size, num_experts)

        # Select top-k experts
        top_k_logits, top_k_indices = torch.topk(gate_logits, self.top_k, dim=1)

        # Compute gating weights with softmax over top-k
        top_k_gates = F.softmax(top_k_logits, dim=1)  # (batch_size, top_k)

        # ----- The local loss calculation is REMOVED -----

        batch_size = x.shape[0]

        # Fully vectorized expert routing using batched tensor operations
        # Expand input for each selected expert: (batch_size, top_k, input_dim)
        x_expanded = x.unsqueeze(1).expand(-1, self.top_k, -1)  # (batch_size, top_k, input_dim)
        x_flat = x_expanded.reshape(-1, self.input_dim)  # (batch_size * top_k, input_dim)

        # Flatten expert indices and gates
        expert_indices_flat = top_k_indices.reshape(-1)  # (batch_size * top_k,)
        gate_weights_flat = top_k_gates.reshape(-1, 1)  # (batch_size * top_k, 1)

        # Create batch indices for scatter operations
        batch_indices = (
            torch.arange(batch_size, device=x.device)
            .unsqueeze(1)
            .expand(-1, self.top_k)
            .reshape(-1)
        )  # (batch_size * top_k,)

        # Group inputs by expert for efficient batched processing
        # Sort by expert index to group all inputs for the same expert together
        sorted_indices = torch.argsort(expert_indices_flat)
        expert_indices_sorted = expert_indices_flat[sorted_indices]
        x_sorted = x_flat[sorted_indices]
        gate_weights_sorted = gate_weights_flat[sorted_indices]
        batch_indices_sorted = batch_indices[sorted_indices]

        # Find unique experts and their boundaries for batched processing
        # Use diff to find where expert indices change
        expert_changes = torch.cat(
            [
                torch.tensor([True], device=x.device),
                expert_indices_sorted[1:] != expert_indices_sorted[:-1],
                torch.tensor([True], device=x.device),
            ]
        )
        expert_boundaries = torch.where(expert_changes)[0]

        # Initialize output
        output = torch.zeros_like(x)

        # Process each expert's inputs in a single batched forward pass
        # This minimizes loop overhead by processing all inputs for each expert at once
        for i in range(len(expert_boundaries) - 1):
            start_idx = expert_boundaries[i]
            end_idx = expert_boundaries[i + 1]

            if end_idx > start_idx:
                expert_idx = int(expert_indices_sorted[start_idx].item())

                # Extract all inputs for this expert in one slice
                expert_input = x_sorted[start_idx:end_idx]  # (n_items, input_dim)

                # Single batched forward pass through the expert
                expert_output = self.experts[expert_idx](expert_input)  # (n_items, input_dim)

                # Get corresponding gate weights and batch indices
                expert_gates = gate_weights_sorted[start_idx:end_idx]  # (n_items, 1)
                expert_batch_indices = batch_indices_sorted[start_idx:end_idx]  # (n_items,)

                # Weight and accumulate using efficient scatter-add
                weighted_output = expert_output * expert_gates  # (n_items, input_dim)
                output.index_add_(0, expert_batch_indices, weighted_output)

        # Return materials for global loss calculation
        return output, gate_probs, top_k_indices


# -------------------------------------------------------------------------------------------------
# Helper function for creating MoE
# -------------------------------------------------------------------------------------------------


def create_moe_layer(
    input_dim: int, hidden_dim: int = 1024, num_experts: int = 4, top_k: int = 2
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
        input_dim=input_dim, hidden_dim=hidden_dim, num_experts=num_experts, top_k=top_k
    )
