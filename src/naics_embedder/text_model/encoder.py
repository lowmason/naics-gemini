# -------------------------------------------------------------------------------------------------
# Imports and settings
# -------------------------------------------------------------------------------------------------

import logging
from typing import Dict, Optional

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from transformers import AutoModel

from naics_embedder.text_model.hyperbolic import HyperbolicProjection
from naics_embedder.text_model.moe import MixtureOfExperts

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------------------------------
# Multi-Channel Encoder with LoRA
# -------------------------------------------------------------------------------------------------

class MultiChannelEncoder(nn.Module):

    def __init__(
        self,
        base_model_name: str = 'sentence-transformers/all-mpnet-base-v2',
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        num_experts: int = 4,
        top_k: int = 2,
        moe_hidden_dim: int = 1024,
        use_gradient_checkpointing: bool = True,
        curvature: float = 1.0,
    ):
        super().__init__()

        self.channels = ['title', 'description', 'excluded', 'examples']

        # Get base model to determine embedding dimension
        base_model = AutoModel.from_pretrained(base_model_name)
        self.embedding_dim = base_model.config.hidden_size

        # Configure LoRA with universal target (works with any model)
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules='all-linear',  # Universal: targets all linear layers
            lora_dropout=lora_dropout,
            bias='none',
            task_type='FEATURE_EXTRACTION',
        )

        # Create separate LoRA-adapted encoder for each channel
        logger.info(f'Creating {len(self.channels)} channel encoders with LoRA (r={lora_r})...\n')
        self.encoders = nn.ModuleDict(
            {
                channel: get_peft_model(AutoModel.from_pretrained(base_model_name), lora_config)
                for channel in self.channels
            }
        )

        # Enable gradient checkpointing to save memory
        if use_gradient_checkpointing:
            for channel in self.channels:
                encoder = self.encoders[channel]  # type: ignore[assignment]
                encoder.enable_input_require_grads()  # type: ignore[attr-defined]
                base_model = encoder.base_model  # type: ignore[attr-defined]
                base_model.gradient_checkpointing_enable()  # type: ignore[attr-defined]
            logger.info('Gradient checkpointing enabled for memory efficiency\n')

        # Mixture of Experts
        logger.info(f'Initializing MoE with {num_experts} experts (top-k={top_k})...\n')
        self.moe = MixtureOfExperts(
            input_dim=self.embedding_dim * len(self.channels),  # 4 channels concatenated
            hidden_dim=moe_hidden_dim,
            num_experts=num_experts,
            top_k=top_k,
        )
        # Projection to reduce MoE output back to embedding_dim
        self.moe_projection = nn.Linear(self.embedding_dim * len(self.channels), self.embedding_dim)

        # Hyperbolic projection: maps Euclidean embeddings to Lorentz hyperboloid
        # This ensures embeddings stay in hyperbolic space end-to-end
        self.hyperbolic_proj = HyperbolicProjection(
            input_dim=self.embedding_dim, curvature=curvature
        )
        self.curvature = curvature

        # Count trainable vs frozen parameters
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())

        logger.info(
            'Encoder initialized:\n'
            f'  • embedding_dim={self.embedding_dim}\n'
            f'  • trainable params: {trainable_params:,} / {total_params:,} '
            f'({100 * trainable_params / total_params:.2f}%)\n'
        )

    def forward(self, channel_inputs: Dict[str, Dict[str, torch.Tensor]]
                ) -> Dict[str, Optional[torch.Tensor]]:
        '''
        Forward pass through multi-channel encoder.

        Args:
            channel_inputs: Dict mapping channel names to tokenized inputs
                           Each channel has 'input_ids' and 'attention_mask'

        Returns:
            Dict with:
                - 'embedding': Hyperbolic embedding in Lorentz model
                    (batch_size, embedding_dim+1)
                - 'embedding_euc': Euclidean embedding before projection
                    (batch_size, embedding_dim) [optional]
                - 'gate_probs': MoE gating probabilities
                - 'top_k_indices': MoE top-k expert indices
        '''
        channel_embeddings = []

        # Encode each channel separately
        for channel in self.channels:
            inputs = channel_inputs[channel]

            # Forward through LoRA-adapted encoder
            outputs = self.encoders[channel](
                input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask']
            )

            # Mean pooling over sequence dimension
            embeddings = outputs.last_hidden_state
            mask_expanded = inputs['attention_mask'].unsqueeze(-1).expand(embeddings.shape).float()
            sum_embeddings = torch.sum(embeddings * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            pooled = sum_embeddings / sum_mask

            channel_embeddings.append(pooled)

        # Concatenate channel embeddings
        combined = torch.cat(channel_embeddings, dim=1)  # (batch, embedding_dim * 4)

        # Pass through MoE
        moe_output, gate_probs, top_k_indices = self.moe(combined)

        # Project MoE output back to embedding_dim (Euclidean space)
        embedding_euc = self.moe_projection(moe_output)  # (batch_size, embedding_dim)

        # Project to hyperbolic space (Lorentz model)
        embedding_hyp = self.hyperbolic_proj(embedding_euc)  # (batch_size, embedding_dim+1)

        return {
            'embedding': embedding_hyp,  # Primary output: hyperbolic embedding
            'embedding_euc': embedding_euc,  # Optional: Euclidean embedding for diagnostics
            'gate_probs': gate_probs,
            'top_k_indices': top_k_indices,
        }
