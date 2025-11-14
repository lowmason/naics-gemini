# -------------------------------------------------------------------------------------------------
# Imports and settings
# -------------------------------------------------------------------------------------------------

import logging
from typing import Dict, Optional

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from transformers import AutoModel

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
        use_moe: bool = True,
        num_experts: int = 4,
        top_k: int = 2,
        moe_hidden_dim: int = 1024,
        use_gradient_checkpointing: bool = True
    ):
        super().__init__()
        
        self.channels = ['title', 'description', 'excluded', 'examples']
        self.use_moe = use_moe
        
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
            task_type='FEATURE_EXTRACTION'
        )
        
        # Create separate LoRA-adapted encoder for each channel
        logger.info(f'Creating {len(self.channels)} channel encoders with LoRA (r={lora_r})...\n')
        self.encoders = nn.ModuleDict({
            channel: get_peft_model(
                AutoModel.from_pretrained(base_model_name),
                lora_config
            )
            for channel in self.channels
        })
        
        # Enable gradient checkpointing to save memory
        if use_gradient_checkpointing:
            for channel in self.channels:
                self.encoders[channel].enable_input_require_grads()
                self.encoders[channel].base_model.gradient_checkpointing_enable()
            logger.info('Gradient checkpointing enabled for memory efficiency\n')
        
        # Optional: Mixture of Experts
        if use_moe:
            from naics_embedder.model.moe import MixtureOfExperts
            logger.info(f'Initializing MoE with {num_experts} experts (top-k={top_k})...\n')
            self.moe = MixtureOfExperts(
                input_dim=self.embedding_dim * len(self.channels),  # 4 channels concatenated
                hidden_dim=moe_hidden_dim,
                num_experts=num_experts,
                top_k=top_k
            )
            # Projection to reduce MoE output back to embedding_dim
            self.moe_projection = nn.Linear(
                self.embedding_dim * len(self.channels),
                self.embedding_dim
            )
        else:
            self.moe = None
            self.moe_projection = None
        
        # Count trainable vs frozen parameters
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        
        logger.info(
            'Encoder initialized:\n'
            f'  • embedding_dim={self.embedding_dim}\n'
            f'  • use_moe={use_moe}\n'
            f'  • trainable params: {trainable_params:,} / {total_params:,} '
            f'({100 * trainable_params / total_params:.2f}%)\n'
        )
    
    
    def forward(
        self,
        channel_inputs: Dict[str, Dict[str, torch.Tensor]]
    ) -> Dict[str, Optional[torch.Tensor]]:
        '''
        Forward pass through multi-channel encoder.
        
        Args:
            channel_inputs: Dict mapping channel names to tokenized inputs
                           Each channel has 'input_ids' and 'attention_mask'
        
        Returns:
            Dict with 'embedding' and optional 'load_balancing_loss'
        '''
        channel_embeddings = []
        
        # Encode each channel separately
        for channel in self.channels:
            inputs = channel_inputs[channel]
            
            # Forward through LoRA-adapted encoder
            outputs = self.encoders[channel](
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask']
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
        
        # Optional: Pass through MoE
        if self.moe is not None:
            moe_output, gate_probs, top_k_indices = self.moe(combined)
            
            output = self.moe_projection(moe_output)
            return {
                'embedding': output,
                'gate_probs': gate_probs,
                'top_k_indices': top_k_indices
            }
        else:
            # Simple projection to reduce dimension
            projection = nn.Linear(
                self.embedding_dim * len(self.channels),
                self.embedding_dim
            ).to(combined.device)
            output = projection(combined)
            
            return {
                'embedding': output,
                'gate_probs': None,
                'top_k_indices': None
            }