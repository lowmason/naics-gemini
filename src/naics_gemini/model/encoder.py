# -------------------------------------------------------------------------------------------------
# Imports and settings
# -------------------------------------------------------------------------------------------------

import logging
from typing import Dict

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from transformers import AutoModel

from naics_gemini.model.moe import MixtureOfExpertsLayer

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------------------------------------
# Class to encode multiple text channels and fuse embeddings
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
        moe_hidden_dim: int = 1024
    ):
        
        super().__init__()
        
        self.channels = ['title', 'description', 'excluded', 'examples']
        self.use_moe = use_moe
        
        base_model = AutoModel.from_pretrained(base_model_name)
        self.embedding_dim = base_model.config.hidden_size
        
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=['query', 'value'],
            lora_dropout=lora_dropout,
            bias='none',
            task_type='FEATURE_EXTRACTION'
        )
        
        self.encoders = nn.ModuleDict({
            channel: get_peft_model(
                AutoModel.from_pretrained(base_model_name),
                lora_config
            )
            for channel in self.channels
        })
        
        if use_moe:
            self.moe = MixtureOfExpertsLayer(
                input_dim=self.embedding_dim * len(self.channels),
                hidden_dim=moe_hidden_dim,
                num_experts=num_experts,
                top_k=top_k
            )
        else:
            self.fusion = nn.Linear(
                self.embedding_dim * len(self.channels),
                self.embedding_dim
            )
    
    def encode_channel(
        self,
        channel: str,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        
        outputs = self.encoders[channel](
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        embeddings = outputs.last_hidden_state[:, 0, :]
        
        return embeddings
    
    def forward(
        self,
        channel_inputs: Dict[str, Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        
        channel_embeddings = []
        
        for channel in self.channels:
            input_ids = channel_inputs[channel]['input_ids']
            attention_mask = channel_inputs[channel]['attention_mask']
            
            embedding = self.encode_channel(channel, input_ids, attention_mask)
            channel_embeddings.append(embedding)
        
        concatenated = torch.cat(channel_embeddings, dim=-1)
        
        if self.use_moe:
            fused_embedding, load_loss = self.moe(concatenated)
            return {
                'embedding': fused_embedding,
                'load_balancing_loss': load_loss
            }
        else:
            fused_embedding = self.fusion(concatenated)
            return {
                'embedding': fused_embedding,
                'load_balancing_loss': torch.tensor(0.0, device=fused_embedding.device)
            }
