# -------------------------------------------------------------------------------------------------
# Imports and settings
# -------------------------------------------------------------------------------------------------

import logging
from typing import Dict

import pytorch_lightning as pl
import torch

from naics_gemini.model.encoder import MultiChannelEncoder
from naics_gemini.model.loss import HyperbolicInfoNCELoss

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------------------------------------
# Main NAICS Contrastive Learning Model: combining encoder, loss, MoE, and hyperbolic projections
# -------------------------------------------------------------------------------------------------

class NAICSContrastiveModel(pl.LightningModule):
    
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
        temperature: float = 0.07,
        curvature: float = 1.0,
        learning_rate: float = 2e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 500,
        load_balancing_coef: float = 0.01
    ):
        super().__init__()
        
        self.save_hyperparameters()
        
        self.encoder = MultiChannelEncoder(
            base_model_name=base_model_name,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            use_moe=use_moe,
            num_experts=num_experts,
            top_k=top_k,
            moe_hidden_dim=moe_hidden_dim
        )
        
        self.loss_fn = HyperbolicInfoNCELoss(
            embedding_dim=self.encoder.embedding_dim,
            temperature=temperature,
            curvature=curvature
        )
        
        self.load_balancing_coef = load_balancing_coef

    
    def forward(
        self, 
        channel_inputs: Dict[str, Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        
        return self.encoder(channel_inputs)
    

    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        
        anchor_output = self(batch['anchor'])
        positive_output = self(batch['positive'])
        negative_output = self(batch['negatives'])
        
        anchor_emb = anchor_output['embedding']
        positive_emb = positive_output['embedding']
        negative_emb = negative_output['embedding']
        
        batch_size = batch['batch_size']
        k_negatives = batch['k_negatives']
        
        contrastive_loss = self.loss_fn(
            anchor_emb,
            positive_emb,
            negative_emb,
            batch_size,
            k_negatives
        )
        
        load_balancing_loss = (
            anchor_output['load_balancing_loss'] +
            positive_output['load_balancing_loss'] +
            negative_output['load_balancing_loss']
        ) / 3.0
        
        total_loss = contrastive_loss + self.load_balancing_coef * load_balancing_loss
        
        self.log('train/contrastive_loss', contrastive_loss, prog_bar=True)
        self.log('train/load_balancing_loss', load_balancing_loss, prog_bar=True)
        self.log('train/total_loss', total_loss, prog_bar=True)
        
        return total_loss
    

    def validation_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        
        anchor_output = self(batch['anchor'])
        positive_output = self(batch['positive'])
        negative_output = self(batch['negatives'])
        
        anchor_emb = anchor_output['embedding']
        positive_emb = positive_output['embedding']
        negative_emb = negative_output['embedding']
        
        batch_size = batch['batch_size']
        k_negatives = batch['k_negatives']
        
        contrastive_loss = self.loss_fn(
            anchor_emb,
            positive_emb,
            negative_emb,
            batch_size,
            k_negatives
        )
        
        self.log('val/contrastive_loss', contrastive_loss, prog_bar=True)
        
        return contrastive_loss
    

    def configure_optimizers(self):
        
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs,
            eta_min=1e-6
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1
            }
        }
