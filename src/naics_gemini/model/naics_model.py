# -------------------------------------------------------------------------------------------------
# Imports and settings
# -------------------------------------------------------------------------------------------------

import logging
from typing import Dict, Optional

import polars as pl
import pytorch_lightning as pyl
import torch

from naics_gemini.model.encoder import MultiChannelEncoder
from naics_gemini.model.evaluation import (
    EmbeddingEvaluator,
    EmbeddingStatistics,
    HierarchyMetrics,
)
from naics_gemini.model.loss import HyperbolicInfoNCELoss

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------------------------------------
# Main NAICS Contrastive Learning Model: combining encoder, loss, MoE, and hyperbolic projections
# -------------------------------------------------------------------------------------------------

class NAICSContrastiveModel(pyl.LightningModule):
    
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
        load_balancing_coef: float = 0.01,
        
        # Evaluation settings
        distances_path: Optional[str] = None,
        eval_every_n_epochs: int = 1,
        eval_sample_size: int = 500  # Subsample for faster evaluation
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
        
        # Initialize evaluation components
        self.embedding_eval = EmbeddingEvaluator()
        self.embedding_stats = EmbeddingStatistics()
        self.hierarchy_metrics = HierarchyMetrics()
        
        # Load ground truth distances for evaluation
        self.ground_truth_distances = None
        self.code_to_idx = None
        if distances_path:
            self._load_ground_truth_distances(distances_path)
        
        # Cache for validation embeddings
        self.validation_embeddings = {}
        self.validation_codes = []
    
    
    def _load_ground_truth_distances(self, distances_path: str):
        '''Load ground truth NAICS tree distances for evaluation.'''
        try:
            logger.info(f'Loading ground truth distances from {distances_path}')
            
            df = pl.read_parquet(distances_path)
            
            # Create code to index mapping
            all_codes = sorted(set(df['code_i'].to_list() + df['code_j'].to_list()))
            self.code_to_idx = {code: idx for idx, code in enumerate(all_codes)}
            
            # Build distance matrix
            n_codes = len(all_codes)
            dist_matrix = torch.zeros(n_codes, n_codes)
            
            for row in df.iter_rows(named=True):
                i = self.code_to_idx[row['code_i']]
                j = self.code_to_idx[row['code_j']]
                dist = row['distance']
                dist_matrix[i, j] = dist
                dist_matrix[j, i] = dist
            
            self.ground_truth_distances = dist_matrix
            logger.info(f'Loaded distance matrix: {n_codes} codes')
            
        except Exception as e:
            logger.warning(f'Could not load ground truth distances: {e}')
            self.ground_truth_distances = None
    
    
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
        
        # Cache embeddings for epoch-level evaluation
        if 'anchor_code' in batch:
            for i, code in enumerate(batch['anchor_code']):
                if code not in self.validation_embeddings:
                    self.validation_embeddings[code] = anchor_emb[i].detach().cpu()
                    self.validation_codes.append(code)
        
        return contrastive_loss
    
    
    def on_validation_epoch_end(self):
        '''Compute comprehensive evaluation metrics at the end of each validation epoch.'''
        
        # Only evaluate every N epochs
        if self.current_epoch % self.hparams.eval_every_n_epochs != 0:
            return
        
        # Need cached embeddings and ground truth distances
        if not self.validation_embeddings or self.ground_truth_distances is None:
            logger.warning('Skipping evaluation: missing embeddings or ground truth distances')
            return
        
        try:
            logger.info(f'Running evaluation metrics (epoch {self.current_epoch})...')
            
            # Stack embeddings in consistent order
            codes = sorted(self.validation_embeddings.keys())
            embeddings = torch.stack([
                self.validation_embeddings[code] for code in codes
            ]).to(self.device)
            
            # Subsample if too large
            if len(codes) > self.hparams.eval_sample_size:
                indices = torch.randperm(len(codes))[:self.hparams.eval_sample_size]
                embeddings = embeddings[indices]
                codes = [codes[i] for i in indices]
            
            # Get corresponding ground truth distances
            code_indices = [self.code_to_idx[code] for code in codes if code in self.code_to_idx]
            if len(code_indices) < 2:
                logger.warning('Not enough codes in ground truth for evaluation')
                return
            
            gt_dists = self.ground_truth_distances[code_indices][:, code_indices].to(self.device)
            
            # 1. Embedding statistics
            stats = self.embedding_stats.compute_statistics(embeddings)
            self.log('val/mean_norm', stats['mean_norm'])
            self.log('val/std_norm', stats['std_norm'])
            self.log('val/mean_pairwise_distance', stats['mean_pairwise_distance'])
            self.log('val/std_embedding', stats['std_embedding'])
            
            # 2. Collapse check
            collapse = self.embedding_stats.check_collapse(embeddings, threshold=0.01)
            self.log('val/variance_collapsed', float(collapse['variance_collapsed']))
            self.log('val/norm_collapsed', float(collapse['norm_collapsed']))
            self.log('val/distance_collapsed', float(collapse['distance_collapsed']))
            
            # 3. Compute embedding distances
            emb_dists = self.embedding_eval.compute_pairwise_distances(
                embeddings,
                metric='euclidean'
            )
            
            # 4. Hierarchy preservation metrics
            cophenetic_corr = self.hierarchy_metrics.cophenetic_correlation(
                emb_dists,
                gt_dists
            )
            self.log('val/cophenetic_correlation', cophenetic_corr, prog_bar=True)
            
            spearman_corr = self.hierarchy_metrics.spearman_correlation(
                emb_dists,
                gt_dists
            )
            self.log('val/spearman_correlation', spearman_corr)
            
            # 5. Distortion metrics
            distortion = self.hierarchy_metrics.distortion(emb_dists, gt_dists)
            self.log('val/mean_distortion', distortion['mean_distortion'])
            self.log('val/std_distortion', distortion['std_distortion'])
            
            logger.info(
                f'Evaluation complete: cophenetic={cophenetic_corr:.4f}, '
                f'spearman={spearman_corr:.4f}, '
                f'collapse={collapse["any_collapse"]}'
            )
            
        except Exception as e:
            logger.error(f'Error during evaluation: {e}', exc_info=True)
        
        finally:
            # Clear cache for next epoch
            self.validation_embeddings.clear()
            self.validation_codes.clear()
    

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
