# -------------------------------------------------------------------------------------------------
# Imports and settings
# -------------------------------------------------------------------------------------------------

import logging
import math
from typing import Dict, Optional

import polars as pl
import pytorch_lightning as pyl
import torch
from sklearn.cluster import KMeans

from naics_embedder.model.encoder import MultiChannelEncoder
from naics_embedder.model.evaluation import (
    EmbeddingEvaluator,
    EmbeddingStatistics,
    HierarchyMetrics,
)
from naics_embedder.model.loss import HyperbolicInfoNCELoss

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
        fn_curriculum_start_epoch: int = 10,
        fn_cluster_every_n_epochs: int = 5,
        fn_num_clusters: int = 1000,
        
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

        self.code_to_pseudo_label: Dict[str, int] = {}
    
    
    def _load_ground_truth_distances(self, distances_path: str):

        '''Load ground truth NAICS tree distances for evaluation.'''
        
        try:
            logger.info(
                f'Loading ground truth distances\n'
                f'  • from: {distances_path}')
            
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
            logger.info(f'  • distance matrix: [{n_codes}, {n_codes}]\n')
            
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
        
        false_negative_mask = None
        if (
            self.current_epoch >= self.hparams.fn_curriculum_start_epoch and 
            self.code_to_pseudo_label and 
            'negatives_code' in batch
        ):
            
            try:
                anchor_labels = torch.tensor(
                    [self.code_to_pseudo_label.get(code, -1) for code in batch['anchor_code']],
                    device=self.device
                )

                neg_labels = torch.tensor(
                    [
                        [self.code_to_pseudo_label.get(code, -2) for code in neg_codes_for_anchor]
                        for neg_codes_for_anchor in batch['negatives_code']
                    ],
                    device=self.device
                )

                false_negative_mask = (anchor_labels.unsqueeze(1) == neg_labels)
                
                valid_anchor_mask = (anchor_labels > -1).unsqueeze(1)
                valid_neg_mask = (neg_labels > -2)
                false_negative_mask = false_negative_mask & valid_anchor_mask & valid_neg_mask
            
            except Exception as e:
                logger.warning(f'Failed to create false negative mask: {e}')
                false_negative_mask = None
        
        contrastive_loss = self.loss_fn(
            anchor_emb,
            positive_emb,
            negative_emb,
            batch_size,
            k_negatives,false_negative_mask=false_negative_mask
        )
        
        all_gate_probs = []
        all_top_k_indices = []
        for output in (anchor_output, positive_output, negative_output):
            if output['gate_probs'] is not None:
                all_gate_probs.append(output['gate_probs'])
                all_top_k_indices.append(output['top_k_indices'])
        
        if not all_gate_probs:
            # MoE is disabled, loss is zero
            full_load_balancing_loss = torch.tensor(0.0, device=self.device)
        else:
            # Concatenate all stats from anchor, positive, and negatives
            gate_probs = torch.cat(all_gate_probs, dim=0) # (total_batch_size, num_experts)
            top_k_indices = torch.cat(all_top_k_indices, dim=0) # (total_batch_size, top_k)
            
            # These are the MICRO-BATCH stats
            total_tokens = gate_probs.shape[0]
            num_experts = gate_probs.shape[1]

            # Calculate P_i (average probability for expert i)
            # P_i = gate_probs[:, i].mean()
            P_micro = gate_probs.mean(dim=0) # Shape (num_experts,)
            
            # Calculate f_i (fraction of tokens routed to expert i)
            expert_counts_micro = torch.zeros(num_experts, device=self.device)
            for i in range(num_experts):
                expert_counts_micro[i] = (top_k_indices == i).any(dim=1).sum()
            
            f_micro = expert_counts_micro / total_tokens # Shape (num_experts,)
            
            # --- GLOBAL-BATCH Synchronization [cite: 278-279] ---
            # Sync P and f. We sync the components (sums) for a more stable average.
            if self.trainer.is_global_zero and torch.distributed.is_initialized():
                # Get world size
                world_size = torch.distributed.get_world_size()
                
                if world_size > 1:
                    # Sums for calculating global P_i
                    global_prob_sum = gate_probs.sum(dim=0)
                    # Sums for calculating global f_i
                    global_expert_counts = expert_counts_micro * total_tokens
                    global_total_tokens = torch.tensor(
                        total_tokens, 
                        dtype=torch.float, 
                        device=self.device
                    )
                    
                    # Sum across all workers
                    torch.distributed.all_reduce(
                        global_prob_sum, 
                        op=torch.distributed.ReduceOp.SUM
                    )
                    torch.distributed.all_reduce(
                        global_expert_counts, 
                        op=torch.distributed.ReduceOp.SUM
                    )
                    torch.distributed.all_reduce(
                        global_total_tokens, 
                        op=torch.distributed.ReduceOp.SUM
                    )

                    # Calculate global f_i and P_i
                    global_total_tokens_safe = torch.clamp(global_total_tokens, min=1e-6)
                    
                    f = global_expert_counts / global_total_tokens_safe
                    P = global_prob_sum / global_total_tokens_safe
                else:
                    f = f_micro
                    P = P_micro
            else:
                f = f_micro
                P = P_micro

            # Calculate Loss as per PDF: L_aux = alpha * N * sum(f_i * P_i) 
            # alpha is self.load_balancing_coef
            # N is num_experts
            
            # This is N * sum(f_i * P_i)
            unscaled_loss = num_experts * torch.sum(f * P)
            
            # This is alpha * (N * sum(f_i * P_i))
            full_load_balancing_loss = self.load_balancing_coef * unscaled_loss

        total_loss = contrastive_loss + full_load_balancing_loss

        batch_size = batch['batch_size']
        
        self.log(
            'train/contrastive_loss', 
            contrastive_loss, 
            prog_bar=True, 
            batch_size=batch_size
        )
        self.log(
            'train/load_balancing_loss', 
            full_load_balancing_loss, 
            prog_bar=True, 
            batch_size=batch_size
        )
        self.log(
            'train/total_loss', 
            total_loss, 
            prog_bar=True, 
            batch_size=batch_size
        )

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
        
        self.log(
            'val/contrastive_loss', 
            contrastive_loss, 
            prog_bar=True, 
            batch_size=batch_size
        )

        # Cache embeddings for epoch-level evaluation
        if 'anchor_code' in batch:
            for i, code in enumerate(batch['anchor_code']):
                if code not in self.validation_embeddings:
                    self.validation_embeddings[code] = anchor_emb[i].detach().cpu()
                    self.validation_codes.append(code)
        
        return contrastive_loss
    
    
    def _to_python_scalar(self, value):
        '''Convert any numeric value to a Python scalar for logging.'''
        if isinstance(value, torch.Tensor):
            return value.item()
        elif isinstance(value, (bool, int)):
            return int(value)
        else:
            return float(value)
        
    def _update_pseudo_labels(self):

        '''
        Runs clustering on the training dataset to generate pseudo-labels
        for false negative detection. [cite: 231-234]
        '''

        if not hasattr(self.trainer, 'train_dataloader'):
            logger.warning('Trainer has no train_dataloader, cannot update pseudo-labels.')
            return

        logger.info('Generating embeddings for pseudo-label clustering...')
        self.eval()
        all_embeddings = []
        all_codes = []

        try:            
            dataloader = self.trainer.train_dataloader
            
            for batch in dataloader:
                batch = self.transfer_batch_to_device(batch, self.device, 0) # 0 is dataloader_idx
                
                with torch.no_grad():
                    anchor_output = self(batch['anchor'])
                    embs = anchor_output['embedding'].cpu()
                    all_embeddings.append(embs)
                    all_codes.extend(batch['anchor_code'])
            
            all_embeddings = torch.cat(all_embeddings, dim=0).numpy()
            
            n_clusters = self.hparams.fn_num_clusters
            logger.info(
                f'Clustering {len(all_embeddings)} embeddings into {n_clusters} clusters...'
            )
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, verbose=0)
            labels = kmeans.fit_predict(all_embeddings)
            
            # Store pseudo-labels
            self.code_to_pseudo_label = {code: int(label) for code, label in zip(all_codes, labels)}
            logger.info(f'Pseudo-label map updated with {len(self.code_to_pseudo_label)} entries.')
        
        except Exception as e:
            logger.error(f'Failed to update pseudo-labels: {e}', exc_info=True)
        
        finally:
            self.train() # Set model back to train mode
    

    def on_validation_epoch_end(self):
        
        '''
        Compute evaluation metrics and trigger pseudo-label update based on the curriculum schedule.
        '''
        
        # Only evaluate every N epochs
        if self.current_epoch % self.hparams.eval_every_n_epochs != 0:
            return
        
        # Need cached embeddings and ground truth distances
        if not self.validation_embeddings or self.ground_truth_distances is None:
            logger.warning('Skipping evaluation: missing embeddings or ground truth distances')
            return
        
        try:
            logger.info(f'\nRunning evaluation metrics (epoch {self.current_epoch})...')
            
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
            
            # Number of samples for logging
            num_samples = len(embeddings)
            
            # 1. Embedding statistics
            stats = self.embedding_stats.compute_statistics(embeddings)
            self.log(
                'val/mean_norm', 
                self._to_python_scalar(stats['mean_norm']), 
                batch_size=num_samples
            )
            self.log(
                'val/std_norm',
                self._to_python_scalar(stats['std_norm']), 
                batch_size=num_samples
            )
            self.log(
                'val/mean_pairwise_distance', 
                self._to_python_scalar(stats['mean_pairwise_distance']), 
                batch_size=num_samples
            )
            self.log(
                'val/std_pairwise_distance', 
                self._to_python_scalar(stats['std_pairwise_distance']), 
                batch_size=num_samples
            )
            
            # 2. Improved collapse check with actual values
            collapse = self.embedding_stats.check_collapse(embeddings)
            self.log(
                'val/variance_collapsed', 
                self._to_python_scalar(collapse['variance_collapsed']), 
                batch_size=num_samples
            )
            self.log(
                'val/norm_collapsed', 
                self._to_python_scalar(collapse['norm_collapsed']), 
                batch_size=num_samples
            )
            self.log(
                'val/distance_collapsed', 
                self._to_python_scalar(collapse['distance_collapsed']), 
                batch_size=num_samples
            )
            # Log actual values for insight
            self.log(
                'val/mean_variance', 
                self._to_python_scalar(collapse['mean_variance']), 
                batch_size=num_samples
            )
            self.log(
                'val/norm_cv', 
                self._to_python_scalar(collapse['norm_cv']), 
                prog_bar=True, 
                batch_size=num_samples
            )
            self.log(
                'val/distance_cv', 
                self._to_python_scalar(collapse['distance_cv']), 
                prog_bar=True, 
                batch_size=num_samples
            )
            
            # 3. Compute embedding distances
            emb_dists = self.embedding_eval.compute_pairwise_distances(
                embeddings,
                metric='euclidean'
            )
            
            # 4. Improved hierarchy preservation metrics
            cophenetic_result = self.hierarchy_metrics.cophenetic_correlation(
                emb_dists,
                gt_dists
            )
            self.log(
                'val/cophenetic_correlation', 
                self._to_python_scalar(cophenetic_result['correlation']), 
                prog_bar=True, 
                batch_size=num_samples
            )
            self.log(
                'val/cophenetic_n_pairs', 
                self._to_python_scalar(cophenetic_result['n_pairs']), 
                batch_size=num_samples
            )

            spearman_result = self.hierarchy_metrics.spearman_correlation(
                emb_dists,
                gt_dists
            )
            self.log(
                'val/spearman_correlation', 
                self._to_python_scalar(spearman_result['correlation']),
                batch_size=num_samples
            )
            self.log(
                'val/spearman_n_pairs', 
                self._to_python_scalar(spearman_result['n_pairs']), 
                batch_size=num_samples
            )
            
            # 5. Distortion metrics
            distortion = self.hierarchy_metrics.distortion(emb_dists, gt_dists)
            self.log(
                'val/mean_distortion', 
                self._to_python_scalar(distortion['mean_distortion']), 
                batch_size=num_samples
            )
            self.log(
                'val/std_distortion', 
                self._to_python_scalar(distortion['std_distortion']), 
                batch_size=num_samples
            )
            self.log(
                'val/median_distortion', 
                self._to_python_scalar(distortion['median_distortion']), 
                prog_bar=True, 
                batch_size=num_samples
            )
            
            # Check if it's a clustering epoch
            if self.current_epoch >= self.hparams.fn_curriculum_start_epoch:
                
                # Check if clustering is enabled
                if self.hparams.fn_cluster_every_n_epochs > 0:
                    
                    epochs_since_start = (
                        self.current_epoch - 
                        self.hparams.fn_curriculum_start_epoch
                    )
                    
                    # Trigger on the first curriculum epoch, and every N epochs after
                    if epochs_since_start % self.hparams.fn_cluster_every_n_epochs == 0:
                        self._update_pseudo_labels()            
                        
            logger.info(
                f'Evaluation complete: '
                f'cophenetic={cophenetic_result["correlation"]:.4f} '
                f'({cophenetic_result["n_pairs"]} pairs), '
                f'spearman={spearman_result["correlation"]:.4f}, '
                f'norm_cv={collapse["norm_cv"]:.4f}, '
                f'dist_cv={collapse["distance_cv"]:.4f}, '
                f'collapse={collapse["any_collapse"]}\n'
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
        
        # Calculate total training steps safely
        # Use estimated_stepping_batches which handles all edge cases
        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = self.hparams.warmup_steps
        
        def lr_lambda(current_step: int):

            # Linear warmup
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            
            # Cosine decay after warmup
            progress_1 = float(current_step - warmup_steps)
            progress_2 = float(max(1, total_steps - warmup_steps))
            progress = progress_1 / progress_2
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1
            }
        }
