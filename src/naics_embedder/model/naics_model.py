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
from naics_embedder.model.hyperbolic import (
    log_hyperbolic_diagnostics,
    check_lorentz_manifold_validity
)

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
        num_experts: int = 4,
        top_k: int = 2,
        moe_hidden_dim: int = 1024,
        temperature: float = 0.07,
        curvature: float = 1.0,
        hierarchy_weight: float = 0.1,
        learning_rate: float = 2e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 500,
        load_balancing_coef: float = 0.01,
        fn_curriculum_start_epoch: int = 10,
        fn_cluster_every_n_epochs: int = 5,
        fn_num_clusters: int = 500,
        distance_matrix_path: Optional[str] = None,
        eval_every_n_epochs: int = 1,
        eval_sample_size: int = 500
    ):
        super().__init__()
        
        self.save_hyperparameters()
        
        self.encoder = MultiChannelEncoder(
            base_model_name=base_model_name,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            num_experts=num_experts,
            top_k=top_k,
            moe_hidden_dim=moe_hidden_dim,
            curvature=curvature  # Pass curvature to encoder for hyperbolic projection
        )
        
        self.loss_fn = HyperbolicInfoNCELoss(
            embedding_dim=self.encoder.embedding_dim,
            temperature=temperature,
            curvature=curvature
        )
        
        self.load_balancing_coef = load_balancing_coef
        
        self.embedding_eval = EmbeddingEvaluator()
        self.embedding_stats = EmbeddingStatistics()
        self.hierarchy_metrics = HierarchyMetrics()
        
        self.ground_truth_distances = None
        self.code_to_idx = None
        if distance_matrix_path:
            self._load_ground_truth_distances(distance_matrix_path)
        
        # Add hierarchy preservation loss if ground truth available
        self.hierarchy_loss_fn = None
        hierarchy_weight = getattr(self.hparams, 'hierarchy_weight', 0.1)
        if self.ground_truth_distances is not None and self.code_to_idx is not None and hierarchy_weight > 0:
            from naics_embedder.model.loss import HierarchyPreservationLoss
            self.hierarchy_loss_fn = HierarchyPreservationLoss(
                tree_distances=self.ground_truth_distances,
                code_to_idx=self.code_to_idx,
                weight=hierarchy_weight
            )
        
        self.validation_embeddings = {}
        self.validation_codes = []

        self.code_to_pseudo_label: Dict[str, int] = {}
    
    
    def _load_ground_truth_distances(self, distance_matrix_path: str):

        '''Load ground truth NAICS tree distances for evaluation.'''

        try:
            print(
                f'Loading ground truth distances\n'
                f'  • from: {distance_matrix_path}')
            
            df = pl.read_parquet(distance_matrix_path)
            n_codes = df.height
            
            ground_truth_distances = df.to_torch()
            print(f'  • distance matrix: [{n_codes}, {n_codes}]\n')
            
            code_to_idx = {}
            for col in df.columns:
                idx_col, code_col = col.split('-')
                idx = int(idx_col.replace('idx_', ''))
                code = code_col.replace('code_', '')
                code_to_idx[code] = idx
                
            self.ground_truth_distances = ground_truth_distances
            self.code_to_idx = code_to_idx
            
        except Exception as e:
            print(f'Could not load ground truth distances: {e}')
            ground_truth_distances = None
            code_to_idx = None
    
    
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
            'negative_codes' in batch 
        ):
            
            try:
                
                assert_conds = [
                    isinstance(batch['negative_codes'], list),
                    len(batch['negative_codes']) == batch_size,
                    all(isinstance(codes, list) for codes in batch['negative_codes'])
                ]
                assert_messages = [
                    'negative_codes must be a list',
                    f"Expected {batch_size} groups, got {len(batch['negative_codes'])}",
                    'Each entry must be a list of codes'                    
                ]
                
                for cond, msg in zip(assert_conds, assert_messages):
                    assert cond, msg
                
                anchor_labels = torch.tensor(
                    [self.code_to_pseudo_label.get(code, -1) for code in batch['anchor_code']],
                    device=self.device
                )

                neg_labels = torch.tensor(
                    [
                        [self.code_to_pseudo_label.get(code, -2) for code in neg_codes_for_anchor]
                        for neg_codes_for_anchor in batch['negative_codes']
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
            full_load_balancing_loss = torch.tensor(0.0, device=self.device)
            
        else:
            gate_probs = torch.cat(all_gate_probs, dim=0)
            top_k_indices = torch.cat(all_top_k_indices, dim=0)
            
            total_tokens = gate_probs.shape[0]
            num_experts = gate_probs.shape[1]

            P_micro = gate_probs.mean(dim=0)
            
            expert_counts_micro = torch.zeros(num_experts, device=self.device)
            for i in range(num_experts):
                expert_counts_micro[i] = (top_k_indices == i).any(dim=1).sum()
            
            f_micro = expert_counts_micro / total_tokens
            
            if torch.distributed.is_initialized():
                world_size = torch.distributed.get_world_size()
                
                if world_size > 1:
                    global_prob_sum = gate_probs.sum(dim=0)
                    global_expert_counts = expert_counts_micro * total_tokens
                    global_total_tokens = torch.tensor(
                        total_tokens, 
                        dtype=torch.float, 
                        device=self.device
                    )
                    
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

                    global_total_tokens_safe = torch.clamp(global_total_tokens, min=1.0)
                    
                    f = global_expert_counts / global_total_tokens_safe
                    P = global_prob_sum / global_total_tokens_safe
                    
                    if self.trainer.is_global_zero:
                        logger.debug(f'Global load balancing: f={f.mean():.4f}, P={P.mean():.4f}')
                        
                else:
                    f = f_micro
                    P = P_micro
            else:
                f = f_micro
                P = P_micro

            unscaled_loss = num_experts * torch.sum(f * P)
            full_load_balancing_loss = self.load_balancing_coef * unscaled_loss

            # Log MoE expert utilization metrics
            # f_i: fraction of tokens routed to each expert
            # P_i: average gating probability for each expert
            if self.trainer.is_global_zero or not torch.distributed.is_initialized():
                # Log per-expert utilization (f_i)
                for i in range(num_experts):
                    self.log(
                        f'train/moe/expert_{i}_utilization',
                        f[i].item(),
                        batch_size=batch_size,
                        on_step=False,  # Log per-epoch to reduce noise
                        on_epoch=True
                    )
                
                # Log per-expert gating probability (P_i)
                for i in range(num_experts):
                    self.log(
                        f'train/moe/expert_{i}_gating_prob',
                        P[i].item(),
                        batch_size=batch_size,
                        on_step=False,
                        on_epoch=True
                    )
                
                # Log histogram of expert utilization (f_i)
                # Only if TensorBoard logger is available
                if hasattr(self.logger, 'experiment') and hasattr(self.logger.experiment, 'add_histogram'):
                    try:
                        self.logger.experiment.add_histogram(
                            'train/moe/expert_utilization_hist',
                            f,
                            global_step=self.global_step
                        )
                        
                        # Log histogram of gating probabilities (P_i)
                        self.logger.experiment.add_histogram(
                            'train/moe/gating_prob_hist',
                            P,
                            global_step=self.global_step
                        )
                    except Exception as e:
                        logger.debug(f'Could not log histograms: {e}')
                
                # Log summary statistics for f_i
                f_mean = f.mean().item()
                f_std = f.std().item()
                f_min = f.min().item()
                f_max = f.max().item()
                f_cv = (f_std / f_mean) if f_mean > 0 else 0.0  # Coefficient of variation
                
                self.log(
                    'train/moe/utilization_mean',
                    f_mean,
                    batch_size=batch_size,
                    on_step=False,
                    on_epoch=True
                )
                self.log(
                    'train/moe/utilization_std',
                    f_std,
                    batch_size=batch_size,
                    on_step=False,
                    on_epoch=True
                )
                self.log(
                    'train/moe/utilization_cv',
                    f_cv,
                    batch_size=batch_size,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True  # Show in progress bar for quick monitoring
                )
                self.log(
                    'train/moe/utilization_min',
                    f_min,
                    batch_size=batch_size,
                    on_step=False,
                    on_epoch=True
                )
                self.log(
                    'train/moe/utilization_max',
                    f_max,
                    batch_size=batch_size,
                    on_step=False,
                    on_epoch=True
                )
                
                # Log summary statistics for P_i
                P_mean = P.mean().item()
                P_std = P.std().item()
                P_min = P.min().item()
                P_max = P.max().item()
                P_cv = (P_std / P_mean) if P_mean > 0 else 0.0
                
                self.log(
                    'train/moe/gating_prob_mean',
                    P_mean,
                    batch_size=batch_size,
                    on_step=False,
                    on_epoch=True
                )
                self.log(
                    'train/moe/gating_prob_std',
                    P_std,
                    batch_size=batch_size,
                    on_step=False,
                    on_epoch=True
                )
                self.log(
                    'train/moe/gating_prob_cv',
                    P_cv,
                    batch_size=batch_size,
                    on_step=False,
                    on_epoch=True
                )
                self.log(
                    'train/moe/gating_prob_min',
                    P_min,
                    batch_size=batch_size,
                    on_step=False,
                    on_epoch=True
                )
                self.log(
                    'train/moe/gating_prob_max',
                    P_max,
                    batch_size=batch_size,
                    on_step=False,
                    on_epoch=True
                )
                
                # Log load balancing imbalance (lower is better, 0 = perfectly balanced)
                # Ideal: f_i = 1/num_experts for all i, so std should be 0
                ideal_utilization = 1.0 / num_experts
                utilization_imbalance = torch.abs(f - ideal_utilization).mean().item()
                self.log(
                    'train/moe/utilization_imbalance',
                    utilization_imbalance,
                    batch_size=batch_size,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True
                )

        # Add hierarchy preservation loss if available
        hierarchy_loss = torch.tensor(0.0, device=self.device)
        if self.hierarchy_loss_fn is not None and 'anchor_code' in batch:
            try:
                # Get all codes and embeddings in batch
                all_codes = batch['anchor_code'].copy()
                if 'positive_code' in batch:
                    all_codes.extend(batch['positive_code'])
                else:
                    # If no positive codes, duplicate anchor codes
                    all_codes.extend(batch['anchor_code'])
                
                all_embeddings = torch.cat([anchor_emb, positive_emb])
                
                # Use the loss function's lorentz distance
                from naics_embedder.model.hyperbolic import LorentzDistance
                lorentz_dist = LorentzDistance(curvature=self.hparams.curvature)
                
                hierarchy_loss = self.hierarchy_loss_fn(
                    all_embeddings,
                    all_codes,
                    lambda x, y: lorentz_dist(x, y)
                )
                
                self.log('train/hierarchy_loss', hierarchy_loss, batch_size=batch_size)
            except Exception as e:
                logger.warning(f'Failed to compute hierarchy loss: {e}')

        total_loss = contrastive_loss + full_load_balancing_loss + hierarchy_loss

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
        if hierarchy_loss.item() > 0:
            self.log(
                'train/hierarchy_loss',
                hierarchy_loss,
                prog_bar=False,
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

        if 'anchor_code' in batch:
            for i, code in enumerate(batch['anchor_code']):
                if code not in self.validation_embeddings:
                    # Store hyperbolic embeddings (Lorentz model)
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
                batch = self.transfer_batch_to_device(batch, self.device, 0)
                
                with torch.no_grad():
                    anchor_output = self(batch['anchor'])
                    # Use Euclidean embeddings for clustering (KMeans works in Euclidean space)
                    # The encoder returns both 'embedding' (hyperbolic) and 'embedding_euc' (Euclidean)
                    if 'embedding_euc' in anchor_output and anchor_output['embedding_euc'] is not None:
                        embs = anchor_output['embedding_euc'].cpu()
                    else:
                        # Fallback: use hyperbolic embeddings (will project to Euclidean for KMeans)
                        # Extract spatial coordinates (drop time coordinate)
                        hyp_embs = anchor_output['embedding'].cpu()
                        embs = hyp_embs[:, 1:]  # Remove time coordinate, keep spatial coordinates
                    all_embeddings.append(embs)
                    all_codes.extend(batch['anchor_code'])
            
            all_embeddings = torch.cat(all_embeddings, dim=0).numpy()
            
            n_clusters = min((10 * (len(all_embeddings) // 20), self.hparams.fn_num_clusters))
                    
            logger.info(
                f'Clustering {len(all_embeddings)} embeddings into {n_clusters} clusters...'
            )
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, verbose=0)
            labels = kmeans.fit_predict(all_embeddings)
            
            self.code_to_pseudo_label = {code: int(label) for code, label in zip(all_codes, labels)}
            logger.info(f'Pseudo-label map updated with {len(self.code_to_pseudo_label)} entries.')
        
        except Exception as e:
            logger.error(f'Failed to update pseudo-labels: {e}', exc_info=True)
        
        finally:
            self.train()
    

    def on_validation_epoch_end(self):
        
        '''
        Compute evaluation metrics and trigger pseudo-label update based on the curriculum schedule.
        '''
        
        if self.current_epoch % self.hparams.eval_every_n_epochs != 0:
            return
        
        if not self.validation_embeddings or self.ground_truth_distances is None:
            logger.warning('Skipping evaluation: missing embeddings or ground truth distances')
            return
        
        try:
            logger.info(f'\nRunning evaluation metrics (epoch {self.current_epoch})...')
            
            codes = sorted(self.validation_embeddings.keys())
            embeddings = torch.stack([
                self.validation_embeddings[code] for code in codes
            ]).to(self.device)
            
            if len(codes) > self.hparams.eval_sample_size:
                indices = torch.randperm(len(codes))[:self.hparams.eval_sample_size]
                embeddings = embeddings[indices]
                codes = [codes[i] for i in indices]
            
            code_indices = [self.code_to_idx[code] for code in codes if code in self.code_to_idx]
            if len(code_indices) < 2:
                logger.warning('Not enough codes in ground truth for evaluation')
                return
            
            gt_dists = self.ground_truth_distances[code_indices][:, code_indices].to(self.device)
            
            num_samples = len(embeddings)
            
            # Check manifold validity and log diagnostics
            is_valid, lorentz_norms, violations = check_lorentz_manifold_validity(
                embeddings, 
                curvature=self.hparams.curvature
            )
            
            # Log hyperbolic diagnostics
            # Note: level_labels would require NAICS hierarchy level info - can be added later
            diagnostics = log_hyperbolic_diagnostics(
                embeddings,
                curvature=self.hparams.curvature,
                level_labels=None,  # TODO: Add NAICS level labels if available
                logger_instance=logger
            )
            
            # Log manifold validity metrics
            self.log(
                'val/manifold_valid',
                float(is_valid),
                batch_size=num_samples
            )
            self.log(
                'val/lorentz_norm_mean',
                diagnostics['lorentz_norm_mean'],
                batch_size=num_samples
            )
            self.log(
                'val/lorentz_norm_violation_max',
                diagnostics['violation_max'],
                batch_size=num_samples
            )
            self.log(
                'val/hyperbolic_radius_mean',
                diagnostics['radius_mean'],
                batch_size=num_samples
            )
            self.log(
                'val/hyperbolic_radius_std',
                diagnostics['radius_std'],
                batch_size=num_samples
            )
            
            # Warn if manifold constraint is violated
            if not is_valid:
                logger.warning(
                    f'⚠️  Hyperbolic embeddings violate manifold constraint! '
                    f'Max violation: {diagnostics["violation_max"]:.6e}'
                )
            
            # Compute statistics on Euclidean projection for compatibility
            # (embedding_stats expects Euclidean embeddings)
            # Use spatial coordinates (drop time coordinate) for Euclidean stats
            embeddings_euc = embeddings[:, 1:]  # Remove time coordinate
            stats = self.embedding_stats.compute_statistics(embeddings_euc)
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
            
            # Check collapse on Euclidean projection
            collapse = self.embedding_stats.check_collapse(embeddings_euc)
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
            
            # Use Lorentzian distances for hyperbolic embeddings
            emb_dists = self.embedding_eval.compute_pairwise_distances(
                embeddings,
                metric='lorentz',
                curvature=self.hparams.curvature
            )
            
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
            
            if self.current_epoch >= self.hparams.fn_curriculum_start_epoch:
                
                if self.hparams.fn_cluster_every_n_epochs > 0:
                    
                    epochs_since_start = (
                        self.current_epoch - 
                        self.hparams.fn_curriculum_start_epoch
                    )
                    
                    if epochs_since_start % self.hparams.fn_cluster_every_n_epochs == 0:
                        self._update_pseudo_labels()            
                    
            logger.info(
                f'Correlation metrics: \n'
                f'  • Hierarchy preservation: cophenetic={cophenetic_result["correlation"]:.4f} '
                f'({cophenetic_result["n_pairs"]} pairs)\n'
                f'  • Rank preservation: spearman={spearman_result["correlation"]:.4f}\n'
            )        
            logger.info(
                f'Collapse detection metrics: \n'
                f'  • Norm CV: {collapse["norm_cv"]:.4f}\n'
                f'  • Distance CV: {collapse["distance_cv"]:.4f}\n'
                f'  • Collapse: {collapse["any_collapse"]}\n'
            )
            
        except Exception as e:
            logger.error(f'Error during evaluation: {e}', exc_info=True)
        
        finally:
            self.validation_embeddings.clear()
            self.validation_codes.clear()
    

    def configure_optimizers(self):
        
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        
        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = self.hparams.warmup_steps
        
        def lr_lambda(current_step: int):

            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            
            progress_1 = float(current_step - warmup_steps)
            progress_2 = float(max(1, total_steps - warmup_steps))
            progress = progress_1 / progress_2
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        # Add ReduceLROnPlateau for validation-based learning rate reduction
        plateau_scheduler = {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.5,  # Reduce LR by 50%
                patience=3,  # Wait 3 epochs without improvement
                min_lr=1e-6,
                verbose=True
            ),
            'monitor': 'val/contrastive_loss',
            'interval': 'epoch',
            'frequency': 1
        }
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': [
                {
                    'scheduler': scheduler,
                    'interval': 'step',
                    'frequency': 1
                },
                plateau_scheduler
            ]
        }
