# -------------------------------------------------------------------------------------------------
# Imports and settings
# -------------------------------------------------------------------------------------------------

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import polars as pl
import pytorch_lightning as pyl
import torch
import torch.distributed as dist

from naics_embedder.text_model.curriculum import CurriculumScheduler
from naics_embedder.text_model.hyperbolic_clustering import HyperbolicKMeans
from naics_embedder.text_model.encoder import MultiChannelEncoder
from naics_embedder.text_model.evaluation import (
    EmbeddingEvaluator,
    EmbeddingStatistics,
    HierarchyMetrics,
)
from naics_embedder.text_model.hard_negative_mining import (
    LorentzianHardNegativeMiner,
    NormAdaptiveMargin,
    RouterGuidedNegativeMiner,
)
from naics_embedder.text_model.hyperbolic import (
    check_lorentz_manifold_validity,
    log_hyperbolic_diagnostics,
)
from naics_embedder.text_model.loss import HyperbolicInfoNCELoss

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------------------------------------
# Distributed Utilities for Global Batch Sampling
# -------------------------------------------------------------------------------------------------

def gather_embeddings_global(
    local_embeddings: torch.Tensor,
    world_size: Optional[int] = None
) -> torch.Tensor:
    '''
    Gather embeddings from all GPUs using all_gather with gradient support.
    
    Issue #19: Global Batch Sampling - Collect embeddings from all ranks
    to enable hard negative mining across the global batch.
    
    This function uses torch.distributed.all_gather which preserves gradients,
    ensuring that gradients flow back through the gather operation during backprop.
    
    Args:
        local_embeddings: Local embeddings tensor (N_local, D) with requires_grad=True
        world_size: Number of GPUs (auto-detected if None)
    
    Returns:
        Global embeddings tensor (N_global, D) where N_global = N_local * world_size
        Gradients will flow back through this operation during backprop.
    '''
    if not dist.is_initialized():
        # Single GPU case: return local embeddings as-is
        return local_embeddings
    
    if world_size is None:
        world_size = dist.get_world_size()
    
    if world_size == 1:
        return local_embeddings
    
    # Use torch.distributed.all_gather for gradient support
    # This preserves gradients: if local_embeddings requires grad, the gathered
    # tensors will also have gradients flowing back during backprop.
    gathered_list = [
        torch.zeros_like(local_embeddings) for _ in range(world_size)
    ]
    
    # all_gather collects tensors from all ranks into gathered_list
    # Each rank receives all tensors, so gathered_list[i] contains the tensor from rank i
    # Gradients flow back: during backprop, gradients are scattered back to each rank
    dist.all_gather(gathered_list, local_embeddings)
    
    # Concatenate all gathered embeddings along the batch dimension
    # This concatenation also preserves gradients
    global_embeddings = torch.cat(gathered_list, dim=0)
    
    return global_embeddings


def gather_negative_codes_global(
    local_negative_codes: List[List[str]],
    world_size: Optional[int] = None
) -> List[List[str]]:
    '''
    Gather negative codes from all GPUs for false negative masking.
    
    Args:
        local_negative_codes: Local negative codes (batch_size, k_negatives)
        world_size: Number of GPUs (auto-detected if None)
    
    Returns:
        Global negative codes list
    '''
    if not dist.is_initialized():
        return local_negative_codes
    
    if world_size is None:
        world_size = dist.get_world_size()
    
    if world_size == 1:
        return local_negative_codes
    
    # Gather negative codes from all ranks
    # Note: all_gather_object is used for Python objects like lists
    gathered_codes: List[List[str]] = [None] * world_size  # type: ignore
    dist.all_gather_object(gathered_codes, local_negative_codes)
    
    # Flatten the list of lists from all ranks
    global_negative_codes = []
    for codes_per_rank in gathered_codes:
        if codes_per_rank is not None:
            global_negative_codes.extend(codes_per_rank)
    
    return global_negative_codes


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
        rank_order_weight: float = 0.15,
        radius_reg_weight: float = 0.01,
        learning_rate: float = 2e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 500,
        use_warmup_cosine: bool = False,
        load_balancing_coef: float = 0.01,
        fn_curriculum_start_epoch: int = 10,
        fn_cluster_every_n_epochs: int = 5,
        fn_num_clusters: int = 500,
        distance_matrix_path: Optional[str] = None,
        eval_every_n_epochs: int = 1,
        eval_sample_size: int = 500,
        tree_distance_alpha: float = 1.5,
        base_margin: float = 0.5
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
        
        # Phase 2: Hard negative mining and norm-adaptive margin
        self.hard_negative_miner = LorentzianHardNegativeMiner(curvature=curvature)
        self.norm_adaptive_margin = NormAdaptiveMargin(
            base_margin=base_margin,
            curvature=curvature
        )
        
        # Router-guided negative mining for MoE (Issue #16)
        self.router_guided_miner = RouterGuidedNegativeMiner(
            metric='kl_divergence',  # Use KL-divergence to measure confusion
            temperature=1.0
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
            from naics_embedder.text_model.loss import HierarchyPreservationLoss
            self.hierarchy_loss_fn = HierarchyPreservationLoss(
                tree_distances=self.ground_truth_distances,
                code_to_idx=self.code_to_idx,
                weight=hierarchy_weight
            )
        
        # Add LambdaRank loss for global ranking optimization (replaces pairwise ranking)
        self.lambdarank_loss_fn = None
        rank_order_weight = getattr(self.hparams, 'rank_order_weight', 0.15)
        if self.ground_truth_distances is not None and self.code_to_idx is not None and rank_order_weight > 0:
            from naics_embedder.text_model.loss import LambdaRankLoss
            self.lambdarank_loss_fn = LambdaRankLoss(
                tree_distances=self.ground_truth_distances,
                code_to_idx=self.code_to_idx,
                weight=rank_order_weight,
                sigma=1.0,
                ndcg_k=10
            )
        
        self.validation_embeddings = {}
        self.validation_codes = []

        self.code_to_pseudo_label: Dict[str, int] = {}
        
        # Initialize evaluation metrics history for JSON logging
        self.evaluation_metrics_history: List[Dict] = []
        
        # Initialize curriculum scheduler (will be set in on_train_start)
        self.curriculum_scheduler: Optional[CurriculumScheduler] = None
        self.current_curriculum_flags: Dict[str, bool] = {}
        self.previous_phase: Optional[int] = None
    
    
    def _get_metrics_file_path(self) -> Optional[Path]:
        '''Get the path to save evaluation metrics JSON file.'''
        if self.logger is None:
            return None
        
        # Try to get log directory from logger
        if hasattr(self.logger, 'log_dir') and self.logger.log_dir:
            return Path(self.logger.log_dir) / 'evaluation_metrics.json'
        elif hasattr(self.logger, 'save_dir'):
            # Fallback for TensorBoardLogger
            save_dir = getattr(self.logger, 'save_dir', None)
            if save_dir:
                version = getattr(self.logger, 'version', 0)
                name = getattr(self.logger, 'name', 'default')
                return Path(save_dir) / name / f'version_{version}' / 'evaluation_metrics.json'
        
        return None
    
    
    def _load_ground_truth_distances(self, distance_matrix_path: str):
        
        '''
        Load ground truth NAICS tree distances for evaluation.
        '''
        
        try:
            logger.info(
                f'Loading ground truth distances from: {distance_matrix_path}'
            )
            
            df = pl.read_parquet(distance_matrix_path)
            n_codes = df.height
            
            ground_truth_distances = df.to_torch()
            logger.info(f'Distance matrix shape: [{n_codes}, {n_codes}]')
            
            code_to_idx = {}
            for col in df.columns:
                idx_col, code_col = col.split('-')
                idx = int(idx_col.replace('idx_', ''))
                code = code_col.replace('code_', '')
                code_to_idx[code] = idx
                
            self.ground_truth_distances = ground_truth_distances
            self.code_to_idx = code_to_idx
            
        except Exception as e:
            logger.error(f'Could not load ground truth distances: {e}')
            ground_truth_distances = None
            code_to_idx = None
    
    
    def forward(
        self, 
        channel_inputs: Dict[str, Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        
        return self.encoder(channel_inputs)
    
    def _compute_contrastive_loss(
        self,
        anchor_emb: torch.Tensor,
        positive_emb: torch.Tensor,
        negative_emb: torch.Tensor,
        batch_size: int,
        k_negatives: int,
        batch: Dict,
        batch_indices: List[int],
        batch_idx: int,
        anchor_gate_probs: Optional[torch.Tensor] = None,
        negative_gate_probs: Optional[torch.Tensor] = None,
        false_negative_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        '''
        Compute contrastive loss with curriculum features (hard negative mining, etc.).
        
        This is a helper method used by both single-level and multi-level supervision paths.
        '''
        # Phase 2: Hard Negative Mining (HNM) - Embedding-based and Router-guided
        enable_hard_negative_mining = self.current_curriculum_flags.get(
            'enable_hard_negative_mining', False
        )
        enable_router_guided_sampling = self.current_curriculum_flags.get(
            'enable_router_guided_sampling', False
        )
        
        # Reshape negatives for hard negative mining
        negative_emb_reshaped = negative_emb.view(batch_size, k_negatives, -1)
        
        if enable_hard_negative_mining or enable_router_guided_sampling:
            candidate_negatives = negative_emb_reshaped.clone()
            
            if enable_hard_negative_mining:
                hard_negatives, hard_neg_distances = self.hard_negative_miner.mine_hard_negatives(
                    anchor_emb=anchor_emb,
                    candidate_negatives=candidate_negatives,
                    k=k_negatives,
                    return_distances=True
                )
                
                if batch_idx == 0 and hard_neg_distances is not None:
                    self.log(
                        'train/curriculum/hard_neg_avg_distance',
                        hard_neg_distances.mean().item(),
                        batch_size=batch_size,
                        on_step=False,
                        on_epoch=True
                    )
            
            if enable_router_guided_sampling:
                router_confusion_scores = None
                if (
                    anchor_gate_probs is not None and
                    negative_gate_probs is not None
                ):
                    negative_gate_probs_reshaped = negative_gate_probs.view(
                        batch_size, k_negatives, -1
                    )

                    router_hard_negatives, router_confusion_scores = self.router_guided_miner.mine_router_hard_negatives(
                        anchor_gate_probs=anchor_gate_probs,
                        negative_gate_probs=negative_gate_probs_reshaped,
                        candidate_negatives=candidate_negatives,
                        k=k_negatives,
                        return_scores=True
                    )

                    router_mix_ratio = 0.5
                    if enable_hard_negative_mining:
                        n_router = int(k_negatives * router_mix_ratio)
                        n_embedding = k_negatives - n_router
                        router_selected = router_hard_negatives[:, :n_router, :]
                        embedding_selected = hard_negatives[:, :n_embedding, :]
                        negative_emb_reshaped = torch.cat(
                            [router_selected, embedding_selected],
                            dim=1
                        )
                    else:
                        negative_emb_reshaped = router_hard_negatives

                    if batch_idx == 0 and router_confusion_scores is not None:
                        avg_confusion = router_confusion_scores.mean().item()
                        min_confusion = router_confusion_scores.min().item()
                        max_confusion = router_confusion_scores.max().item()
                        self.log(
                            'train/curriculum/router_confusion_mean',
                            avg_confusion,
                            batch_size=batch_size,
                            on_step=False,
                            on_epoch=True
                        )
                        self.log(
                            'train/curriculum/router_confusion_avg',
                            avg_confusion,
                            batch_size=batch_size,
                            on_step=False,
                            on_epoch=True
                        )
                        self.log(
                            'train/curriculum/router_confusion_min',
                            min_confusion,
                            batch_size=batch_size,
                            on_step=False,
                            on_epoch=True
                        )
                        self.log(
                            'train/curriculum/router_confusion_max',
                            max_confusion,
                            batch_size=batch_size,
                            on_step=False,
                            on_epoch=True
                        )
                elif enable_hard_negative_mining:
                    negative_emb_reshaped = hard_negatives
                    logger.debug(
                        'Router-guided sampling skipped: gate probabilities not available'
                    )
                else:
                    logger.debug(
                        'Router-guided sampling skipped: gate probabilities not available'
                    )
            elif enable_hard_negative_mining:
                negative_emb_reshaped = hard_negatives
            
            negative_emb = negative_emb_reshaped.view(batch_size * k_negatives, -1)
        
        # Compute contrastive loss
        loss = self.loss_fn(
            anchor_emb,
            positive_emb,
            negative_emb,
            batch_size,
            k_negatives,
            false_negative_mask=false_negative_mask,
            adaptive_margins=adaptive_margins
        )
        
        return loss

    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        
        # Update curriculum flags based on current epoch
        if self.curriculum_scheduler is not None:
            self.current_curriculum_flags = self.curriculum_scheduler.get_curriculum_flags(self.current_epoch)
            
            # Log phase transitions
            self.curriculum_scheduler.log_phase_transition(
                self.current_epoch,
                self.previous_phase
            )
            self.previous_phase = self.curriculum_scheduler.get_phase(self.current_epoch)
        
        # Check if we have multi-level supervision (Issue #18)
        # With multi-level supervision, the batch is expanded so each positive level
        # is a separate item. The loss will naturally sum over all positive levels,
        # giving us gradient accumulation.
        has_multilevel = 'positive_levels' in batch
        
        anchor_output = self(batch['anchor'])
        positive_output = self(batch['positive'])
        negative_output = self(batch['negatives'])
        
        anchor_emb = anchor_output['embedding']
        positive_emb = positive_output['embedding']
        negative_emb = negative_output['embedding']
        
        batch_size = batch['batch_size']
        k_negatives = batch['k_negatives']
        
        # Log multi-level supervision statistics
        if has_multilevel and batch_idx == 0:
            positive_levels = batch['positive_levels']
            level_counts = {}
            for level in positive_levels:
                level_counts[level] = level_counts.get(level, 0) + 1
            for level, count in sorted(level_counts.items()):
                self.log(
                    f'train/multilevel/positive_level_{level}_count',
                    count,
                    batch_size=batch_size,
                    on_step=False,
                    on_epoch=True
                )
        
        # False negative masking: controlled by curriculum scheduler
        false_negative_mask = None
        enable_clustering = self.current_curriculum_flags.get('enable_clustering', False)
        if (
            enable_clustering and
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
        
        # Log negative sample type distribution (Issue #12)
        if batch_idx == 0 and 'negative_codes' in batch and self.code_to_idx is not None:
            self._log_negative_sample_distribution(batch)
            self._log_negative_tree_distance_distribution(batch)
        
        # Phase 2: Hard Negative Mining (HNM) - Embedding-based and Router-guided
        # When enabled, select top-k hardest negatives based on Lorentzian distance and/or router confusion
        enable_hard_negative_mining = self.current_curriculum_flags.get(
            'enable_hard_negative_mining', False
        )
        enable_router_guided_sampling = self.current_curriculum_flags.get(
            'enable_router_guided_sampling', False
        )
        
        # Issue #19: Global Batch Sampling - Gather negatives from all GPUs
        # This enables hard negative mining across the global batch, which is crucial
        # for finding meaningful "Cousin" negatives that may not appear in small local batches
        use_global_batch = (
            (enable_hard_negative_mining or enable_router_guided_sampling) and
            torch.distributed.is_initialized() and
            torch.distributed.get_world_size() > 1
        )
        
        if use_global_batch:
            # Gather global negatives from all GPUs
            # negative_emb: (batch_size * k_negatives, embedding_dim+1)
            global_negative_emb = gather_embeddings_global(negative_emb)
            
            # Get global batch size and total negatives
            world_size = torch.distributed.get_world_size()
            global_batch_size = batch_size * world_size
            global_k_negatives = global_negative_emb.shape[0] // global_batch_size
            
            # Reshape global negatives: (global_batch_size * global_k_negatives, embedding_dim+1)
            # -> (global_batch_size, global_k_negatives, embedding_dim+1)
            global_negative_emb_reshaped = global_negative_emb.view(
                global_batch_size, global_k_negatives, -1
            )
            
            # Extract global negatives for all anchors (we'll use all of them for mining)
            # But we only need to compute distances for local anchors
            candidate_negatives_global = global_negative_emb_reshaped  # (global_batch_size, global_k_negatives, embedding_dim+1)
            
            # Memory Management (Issue #19):
            # The global batch creates a similarity matrix of size:
            # (batch_size, global_batch_size * global_k_negatives)
            # Memory usage: batch_size * global_batch_size * global_k_negatives * 4 bytes (float32)
            # Example: batch_size=32, world_size=4, k_negatives=24
            #   -> global_batch_size=128, global_k_negatives=24
            #   -> Matrix size: (32, 3072) = ~393KB per batch
            #   -> Global negatives: (128*24, D+1) = ~(3072, 769) = ~9MB per GPU
            # This is manageable for most GPUs, but should be monitored.
            
            # Log memory usage for monitoring
            if batch_idx == 0:
                global_negatives_memory_mb = (
                    global_negative_emb.numel() * global_negative_emb.element_size() / (1024 ** 2)
                )
                # Estimate similarity matrix memory
                similarity_matrix_memory_mb = (
                    batch_size * global_batch_size * global_k_negatives * 4 / (1024 ** 2)
                )
                self.log(
                    'train/global_batch/global_negatives_memory_mb',
                    global_negatives_memory_mb,
                    batch_size=batch_size,
                    on_step=False,
                    on_epoch=True
                )
                self.log(
                    'train/global_batch/similarity_matrix_memory_mb',
                    similarity_matrix_memory_mb,
                    batch_size=batch_size,
                    on_step=False,
                    on_epoch=True
                )
                self.log(
                    'train/global_batch/global_batch_size',
                    global_batch_size,
                    batch_size=batch_size,
                    on_step=False,
                    on_epoch=True
                )
                self.log(
                    'train/global_batch/global_k_negatives',
                    global_k_negatives,
                    batch_size=batch_size,
                    on_step=False,
                    on_epoch=True
                )
        else:
            # Single GPU or global batch disabled: use local negatives
            global_negative_emb_reshaped = None
            candidate_negatives_global = None
            global_batch_size = batch_size
            global_k_negatives = k_negatives
        
        # Reshape local negatives for hard negative mining
        # negative_emb: (batch_size * k_negatives, embedding_dim+1)
        negative_emb_reshaped = negative_emb.view(batch_size, k_negatives, -1)
        
        if enable_hard_negative_mining or enable_router_guided_sampling:
            # Use global negatives if available, otherwise use local
            if use_global_batch and candidate_negatives_global is not None:
                # For each local anchor, mine hard negatives from global pool
                # We need to compute distances from each local anchor to all global negatives
                # This creates a (batch_size, global_k_negatives) distance matrix
                
                # Expand local anchors for broadcasting: (batch_size, 1, embedding_dim+1)
                anchor_emb_expanded = anchor_emb.unsqueeze(1)  # (batch_size, 1, embedding_dim+1)
                
                # Compute distances from each local anchor to all global negatives
                # Reshape global negatives to (global_batch_size * global_k_negatives, embedding_dim+1)
                global_negatives_flat = candidate_negatives_global.view(-1, anchor_emb.shape[-1])
                
                # Compute distances using batched forward
                # anchor_emb: (batch_size, embedding_dim+1)
                # global_negatives_flat: (global_batch_size * global_k_negatives, embedding_dim+1)
                # Result: (batch_size, global_batch_size * global_k_negatives)
                # 
                # Gradient Flow: The gathered global_negatives_flat has gradients from all GPUs.
                # When we compute distances and select hard negatives, gradients will flow back
                # through the all_gather operation to update embeddings on all GPUs.
                global_distances_flat = self.hard_negative_miner.lorentz_distance.batched_forward(
                    anchor_emb,  # (batch_size, embedding_dim+1)
                    global_negatives_flat.unsqueeze(0).expand(batch_size, -1, -1)  # (batch_size, global_total, embedding_dim+1)
                )  # (batch_size, global_total) where global_total = global_batch_size * global_k_negatives
                
                # Select top-k hardest (smallest distances)
                _, topk_indices = torch.topk(
                    global_distances_flat,
                    k=k_negatives,
                    dim=1,
                    largest=False  # Smallest distances = hardest negatives
                )
                
                # Gather selected negatives
                # topk_indices: (batch_size, k_negatives) indexing into flattened global negatives
                batch_indices = torch.arange(
                    batch_size,
                    device=anchor_emb.device
                ).unsqueeze(1).expand(-1, k_negatives)
                
                # Get selected negatives from flattened global pool
                selected_negatives = global_negatives_flat[topk_indices]  # (batch_size, k_negatives, embedding_dim+1)
                
                # Use selected negatives as hard negatives
                hard_negatives = selected_negatives
                hard_neg_distances = global_distances_flat.gather(1, topk_indices)
                
                # Log statistics
                if batch_idx == 0 and hard_neg_distances is not None:
                    avg_hard_neg_dist = hard_neg_distances.mean().item()
                    min_hard_neg_dist = hard_neg_distances.min().item()
                    max_hard_neg_dist = hard_neg_distances.max().item()
                    self.log(
                        'train/curriculum/hard_neg_avg_distance',
                        avg_hard_neg_dist,
                        batch_size=batch_size,
                        on_step=False,
                        on_epoch=True
                    )
                    self.log(
                        'train/curriculum/hard_neg_min_distance',
                        min_hard_neg_dist,
                        batch_size=batch_size,
                        on_step=False,
                        on_epoch=True
                    )
                    self.log(
                        'train/curriculum/hard_neg_max_distance',
                        max_hard_neg_dist,
                        batch_size=batch_size,
                        on_step=False,
                        on_epoch=True
                    )
                    self.log(
                        'train/global_batch/global_hard_negatives_used',
                        True,
                        batch_size=batch_size,
                        on_step=False,
                        on_epoch=True
                    )
            else:
                # Local hard negative mining (original behavior)
                # Store original candidate negatives
                candidate_negatives = negative_emb_reshaped.clone()
                
                # Embedding-based hard negative mining (Issue #15)
                if enable_hard_negative_mining:
                    # Mine top-k hardest negatives for each anchor based on Lorentzian distance
                    hard_negatives, hard_neg_distances = self.hard_negative_miner.mine_hard_negatives(
                        anchor_emb=anchor_emb,
                        candidate_negatives=candidate_negatives,
                        k=k_negatives,
                        return_distances=True
                    )
                
            # Router-guided negative mining (Issue #16)
            # Note: Router-guided sampling with global batch requires gathering gate_probs
            # For now, we'll use local gate_probs and global negatives (hybrid approach)
            if enable_router_guided_sampling:
                # Get gate probabilities for anchors and negatives
                anchor_gate_probs = anchor_output.get('gate_probs')
                negative_gate_probs = negative_output.get('gate_probs')
                
                if anchor_gate_probs is not None and negative_gate_probs is not None:
                    if use_global_batch:
                        # Gather global negative gate probabilities to align with global negatives
                        global_negative_gate_probs = gather_embeddings_global(negative_gate_probs)
                        global_negative_gate_probs_reshaped = global_negative_gate_probs.view(
                            global_batch_size, global_k_negatives, -1
                        )
                        # Flatten to (global_total, num_experts) then broadcast across anchors
                        global_neg_gate_probs_flat = global_negative_gate_probs_reshaped.view(
                            -1, anchor_gate_probs.shape[-1]
                        )
                        negative_gate_probs_expanded = (
                            global_neg_gate_probs_flat
                            .unsqueeze(0)
                            .expand(batch_size, -1, -1)
                        )

                        confusion_scores_global = self.router_guided_miner.compute_confusion_scores(
                            anchor_gate_probs,
                            negative_gate_probs_expanded
                        )  # (batch_size, global_total)

                        _, router_topk_indices = torch.topk(
                            confusion_scores_global,
                            k=k_negatives,
                            dim=1,
                            largest=True
                        )

                        router_hard_negatives = global_negatives_flat[router_topk_indices]
                        router_confusion_scores = confusion_scores_global.gather(
                            1,
                            router_topk_indices
                        )
                    else:
                        # Local router-guided sampling (original behavior)
                        # Reshape negative gate probs: (batch_size * k_negatives, num_experts) -> (batch_size, k_negatives, num_experts)
                        negative_gate_probs_reshaped = negative_gate_probs.view(batch_size, k_negatives, -1)
                        
                        # Mine router-hard negatives (negatives that confuse the gating network)
                        router_hard_negatives, router_confusion_scores = self.router_guided_miner.mine_router_hard_negatives(
                            anchor_gate_probs=anchor_gate_probs,
                            negative_gate_probs=negative_gate_probs_reshaped,
                            candidate_negatives=negative_emb_reshaped,
                            k=k_negatives,
                            return_scores=True
                        )
                    
                    # Mix router-hard negatives with embedding-hard negatives
                    # Strategy: Take 50% from each (or use a configurable ratio)
                    router_mix_ratio = 0.5  # 50% router-hard, 50% embedding-hard
                    
                    if enable_hard_negative_mining:
                        # Mix both types of hard negatives
                        n_router = int(k_negatives * router_mix_ratio)
                        n_embedding = k_negatives - n_router
                        
                        # Select top router-hard and top embedding-hard
                        router_selected = router_hard_negatives[:, :n_router, :]
                        embedding_selected = hard_negatives[:, :n_embedding, :]
                        
                        # Concatenate: router-hard first, then embedding-hard
                        mixed_negatives = torch.cat([router_selected, embedding_selected], dim=1)
                    else:
                        # Only router-hard negatives
                        mixed_negatives = router_hard_negatives
                    
                    # Update negative_emb with mixed negatives
                    negative_emb_reshaped = mixed_negatives
                    
                    # Log router-guided sampling statistics
                    if batch_idx == 0 and router_confusion_scores is not None:
                        avg_confusion = router_confusion_scores.mean().item()
                        min_confusion = router_confusion_scores.min().item()
                        max_confusion = router_confusion_scores.max().item()
                        self.log(
                            'train/curriculum/router_confusion_mean',
                            avg_confusion,
                            batch_size=batch_size,
                            on_step=False,
                            on_epoch=True
                        )
                        self.log(
                            'train/curriculum/router_confusion_min',
                            min_confusion,
                            batch_size=batch_size,
                            on_step=False,
                            on_epoch=True
                        )
                        self.log(
                            'train/curriculum/router_confusion_max',
                            max_confusion,
                            batch_size=batch_size,
                            on_step=False,
                            on_epoch=True
                        )
                else:
                    # Fallback to embedding-based if gate probs not available
                    if enable_hard_negative_mining:
                        negative_emb_reshaped = hard_negatives
                    logger.debug('Router-guided sampling skipped: gate probabilities not available')
            elif enable_hard_negative_mining:
                # Only embedding-based hard negative mining
                negative_emb_reshaped = hard_negatives
            
            # Flatten back to (batch_size * k_negatives, embedding_dim+1)
            negative_emb = negative_emb_reshaped.view(batch_size * k_negatives, -1)
        
        adaptive_margins = self.norm_adaptive_margin(anchor_emb)
        if batch_idx == 0:
            self.log(
                'train/curriculum/adaptive_margin_mean',
                adaptive_margins.mean().item(),
                batch_size=batch_size,
                on_step=False,
                on_epoch=True
            )
            self.log(
                'train/curriculum/adaptive_margin_min',
                adaptive_margins.min().item(),
                batch_size=batch_size,
                on_step=False,
                on_epoch=True
            )
            self.log(
                'train/curriculum/adaptive_margin_max',
                adaptive_margins.max().item(),
                batch_size=batch_size,
                on_step=False,
                on_epoch=True
            )
        
        contrastive_loss = self.loss_fn(
            anchor_emb,
            positive_emb,
            negative_emb,
            batch_size,
            k_negatives,
            false_negative_mask=false_negative_mask
        )
        
        # Router statistics logging: Monitor expert utilization patterns
        all_gate_probs = []
        all_top_k_indices = []
        for output in (anchor_output, positive_output, negative_output):
            if output['gate_probs'] is not None:
                all_gate_probs.append(output['gate_probs'])
                all_top_k_indices.append(output['top_k_indices'])
        
        # Log router-guided sampling metrics if enabled
        if self.current_curriculum_flags.get('enable_router_guided_sampling', False):
            if all_gate_probs:
                # Compute average expert selection diversity using entropy
                gate_probs_combined = torch.cat(all_gate_probs, dim=0)
                # Entropy: -sum(p * log(p)) for each row
                log_probs = torch.log(gate_probs_combined + 1e-8)  # Add small epsilon for numerical stability
                entropy_per_token = -(gate_probs_combined * log_probs).sum(dim=1)
                expert_diversity = entropy_per_token.mean()
                self.log(
                    'train/curriculum/router_expert_diversity',
                    expert_diversity.item(),
                    batch_size=batch_size,
                    on_step=False,
                    on_epoch=True
                )
        
        if not all_gate_probs:
            full_load_balancing_loss = torch.tensor(0.0, device=self.device)
            
        else:
            gate_probs = torch.cat(all_gate_probs, dim=0)
            top_k_indices = torch.cat(all_top_k_indices, dim=0)
            
            total_tokens = gate_probs.shape[0]
            num_experts = gate_probs.shape[1]

            # Compute sums (not means) - divide by total_tokens only after all-reduce in distributed mode
            prob_sum = gate_probs.sum(dim=0)
            
            expert_counts_micro = torch.zeros(num_experts, device=self.device)
            for i in range(num_experts):
                expert_counts_micro[i] = (top_k_indices == i).any(dim=1).sum()
            
            if torch.distributed.is_initialized():
                world_size = torch.distributed.get_world_size()
                
                if world_size > 1:
                    global_prob_sum = prob_sum.clone()
                    global_expert_counts = expert_counts_micro.clone()
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
                    # Single GPU case: divide by local total_tokens
                    f = expert_counts_micro / total_tokens
                    P = prob_sum / total_tokens
            else:
                # Non-distributed case: divide by local total_tokens
                f = expert_counts_micro / total_tokens
                P = prob_sum / total_tokens

            unscaled_loss = num_experts * torch.sum(f * P)
            full_load_balancing_loss = self.load_balancing_coef * unscaled_loss

            # Log MoE expert utilization metrics
            # f_i: fraction of tokens routed to each expert
            # P_i: average gating probability for each expert
            # Issue #6: Fixed distributed training check
            if not torch.distributed.is_initialized() or self.trainer.is_global_zero:
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
                from naics_embedder.text_model.hyperbolic import LorentzDistance
                lorentz_dist = LorentzDistance(curvature=self.hparams.curvature)
                
                hierarchy_loss = self.hierarchy_loss_fn(
                    all_embeddings,
                    all_codes,
                    lambda x, y: lorentz_dist(x, y)
                )
                
                self.log('train/hierarchy_loss', hierarchy_loss, batch_size=batch_size)
            except Exception as e:
                logger.warning(f'Failed to compute hierarchy loss: {e}')

        # Add LambdaRank loss for global ranking (1 positive + k negatives per anchor)
        lambdarank_loss = torch.tensor(0.0, device=self.device)
        if self.lambdarank_loss_fn is not None and 'anchor_code' in batch and 'positive_code' in batch:
            try:
                # Get codes from batch
                anchor_codes = batch['anchor_code']
                positive_codes = batch['positive_code']
                negative_codes = batch.get('negative_codes', [])
                
                # If negative_codes not in batch, we can't compute LambdaRank
                if not negative_codes or len(negative_codes) != batch_size:
                    logger.debug('Skipping LambdaRank: negative_codes not available in batch')
                else:
                    # Use the loss function's lorentz distance
                    from naics_embedder.text_model.hyperbolic import LorentzDistance
                    lorentz_dist = LorentzDistance(curvature=self.hparams.curvature)
                    
                    lambdarank_loss = self.lambdarank_loss_fn(
                        anchor_emb,
                        positive_emb,
                        negative_emb,
                        anchor_codes,
                        positive_codes,
                        negative_codes,
                        lambda x, y: lorentz_dist(x, y),
                        batch_size,
                        k_negatives
                    )
                    
                    self.log('train/lambdarank_loss', lambdarank_loss, batch_size=batch_size)
            except Exception as e:
                logger.warning(f'Failed to compute LambdaRank loss: {e}')
                import traceback
                logger.debug(traceback.format_exc())

        # Add radius regularization to prevent hyperbolic radius instability
        radius_reg_loss = torch.tensor(0.0, device=self.device)
        radius_reg_weight = getattr(self.hparams, 'radius_reg_weight', 0.0)
        if radius_reg_weight > 0:
            # Compute hyperbolic radius for all embeddings in batch
            # Radius = sqrt(x0^2 - 1) where x0 is the time component
            # We want to penalize large radii to keep embeddings near origin
            all_embeddings = torch.cat([anchor_emb, positive_emb, negative_emb])
            x0 = all_embeddings[:, 0]  # Time component
            # Compute radius: r = sqrt(x0^2 - 1/c) for curvature c
            # For c=1: r = sqrt(x0^2 - 1)
            c = self.hparams.curvature
            radius_squared = torch.clamp(x0 ** 2 - 1.0 / c, min=0.0)
            radius = torch.sqrt(radius_squared + 1e-8)  # Add small epsilon for stability
            
            # Penalize radii larger than a threshold (e.g., 10)
            # Use a smooth penalty: max(0, radius - threshold)^2
            radius_threshold = 10.0
            excess_radius = torch.clamp(radius - radius_threshold, min=0.0)
            radius_reg_loss = radius_reg_weight * torch.mean(excess_radius ** 2)
            
            self.log(
                'train/radius_reg_loss',
                radius_reg_loss,
                prog_bar=False,
                batch_size=batch_size
            )
            # Also log mean radius for monitoring
            self.log(
                'train/mean_radius',
                radius.mean(),
                prog_bar=False,
                batch_size=batch_size
            )
            self.log(
                'train/max_radius',
                radius.max(),
                prog_bar=False,
                batch_size=batch_size
            )

        batch_size = batch['batch_size']
        
        total_loss = contrastive_loss + full_load_balancing_loss + hierarchy_loss + lambdarank_loss + radius_reg_loss
        
        # Note: DCL loss may yield negative values (unlike InfoNCE which is always positive)
        # This is expected behavior for DCL: loss = (-pos_sim + logsumexp(neg_sims)).mean()
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
        if lambdarank_loss.item() > 0:
            self.log(
                'train/lambdarank_loss',
                lambdarank_loss,
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
    
    def _log_negative_sample_distribution(self, batch: Dict):
        '''
        Log distribution of negative sample types (child/sibling/cousin/distant).
        
        Issue #12: Track negative sample type distribution per curriculum phase.
        '''
        if self.curriculum_scheduler is None:
            return
        
        try:
            from naics_embedder.utils.utilities import get_relationship
            
            anchor_codes = batch['anchor_code']
            negative_codes = batch['negative_codes']
            
            # Classify negative samples by relationship type
            sample_types = {'child': 0, 'sibling': 0, 'cousin': 0, 'distant': 0, 'unknown': 0}
            total_samples = 0
            
            for anchor_code, neg_codes in zip(anchor_codes, negative_codes):
                for neg_code in neg_codes:
                    try:
                        relation = get_relationship(anchor_code, neg_code)
                        
                        # Classify into categories
                        if relation in ['child', 'grandchild', 'great-grandchild', 'great-great-grandchild']:
                            sample_types['child'] += 1
                        elif relation == 'sibling':
                            sample_types['sibling'] += 1
                        elif relation in ['cousin', 'nephew/niece', 'grand-nephew/niece', 
                                         'cousin_1_times_removed', 'second_cousin']:
                            sample_types['cousin'] += 1
                        elif relation in ['unrelated'] or relation.startswith('third_cousin') or relation.startswith('cousin_'):
                            sample_types['distant'] += 1
                        else:
                            sample_types['unknown'] += 1
                        
                        total_samples += 1
                    except Exception:
                        sample_types['unknown'] += 1
                        total_samples += 1
            
            if total_samples > 0:
                phase = self.curriculum_scheduler.get_phase(self.current_epoch)
                
                # Log to TensorBoard
                for sample_type, count in sample_types.items():
                    self.log(
                        f'train/curriculum/negative_samples_{sample_type}',
                        count / total_samples,
                        batch_size=len(anchor_codes),
                        on_step=False,
                        on_epoch=True
                    )
                
                # Log summary
                if self.current_epoch % 5 == 0:  # Log every 5 epochs to reduce noise
                    logger.info(
                        f'Negative sample distribution (Phase {phase}, Epoch {self.current_epoch}):\n'
                        f'  • Child: {sample_types["child"]/total_samples*100:.1f}%\n'
                        f'  • Sibling: {sample_types["sibling"]/total_samples*100:.1f}%\n'
                        f'  • Cousin: {sample_types["cousin"]/total_samples*100:.1f}%\n'
                        f'  • Distant: {sample_types["distant"]/total_samples*100:.1f}%\n'
                        f'  • Unknown: {sample_types["unknown"]/total_samples*100:.1f}%'
                    )
        
        except Exception as e:
            logger.debug(f'Failed to log negative sample distribution: {e}')
    
    def _log_negative_tree_distance_distribution(self, batch: Dict):
        '''
        Log distribution of negative samples by tree distance bins.
        
        Issue #23: Track tree-distance categories to verify Phase 1 weighting.
        '''
        if self.ground_truth_distances is None or self.code_to_idx is None:
            return
        
        try:
            anchor_codes = batch['anchor_code']
            negative_codes = batch['negative_codes']
            
            bins = {
                'sibling_or_closer': 0,
                'cousin': 0,
                'distant': 0,
                'unknown': 0
            }
            total = 0
            
            for anchor_code, neg_codes in zip(anchor_codes, negative_codes):
                anchor_idx = self.code_to_idx.get(anchor_code)
                if anchor_idx is None:
                    continue
                
                for neg_code in neg_codes:
                    neg_idx = self.code_to_idx.get(neg_code)
                    if neg_idx is None:
                        bins['unknown'] += 1
                        total += 1
                        continue
                    
                    distance = self.ground_truth_distances[anchor_idx, neg_idx].item()
                    
                    if distance <= 2.0:
                        bins['sibling_or_closer'] += 1
                    elif distance <= 4.0:
                        bins['cousin'] += 1
                    else:
                        bins['distant'] += 1
                    
                    total += 1
            
            if total > 0:
                for name, count in bins.items():
                    self.log(
                        f'train/curriculum/tree_distance_{name}',
                        count / total,
                        batch_size=len(anchor_codes),
                        on_step=False,
                        on_epoch=True
                    )
        except Exception as e:
            logger.debug(f'Failed to log tree distance distribution: {e}')
        
    def _update_pseudo_labels(self):
        '''
        Runs clustering on the training dataset to generate pseudo-labels
        for false negative detection. [cite: 231-234]
        
        Issue #17: Uses Hyperbolic K-Means compatible with Lorentz model.
        Clusters embeddings directly in hyperbolic space using Lorentzian distances.
        '''
        if not hasattr(self.trainer, 'train_dataloader'):
            logger.warning('Trainer has no train_dataloader, cannot update pseudo-labels.')
            return

        logger.info('Generating embeddings for pseudo-label clustering (Hyperbolic K-Means)...')
        self.eval()
        all_embeddings = []
        all_codes = []
        
        # Issue #5: Sample a subset of batches for efficiency
        max_batches = 100  # Limit number of batches to process
        batch_count = 0

        try:            
            dataloader = self.trainer.train_dataloader
            
            for batch in dataloader:
                if batch_count >= max_batches:
                    break
                batch = self.transfer_batch_to_device(batch, self.device, 0)
                
                with torch.no_grad():
                    anchor_output = self(batch['anchor'])
                    # Use hyperbolic embeddings directly for hyperbolic K-Means
                    # Issue #17: Cluster in Lorentz space, not Euclidean
                    hyp_embs = anchor_output['embedding'].cpu()
                    all_embeddings.append(hyp_embs)
                    all_codes.extend(batch['anchor_code'])
                    batch_count += 1
            
            if not all_embeddings:
                logger.warning('No embeddings collected for pseudo-labeling')
                return
                
            all_embeddings = torch.cat(all_embeddings, dim=0)
            
            # Issue #5: More efficient cluster count calculation
            n_clusters = min(
                max(50, len(all_embeddings) // 20),  # At least 50, at most 1 per 20 samples
                self.hparams.fn_num_clusters
            )
            # Safeguard: ensure n_clusters >= 1 (KMeans requires at least 1 cluster)
            n_clusters = max(1, n_clusters)
                    
            logger.info(
                f'Clustering {len(all_embeddings)} hyperbolic embeddings into {n_clusters} clusters '
                f'using Hyperbolic K-Means (Lorentz model)...'
            )
            
            # Issue #17: Use Hyperbolic K-Means instead of Euclidean KMeans
            hyperbolic_kmeans = HyperbolicKMeans(
                n_clusters=n_clusters,
                curvature=self.hparams.curvature,
                max_iter=100,
                tol=1e-4,
                random_state=42,
                verbose=False
            )
            labels = hyperbolic_kmeans.fit_predict(all_embeddings)
            
            self.code_to_pseudo_label = {code: int(label) for code, label in zip(all_codes, labels)}
            logger.info(
                f'Pseudo-label map updated with {len(self.code_to_pseudo_label)} entries. '
                f'Clustering inertia: {hyperbolic_kmeans.inertia_:.4f}, '
                f'iterations: {hyperbolic_kmeans.n_iter_}'
            )
        
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

            # Compute NDCG@k for ranking evaluation (position-aware metric)
            # This replaces Spearman as the primary ranking metric
            ndcg_result = self.hierarchy_metrics.ndcg_ranking(
                emb_dists,
                gt_dists,
                k_values=[5, 10, 20]
            )
            for k in [5, 10, 20]:
                self.log(
                    f'val/ndcg@{k}',
                    self._to_python_scalar(ndcg_result[f'ndcg@{k}']),
                    batch_size=num_samples
                )
                self.log(
                    f'val/ndcg@{k}_n_queries',
                    self._to_python_scalar(ndcg_result[f'ndcg@{k}_n_queries']),
                    batch_size=num_samples
                )
            
            # Still compute Spearman for backward compatibility, but don't log to console
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
            
            # Update clustering based on curriculum scheduler
            if self.curriculum_scheduler is not None:
                should_update = self.curriculum_scheduler.should_update_clustering(
                    self.current_epoch,
                    self.hparams.fn_cluster_every_n_epochs
                )
                if should_update:
                    self._update_pseudo_labels()
            else:
                # Fallback to old behavior if scheduler not initialized
                if self.current_epoch >= self.hparams.fn_curriculum_start_epoch:
                    if self.hparams.fn_cluster_every_n_epochs > 0:
                        epochs_since_start = (
                            self.current_epoch - 
                            self.hparams.fn_curriculum_start_epoch
                        )
                        if epochs_since_start % self.hparams.fn_cluster_every_n_epochs == 0:
                            self._update_pseudo_labels()
            
            # Collect all evaluation metrics for JSON logging
            # Get training/validation losses from callback_metrics if available
            train_loss = None
            train_contrastive_loss = None
            val_loss = None
            
            if hasattr(self.trainer, 'callback_metrics'):
                train_loss = self.trainer.callback_metrics.get('train/total_loss', None)
                train_contrastive_loss = self.trainer.callback_metrics.get('train/contrastive_loss', None)
                val_loss = self.trainer.callback_metrics.get('val/contrastive_loss', None)
            
            epoch_metrics = {
                'epoch': self.current_epoch,
                # Training metrics (from callback_metrics)
                'train_loss': self._to_python_scalar(train_loss) if train_loss is not None else None,
                'train_contrastive_loss': self._to_python_scalar(train_contrastive_loss) if train_contrastive_loss is not None else None,
                # Validation metrics
                'val_loss': self._to_python_scalar(val_loss) if val_loss is not None else None,
                # Hyperbolic metrics
                'hyperbolic_radius_mean': self._to_python_scalar(diagnostics['radius_mean']),
                'hyperbolic_radius_std': self._to_python_scalar(diagnostics['radius_std']),
                'lorentz_norm_mean': self._to_python_scalar(diagnostics['lorentz_norm_mean']),
                'lorentz_norm_std': self._to_python_scalar(diagnostics['lorentz_norm_std']),
                'lorentz_norm_violation_max': self._to_python_scalar(diagnostics['violation_max']),
                'manifold_valid': bool(is_valid),
                # Embedding statistics
                'mean_norm': self._to_python_scalar(stats['mean_norm']),
                'std_norm': self._to_python_scalar(stats['std_norm']),
                'mean_pairwise_distance': self._to_python_scalar(stats['mean_pairwise_distance']),
                'std_pairwise_distance': self._to_python_scalar(stats['std_pairwise_distance']),
                # Collapse detection
                'norm_cv': self._to_python_scalar(collapse['norm_cv']),
                'distance_cv': self._to_python_scalar(collapse['distance_cv']),
                'collapse_detected': bool(collapse['any_collapse']),
                # Hierarchy preservation
                'cophenetic_correlation': self._to_python_scalar(cophenetic_result['correlation']),
                'cophenetic_n_pairs': int(cophenetic_result['n_pairs']),
                'spearman_correlation': self._to_python_scalar(spearman_result['correlation']),
                'spearman_n_pairs': int(spearman_result['n_pairs']),
                # Ranking metrics
                'ndcg@5': self._to_python_scalar(ndcg_result['ndcg@5']),
                'ndcg@10': self._to_python_scalar(ndcg_result['ndcg@10']),
                'ndcg@20': self._to_python_scalar(ndcg_result['ndcg@20']),
                'ndcg@5_n_queries': int(ndcg_result['ndcg@5_n_queries']),
                'ndcg@10_n_queries': int(ndcg_result['ndcg@10_n_queries']),
                'ndcg@20_n_queries': int(ndcg_result['ndcg@20_n_queries']),
                # Distortion metrics
                'mean_distortion': self._to_python_scalar(distortion['mean_distortion']),
                'std_distortion': self._to_python_scalar(distortion['std_distortion']),
                'median_distortion': self._to_python_scalar(distortion['median_distortion']),
                # Sample size
                'num_samples': int(num_samples),
            }
            
            # Add to history
            self.evaluation_metrics_history.append(epoch_metrics)
            
            # Save to JSON file
            metrics_file = self._get_metrics_file_path()
            if metrics_file:
                try:
                    metrics_file.parent.mkdir(parents=True, exist_ok=True)
                    with open(metrics_file, 'w') as f:
                        json.dump(self.evaluation_metrics_history, f, indent=2)
                    logger.debug(f'Saved evaluation metrics to {metrics_file}')
                except Exception as e:
                    logger.warning(f'Failed to save evaluation metrics to JSON: {e}')
                    
            logger.info(
                f'Correlation metrics: \n'
                f'  • Hierarchy preservation: cophenetic={cophenetic_result["correlation"]:.4f} '
                f'({cophenetic_result["n_pairs"]} pairs)\n'
                f'  • Ranking quality: NDCG@5={ndcg_result["ndcg@5"]:.4f}, '
                f'NDCG@10={ndcg_result["ndcg@10"]:.4f}, NDCG@20={ndcg_result["ndcg@20"]:.4f}\n'
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
        '''
        Configure optimizer and learning rate scheduler.
        
        Issue #4: Optimizer is reset when starting a new curriculum stage.
        This ensures fresh optimizer state for each curriculum stage.
        Issue #13: Add warmup + cosine decay for large training jobs.
        '''
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        
        use_warmup_cosine = getattr(self.hparams, 'use_warmup_cosine', False)
        
        if use_warmup_cosine:
            # Warmup + Cosine Annealing scheduler
            # Use LambdaLR for flexible scheduling that works even when trainer isn't initialized
            warmup_steps = getattr(self.hparams, 'warmup_steps', 500)
            base_lr = self.hparams.learning_rate
            min_lr = 1e-6
            
            # Capture self in closure for accessing trainer later
            model_self = self
            
            def lr_lambda(step):
                '''Compute learning rate multiplier for current step.'''
                if step < warmup_steps:
                    # Linear warmup: from 0.01 * base_lr to base_lr
                    return 0.01 + 0.99 * (step / warmup_steps)
                else:
                    # Cosine annealing after warmup
                    # Get total steps from trainer if available, otherwise use a large number
                    if hasattr(model_self, 'trainer') and model_self.trainer is not None:
                        if hasattr(model_self.trainer, 'estimated_stepping_batches'):
                            total_steps = model_self.trainer.estimated_stepping_batches
                        elif hasattr(model_self.trainer, 'num_training_batches'):
                            max_epochs = getattr(model_self.trainer, 'max_epochs', 15)
                            total_steps = model_self.trainer.num_training_batches * max_epochs
                        else:
                            # Fallback: use a large number (will be updated dynamically)
                            total_steps = 50000
                    else:
                        # Trainer not available yet, use fallback
                        total_steps = 50000
                    
                    # Cosine annealing: from base_lr to min_lr
                    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
                    progress = min(progress, 1.0)  # Clamp to [0, 1]
                    cosine_factor = 0.5 * (1 + np.cos(np.pi * progress))
                    # Scale from [min_lr/base_lr, 1.0]
                    return max(min_lr / base_lr, cosine_factor)
            
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                lr_lambda=lr_lambda
            )
            
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'step',  # Step-based for warmup and cosine decay
                    'frequency': 1
                }
            }
        else:
            # Use ReduceLROnPlateau for validation-based learning rate reduction
            # This helps prevent overfitting by reducing LR when validation loss plateaus
            scheduler = {
                'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode='min',
                    factor=0.5,  # Reduce LR by 50%
                    patience=3,  # Wait 3 epochs without improvement
                    min_lr=1e-6
                ),
                'monitor': 'val/contrastive_loss',
                'interval': 'epoch',
                'frequency': 1
            }
            
            return {
                'optimizer': optimizer,
                'lr_scheduler': scheduler
            }
    
    def on_train_start(self):
        '''
        Reset optimizer state when starting training (e.g., new curriculum stage).
        Issue #4: Ensures optimizer state is reset for curriculum learning.
        Issue #12: Initialize Structure-Aware Dynamic Curriculum (SADC) scheduler.
        '''
        # Initialize curriculum scheduler
        if hasattr(self, 'trainer') and self.trainer is not None:
            max_epochs = getattr(self.trainer, 'max_epochs', 15)
            self.curriculum_scheduler = CurriculumScheduler(
                max_epochs=max_epochs,
                tree_distance_alpha=getattr(self.hparams, 'tree_distance_alpha', 1.5),
                sibling_distance_threshold=2.0
            )
            logger.info('Curriculum scheduler initialized')
        
        # Reset optimizer state if this is a new curriculum stage
        # This is called by PyTorch Lightning when training starts
        if hasattr(self, 'trainer') and self.trainer is not None:
            # Check if we're resuming from a checkpoint
            if hasattr(self.trainer, 'ckpt_path') and self.trainer.ckpt_path:
                # If resuming same stage, keep optimizer state
                # If starting new stage, optimizer will be recreated fresh
                pass
