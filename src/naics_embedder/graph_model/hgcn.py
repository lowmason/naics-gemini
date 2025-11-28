# -------------------------------------------------------------------------------------------------
# Imports
# -------------------------------------------------------------------------------------------------

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import numpy as np
import polars as pl
import pytorch_lightning as pyl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pytorch_lightning import seed_everything
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax

from naics_embedder.graph_model.dataloader.hgcn_datamodule import HGCNDataModule
from naics_embedder.graph_model.evaluation import compute_validation_metrics
from naics_embedder.metrics.hierarchy_structure import (
    compute_hierarchy_retrieval_metrics,
    compute_radius_structure_metrics,
)
from naics_embedder.text_model.evaluation import EmbeddingEvaluator, HierarchyMetrics
from naics_embedder.text_model.hyperbolic import LorentzOps
from naics_embedder.utils.config import GraphConfig
from naics_embedder.utils.distance_matrix import load_distance_submatrix
from naics_embedder.utils.naics_hierarchy import NaicsHierarchy, load_naics_hierarchy
from naics_embedder.utils.utilities import setup_directory

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------------------------------

# Config is now imported from naics_embedder.utils.config as GraphConfig
# Using GraphConfig for type hints
Config = GraphConfig

# -------------------------------------------------------------------------------------------------
# PyTorch Geometric hyperbolic graph convolution (hgcn)
# -------------------------------------------------------------------------------------------------

class HyperbolicConvolution(MessagePassing):
    '''
    Hyperbolic graph convolution with edge-aware attention.

    Steps:
        1. Map hyperbolic features to tangent space
        2. Apply linear projection
        3. Edge-aware attention with learnable relation embeddings
        4. Residual + LayerNorm
        5. Map back to hyperboloid
    '''

    def __init__(
        self,
        dim: int,
        dropout: float = 0.1,
        learnable_curvature: bool = True,
        *,
        edge_type_count: int = 1,
        attn_hidden_dim: int = 64,
        sibling_type_id: Optional[int] = None,
        sibling_boost: float = 0.15,
    ):
        super().__init__(aggr='add')
        self.dim = dim
        self.lin = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(dim)
        if learnable_curvature:
            self.curvature = nn.Parameter(torch.tensor(1.0))
        else:
            self.register_buffer('curvature', torch.tensor(1.0))

        self.edge_type_emb = nn.Embedding(max(1, edge_type_count), dim)
        self.attn_mlp = nn.Sequential(
            nn.Linear(dim * 3, attn_hidden_dim),
            nn.SiLU(),
            nn.Linear(attn_hidden_dim, 1),
        )
        self.sibling_type_id = sibling_type_id
        self.sibling_boost = nn.Parameter(torch.tensor(sibling_boost))

    def forward(
        self,
        x_hyp: torch.Tensor,
        edge_index: torch.Tensor,
        edge_types: torch.Tensor,
        edge_weights: torch.Tensor,
    ) -> torch.Tensor:
        c_tensor = torch.clamp(self.curvature, min=0.1, max=10.0)
        c = float(c_tensor.item())

        x_tan = LorentzOps.log_map_zero(x_hyp, c=c)
        x_lin = self.lin(x_tan)
        x_agg = self.propagate(
            edge_index,
            x=x_lin,
            edge_type=edge_types,
            edge_weight=edge_weights,
        )

        x_tan_out = x_tan + self.dropout(x_agg)
        x_tan_out = self.ln(x_tan_out)
        return LorentzOps.exp_map_zero(x_tan_out, c=c)

    def message(
        self,
        x_i: torch.Tensor,
        x_j: torch.Tensor,
        edge_type: torch.Tensor,
        edge_weight: torch.Tensor,
        index: torch.Tensor,
    ) -> torch.Tensor:
        edge_emb = self.edge_type_emb(edge_type)
        attn_input = torch.cat([x_i, x_j, edge_emb], dim=-1)
        attn_score = self.attn_mlp(attn_input).squeeze(-1)

        # Relation-aware weighting
        attn_score = attn_score + torch.log(torch.clamp(edge_weight, min=1e-6))

        if self.sibling_type_id is not None:
            sibling_mask = (edge_type == self.sibling_type_id).float()
            attn_score = attn_score + sibling_mask * self.sibling_boost

        alpha = softmax(attn_score, index)
        message = (x_j + edge_emb) * alpha.unsqueeze(-1)
        return message * edge_weight.unsqueeze(-1)

# -------------------------------------------------------------------------------------------------
# HGCN model (stack of HyperbolicConvolution)
# -------------------------------------------------------------------------------------------------

class HGCN(nn.Module):

    def __init__(
        self,
        tangent_dim: int,
        n_layers: int,
        dropout: float,
        learnable_curvature: bool,
        learnable_loss_weights: bool = True,
        *,
        edge_type_count: int,
        edge_attention_hidden_dim: int,
        sibling_type_id: Optional[int],
        sibling_attention_boost: float,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                HyperbolicConvolution(
                    tangent_dim,
                    dropout,
                    learnable_curvature,
                    edge_type_count=edge_type_count,
                    attn_hidden_dim=edge_attention_hidden_dim,
                    sibling_type_id=sibling_type_id,
                    sibling_boost=sibling_attention_boost,
                ) for _ in range(n_layers)
            ]
        )

        # Learnable loss weights with uncertainty weighting
        if learnable_loss_weights:
            self.log_var_triplet = nn.Parameter(torch.zeros(1))
            self.log_var_level = nn.Parameter(torch.zeros(1))
            self.learnable_loss_weights = True
        else:
            self.learnable_loss_weights = False

    def forward(
        self,
        x_hyp: torch.Tensor,
        edge_index: torch.Tensor,
        edge_types: torch.Tensor,
        edge_weights: torch.Tensor,
    ) -> torch.Tensor:
        for layer in self.layers:
            x_hyp = layer(x_hyp, edge_index, edge_types, edge_weights)

        return x_hyp

    def get_loss_weights(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        '''
        Returns adaptive weights based on learned uncertainties.
        '''

        if self.learnable_loss_weights:
            w_triplet = 0.5 * torch.exp(-self.log_var_triplet)
            w_level = 0.5 * torch.exp(-self.log_var_level)
            reg = 0.5 * (self.log_var_triplet + self.log_var_level)
            return w_triplet, w_level, reg
        else:
            return torch.tensor(1.0), torch.tensor(1.0), torch.tensor(0.0)

# -------------------------------------------------------------------------------------------------
# Losses
# -------------------------------------------------------------------------------------------------

def triplet_loss_hyp(
    emb: torch.Tensor,
    anchors: torch.Tensor,
    positives: torch.Tensor,
    negatives: torch.Tensor,
    margin: Union[float, torch.Tensor],
    temperature: float = 1.0,
    c: float = 1.0,
    negative_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    '''
    Triplet loss with temperature scaling for curriculum learning.

    Args:
        temperature: Scale factor for loss (lower = sharper gradients)
    '''

    B = anchors.size(0)
    K = negatives.size(1)
    a = emb[anchors]
    p = emb[positives]
    d_ap = LorentzOps.lorentz_distance(a, p, c=c).unsqueeze(1)

    n = emb[negatives.reshape(-1)].view(B, K, -1)
    a_exp = a.unsqueeze(1).expand(-1, K, -1).reshape(-1, a.size(-1))

    d_an = LorentzOps.lorentz_distance(a_exp, n.reshape(-1, n.size(-1)), c=c).view(B, K)

    if isinstance(margin, torch.Tensor):
        margin_term = margin.view(B, 1)
    else:
        margin_term = torch.full((B, 1), float(margin), device=emb.device, dtype=d_ap.dtype)

    loss = F.relu((d_ap - d_an + margin_term) / temperature)

    if negative_weights is not None:
        weights = negative_weights
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)
        loss = (loss * weights).sum(dim=1)
    else:
        loss = loss.mean(dim=1)

    return loss.mean()

def level_radius_loss(emb: torch.Tensor, idx: torch.Tensor, levels: torch.Tensor) -> torch.Tensor:
    # radial distance on Lorentz model: sqrt(x0^2 - 1)
    radial = torch.sqrt(torch.clamp(emb[idx, 0].pow(2) - 1, min=1e-8))
    target = (levels[idx].float() - 2) * 0.5

    # Use Huber loss (smooth_l1) instead of MSE
    # More robust to outliers and prevents loss explosion
    # Quadratic for small errors, linear for large errors
    return F.smooth_l1_loss(radial, target)

@dataclass
class CurriculumState:
    '''Lightweight container for future curriculum features (Issues #62/#63).'''

    name: str = 'baseline'
    temperature: float = 1.0
    margin_scale: float = 1.0
    use_hard_negatives: bool = False
    max_relation: Optional[int] = None
    max_distance: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

# -------------------------------------------------------------------------------------------------
# Lightning Module
# -------------------------------------------------------------------------------------------------

class HGCNLightningModule(pyl.LightningModule):
    '''PyTorch Lightning module for single-stage HGCN training.'''

    def __init__(
        self,
        cfg: GraphConfig,
        embeddings: torch.Tensor,
        levels: torch.Tensor,
        edge_index: torch.Tensor,
        edge_types: torch.Tensor,
        edge_weights: torch.Tensor,
        edge_metadata: Dict[str, Any],
        node_metadata: Optional[pl.DataFrame] = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.model = HGCN(
            cfg.tangent_dim,
            cfg.n_hgcn_layers,
            cfg.dropout,
            cfg.learnable_curvature,
            cfg.learnable_loss_weights,
            edge_type_count=edge_metadata.get('edge_type_count', 1),
            edge_attention_hidden_dim=cfg.edge_attention_hidden_dim,
            sibling_type_id=edge_metadata.get('sibling_type_id'),
            sibling_attention_boost=cfg.edge_sibling_boost,
        )
        self.embeddings = nn.Parameter(embeddings.clone())
        self.register_buffer('edge_index', edge_index)
        self.register_buffer('edge_types', edge_types)
        self.register_buffer('edge_weights', edge_weights)
        self.register_buffer('levels', levels)
        self._optimizer: Optional[optim.Optimizer] = None
        self._train_losses: List[float] = []
        self._train_triplet: List[float] = []
        self._train_level: List[float] = []
        self._train_w_triplet: List[float] = []
        self._train_w_level: List[float] = []
        self._val_epoch_metrics: List[Dict[str, float]] = []
        self.history: List[Dict[str, Any]] = []
        self.curriculum_enabled = cfg.curriculum_enabled
        self._difficulty_thresholds = self._load_curriculum_thresholds(cfg)
        self.save_hyperparameters(
            ignore=[
                'embeddings',
                'levels',
                'edge_index',
                'edge_types',
                'edge_weights',
                'node_metadata',
            ]
        )
        self._ndcg_k_values = self._normalize_ndcg_k_values(cfg.ndcg_k_values)
        self._primary_ndcg = (
            10 if 10 in self._ndcg_k_values else self._ndcg_k_values[len(self._ndcg_k_values) // 2]
        )
        self.node_codes = self._extract_node_codes(node_metadata)
        distance_tensor = self._load_tree_distance_tensor(self.node_codes)
        if distance_tensor is not None:
            self.register_buffer('tree_distances', distance_tensor, persistent=False)
        else:
            self.tree_distances = None
        self.embedding_evaluator: Optional[EmbeddingEvaluator] = (
            EmbeddingEvaluator() if self.tree_distances is not None else None
        )
        self.hierarchy_metrics: Optional[HierarchyMetrics] = (
            HierarchyMetrics() if self.tree_distances is not None else None
        )
        self.parent_eval_top_k: int = getattr(cfg, 'parent_eval_top_k', 1)
        self.child_eval_top_k: int = getattr(cfg, 'child_eval_top_k', 5)
        self.naics_hierarchy: Optional[NaicsHierarchy] = None
        relations_path = getattr(cfg, 'relations_parquet', None)
        if relations_path:
            try:
                self.naics_hierarchy = load_naics_hierarchy(relations_path)
            except FileNotFoundError:
                logger.warning(
                    'NAICS relations parquet not found at %s; hierarchy diagnostics disabled',
                    relations_path,
                )
        self._full_val_metrics: Dict[str, float] = {}
        self._full_eval_done = False

    def _current_temperature(self) -> float:
        if self.cfg.num_epochs <= 1:
            return float(self.cfg.temperature_end)

        epoch = min(self.current_epoch, self.cfg.num_epochs - 1)
        progress = epoch / (self.cfg.num_epochs - 1)
        return float(
            self.cfg.temperature_start + progress *
            (self.cfg.temperature_end - self.cfg.temperature_start)
        )

    def _current_lr(self) -> float:
        if self._optimizer is None:
            return self.cfg.lr
        return float(self._optimizer.param_groups[0]['lr'])

    def forward(self) -> torch.Tensor:
        return self.model(self.embeddings, self.edge_index, self.edge_types, self.edge_weights)

    def _extract_batch_indices(self, batch: Dict[str, torch.Tensor]) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        anchors = batch['anchor_idx'].to(self.device)
        positives = batch['positive_idx'].to(self.device)
        negatives = batch['negative_indices'].to(self.device)
        neg_rel_ids = batch.get('negative_relation_ids')
        neg_distances = batch.get('negative_distances')
        neg_rel_margins = batch.get('negative_relation_margins')
        neg_dist_margins = batch.get('negative_distance_margins')

        if neg_rel_ids is None:
            neg_rel_ids_tensor = torch.full_like(
                negatives, -1, dtype=torch.long, device=self.device
            )
        else:
            neg_rel_ids_tensor = neg_rel_ids.to(self.device)

        if neg_distances is None:
            neg_dist_tensor = torch.zeros_like(negatives, dtype=torch.float32, device=self.device)
        else:
            neg_dist_tensor = neg_distances.to(self.device)

        if neg_rel_margins is None:
            neg_rel_margin_tensor = torch.zeros_like(
                negatives, dtype=torch.float32, device=self.device
            )
        else:
            neg_rel_margin_tensor = neg_rel_margins.to(self.device)

        if neg_dist_margins is None:
            neg_dist_margin_tensor = torch.zeros_like(
                negatives, dtype=torch.float32, device=self.device
            )
        else:
            neg_dist_margin_tensor = neg_dist_margins.to(self.device)

        return (
            anchors,
            positives,
            negatives,
            neg_rel_ids_tensor,
            neg_dist_tensor,
            neg_rel_margin_tensor,
            neg_dist_margin_tensor,
        )

    def on_train_epoch_start(self) -> None:
        self._train_losses.clear()
        self._train_triplet.clear()
        self._train_level.clear()
        self._train_w_triplet.clear()
        self._train_w_level.clear()

    def _load_curriculum_thresholds(self, cfg: GraphConfig) -> Dict[str, Any]:
        cache_dir = Path(cfg.curriculum_cache_dir)
        thresholds_path = cache_dir / 'difficulty_thresholds.json'
        if not thresholds_path.exists():
            return {}

        try:
            with open(thresholds_path, 'r') as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            return {}

    def _extract_node_codes(self, node_metadata: Optional[pl.DataFrame]) -> Optional[List[str]]:
        if node_metadata is None or 'code' not in node_metadata.columns:
            return None

        try:
            return node_metadata['code'].to_list()
        except Exception as exc:  # pragma: no cover - defensive safeguard
            logger.warning('Failed to extract node codes from metadata: %s', exc)
            return None

    def _normalize_ndcg_k_values(self, values: List[int]) -> List[int]:
        normalized = sorted({int(v) for v in values if int(v) > 0})
        return normalized or [10]

    def _load_tree_distance_tensor(self, node_codes: Optional[List[str]]) -> Optional[torch.Tensor]:
        path = getattr(self.cfg, 'distance_matrix_parquet', None)
        if not path or node_codes is None:
            if path and node_codes is None:
                logger.warning('Skipping hierarchy metrics: node codes unavailable')
            return None

        try:
            return load_distance_submatrix(path, node_codes)
        except (FileNotFoundError, ValueError) as exc:
            logger.warning('Skipping hierarchy metrics: %s', exc)
            return None

    def _threshold_value(
        self,
        key: str,
        fallback: Optional[Union[int, float]] = None,
    ) -> Optional[Union[int, float]]:
        if not self._difficulty_thresholds:
            return fallback

        value = self._difficulty_thresholds.get(key, fallback)
        if value is None:
            return fallback

        if isinstance(fallback, int):
            return int(value)
        if isinstance(fallback, float):
            return float(value)
        if isinstance(value, (int, float)):
            return value
        return fallback

    def _int_threshold(
        self,
        key: str,
        fallback: Optional[int] = None,
    ) -> Optional[int]:
        value = self._threshold_value(key, fallback)
        if value is None:
            return None
        return int(value)

    def _float_threshold(
        self,
        key: str,
        fallback: Optional[float] = None,
    ) -> Optional[float]:
        value = self._threshold_value(key, fallback)
        if value is None:
            return None
        return float(value)

    def _get_curriculum_state(self, batch_idx: int) -> CurriculumState:
        '''
        Determine the current curriculum phase and associated hyper-parameters.
        '''
        metadata = {
            'epoch': int(self.current_epoch),
            'batch_idx': int(batch_idx),
        }
        base_temperature = self._current_temperature()

        if not self.curriculum_enabled:
            return CurriculumState(
                name='static',
                temperature=base_temperature,
                margin_scale=1.0,
                use_hard_negatives=False,
                metadata=metadata,
            )

        epoch = self.current_epoch
        if epoch < self.cfg.curriculum_warmup_epochs:
            return CurriculumState(
                name='warmup',
                temperature=base_temperature,
                margin_scale=self.cfg.curriculum_margin_warmup_scale,
                use_hard_negatives=False,
                max_relation=self._int_threshold('phase1_max_relation', 1),
                max_distance=self._float_threshold('phase1_max_distance'),
                metadata=metadata,
            )

        if epoch < self.cfg.hard_negative_start_epoch:
            return CurriculumState(
                name='expansion',
                temperature=base_temperature,
                margin_scale=1.0,
                use_hard_negatives=False,
                max_relation=self._int_threshold('phase2_max_relation', 4),
                max_distance=self._float_threshold('phase2_max_distance'),
                metadata=metadata,
            )

        hard_temp = max(self.cfg.temperature_end * 0.9, base_temperature * 0.9)
        return CurriculumState(
            name='discrimination',
            temperature=hard_temp,
            margin_scale=1.0,
            use_hard_negatives=True,
            max_relation=None,
            max_distance=None,
            metadata=metadata,
        )

    def _prepare_negatives(
        self,
        embeddings: torch.Tensor,
        anchors: torch.Tensor,
        positives: torch.Tensor,
        negatives: torch.Tensor,
        neg_relation_ids: torch.Tensor,
        neg_distances: torch.Tensor,
        neg_relation_margins: torch.Tensor,
        neg_distance_margins: torch.Tensor,
        curriculum: CurriculumState,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        '''
        Apply curriculum filters followed by optional hard-negative mining.
        '''
        del positives, neg_relation_margins, neg_distance_margins  # currently unused
        filtered = self._filter_negatives_by_curriculum(
            negatives, neg_relation_ids, neg_distances, curriculum
        )
        return self._mine_hard_negatives(embeddings, anchors, filtered, curriculum)

    def _filter_negatives_by_curriculum(
        self,
        negatives: torch.Tensor,
        neg_relation_ids: torch.Tensor,
        neg_distances: torch.Tensor,
        curriculum: CurriculumState,
    ) -> torch.Tensor:
        if curriculum.max_relation is None and curriculum.max_distance is None:
            return negatives

        filtered_rows: List[torch.Tensor] = []
        for row_idx in range(negatives.size(0)):
            row_neg = negatives[row_idx]
            mask = torch.ones_like(row_neg, dtype=torch.bool, device=row_neg.device)

            if curriculum.max_relation is not None:
                mask = mask & (neg_relation_ids[row_idx] <= int(curriculum.max_relation))

            if curriculum.max_distance is not None:
                mask = mask & (neg_distances[row_idx] <= float(curriculum.max_distance))

            if not torch.any(mask):
                filtered_rows.append(row_neg)
                continue

            selected = row_neg[mask]
            if selected.size(0) < row_neg.size(0):
                padding = selected[-1:].repeat(row_neg.size(0) - selected.size(0))
                selected = torch.cat([selected, padding], dim=0)
            elif selected.size(0) > row_neg.size(0):
                selected = selected[:row_neg.size(0)]

            filtered_rows.append(selected)

        return torch.stack(filtered_rows, dim=0)

    def _mine_hard_negatives(
        self,
        embeddings: torch.Tensor,
        anchors: torch.Tensor,
        negatives: torch.Tensor,
        curriculum: CurriculumState,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if not curriculum.use_hard_negatives:
            return negatives, None

        batch_size, num_negatives = negatives.shape
        anchor_emb = embeddings[anchors]
        negative_emb = embeddings[negatives.reshape(-1)].view(batch_size, num_negatives, -1)

        anchor_exp = anchor_emb.unsqueeze(1).expand_as(negative_emb)
        dists = LorentzOps.lorentz_distance(
            anchor_exp.reshape(-1, anchor_exp.size(-1)),
            negative_emb.reshape(-1, negative_emb.size(-1)),
            c=1.0,
        ).view(batch_size, num_negatives)

        hard_fraction = max(1, int(num_negatives * max(self.cfg.hard_negative_fraction, 1e-3)))
        temp = max(self.cfg.hard_negative_temperature, 1e-3)
        weights = torch.softmax(-dists / temp, dim=1)

        hard_mask = torch.zeros_like(weights)
        topk_idx = torch.topk(-dists, k=hard_fraction, dim=1).indices
        hard_mask.scatter_(1, topk_idx, 1.0)

        weights = weights * hard_mask
        norm = weights.sum(dim=1, keepdim=True)
        safe_norm = norm + (norm == 0).float()
        weights = torch.where(
            norm > 0, weights / safe_norm, torch.full_like(weights, 1.0 / num_negatives)
        )

        order = torch.argsort(dists, dim=1)
        negatives_sorted = torch.gather(negatives, 1, order)
        weights_sorted = torch.gather(weights, 1, order)

        return negatives_sorted, weights_sorted

    def _compute_adaptive_margin(
        self,
        embeddings: torch.Tensor,
        anchors: torch.Tensor,
        positives: torch.Tensor,
        curriculum: CurriculumState,
    ) -> torch.Tensor:
        '''
        Adaptive margin inspired by NormAdaptiveMargin from Stage 3.
        '''
        base = torch.full(
            (anchors.size(0), ),
            float(self.cfg.triplet_margin),
            device=embeddings.device,
            dtype=embeddings.dtype,
        )

        if not self.cfg.use_adaptive_margin:
            return base * curriculum.margin_scale

        anchor_emb = embeddings[anchors]
        positive_emb = embeddings[positives]
        d_pos = LorentzOps.lorentz_distance(anchor_emb, positive_emb, c=1.0)
        anchor_radius = torch.sqrt(torch.clamp(anchor_emb[:, 0].pow(2) - 1, min=1e-8))

        norm_term = torch.tanh(anchor_radius / (self.cfg.adaptive_margin_norm_scale + 1e-6))
        distance_term = torch.tanh(d_pos / max(self.cfg.adaptive_margin_distance_scale, 1e-3))

        adjustments = self.cfg.adaptive_margin_alpha * (1.0 - distance_term) + 0.2 * (
            1.0 - norm_term
        )
        margin = base + adjustments
        margin = torch.clamp(margin, self.cfg.adaptive_margin_min, self.cfg.adaptive_margin_max)

        if curriculum.margin_scale != 1.0:
            margin = margin * curriculum.margin_scale

        return margin

    def _compute_triplet_component(
        self,
        embeddings: torch.Tensor,
        anchors: torch.Tensor,
        positives: torch.Tensor,
        negatives: torch.Tensor,
        curriculum: CurriculumState,
        negative_weights: Optional[torch.Tensor],
    ) -> torch.Tensor:
        margin = self._compute_adaptive_margin(embeddings, anchors, positives, curriculum)
        return triplet_loss_hyp(
            embeddings,
            anchors,
            positives,
            negatives,
            margin,
            temperature=curriculum.temperature,
            c=1.0,
            negative_weights=negative_weights,
        )

    def _compute_level_component(
        self,
        embeddings: torch.Tensor,
        anchors: torch.Tensor,
        positives: torch.Tensor,
        negatives: torch.Tensor,
    ) -> torch.Tensor:
        unique_idx = torch.cat([anchors, positives, negatives.view(-1)]).unique()
        levels_tensor: torch.Tensor = self.levels  # type: ignore[assignment]
        return level_radius_loss(embeddings, unique_idx, levels_tensor)

    def _apply_loss_weights(
        self,
        triplet_loss_value: torch.Tensor,
        level_loss_value: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, float]]]:
        if self.model.learnable_loss_weights:
            w_trip, w_lvl, reg = self.model.get_loss_weights()  # type: ignore[misc]
            loss = (w_trip * triplet_loss_value) + (w_lvl * level_loss_value) + reg
            weights = {
                'w_triplet': float(w_trip.detach().cpu()),
                'w_level': float(w_lvl.detach().cpu()),
            }
            return loss, weights

        loss = (self.cfg.w_triplet * triplet_loss_value) + (self.cfg.w_per_level * level_loss_value)
        return loss, None

    def _record_training_step(
        self,
        loss: torch.Tensor,
        triplet_loss_value: torch.Tensor,
        level_loss_value: torch.Tensor,
        curriculum: CurriculumState,
        weight_metadata: Optional[Dict[str, float]],
    ) -> None:

        def _to_float(value: torch.Tensor) -> float:
            return float(value.detach().cpu())

        self._train_losses.append(_to_float(loss))
        self._train_triplet.append(_to_float(triplet_loss_value))
        self._train_level.append(_to_float(level_loss_value))

        if weight_metadata is not None:
            self._train_w_triplet.append(weight_metadata['w_triplet'])
            self._train_w_level.append(weight_metadata['w_level'])

        self.log('train/loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train/triplet_loss', triplet_loss_value, on_step=False, on_epoch=True)
        self.log('train/level_loss', level_loss_value, on_step=False, on_epoch=True)
        self.log(
            'train/temperature',
            curriculum.temperature,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        (
            anchors,
            positives,
            negatives,
            neg_relation_ids,
            neg_distances,
            neg_relation_margins,
            neg_distance_margins,
        ) = self._extract_batch_indices(batch)
        embeddings = self.forward()

        curriculum = self._get_curriculum_state(batch_idx)
        negatives, negative_weights = self._prepare_negatives(
            embeddings,
            anchors,
            positives,
            negatives,
            neg_relation_ids,
            neg_distances,
            neg_relation_margins,
            neg_distance_margins,
            curriculum,
        )

        triplet_component = self._compute_triplet_component(
            embeddings,
            anchors,
            positives,
            negatives,
            curriculum,
            negative_weights,
        )
        level_component = self._compute_level_component(
            embeddings,
            anchors,
            positives,
            negatives,
        )

        loss, weight_metadata = self._apply_loss_weights(triplet_component, level_component)
        self._record_training_step(
            loss, triplet_component, level_component, curriculum, weight_metadata
        )

        return loss

    def _should_run_full_eval(self, batch_idx: int) -> bool:
        if (
            self.tree_distances is None or self.embedding_evaluator is None
            or self.hierarchy_metrics is None
        ):
            return False

        if self.cfg.full_eval_frequency <= 0 or self._full_eval_done:
            return False

        trainer = getattr(self, '_trainer', None)
        if trainer is not None and getattr(trainer, 'sanity_checking', False):
            return False

        if batch_idx != 0:
            return False

        step = int(getattr(self, 'global_step', 0))
        return (step % self.cfg.full_eval_frequency) == 0

    def _compute_full_validation_metrics(self, embeddings: torch.Tensor
                                         ) -> Optional[Dict[str, torch.Tensor]]:
        if (
            self.tree_distances is None or self.embedding_evaluator is None
            or self.hierarchy_metrics is None
        ):
            return None

        evaluator = self.embedding_evaluator
        hierarchy = self.hierarchy_metrics
        evaluator.device = str(self.device)
        hierarchy.device = str(self.device)

        try:
            with torch.no_grad():
                emb_dists = evaluator.compute_pairwise_distances(
                    embeddings.detach(), metric='lorentz', curvature=1.0
                )
                tree_dists = self.tree_distances
                if emb_dists.shape != tree_dists.shape:
                    logger.warning(
                        'Skipping hierarchy metrics: embedding distance shape %s != '
                        'tree distance shape %s',
                        emb_dists.shape,
                        tree_dists.shape,
                    )
                    return None

                cophenetic = hierarchy.cophenetic_correlation(emb_dists, tree_dists)
                spearman = hierarchy.spearman_correlation(emb_dists, tree_dists)
                ndcg = hierarchy.ndcg_ranking(emb_dists, tree_dists, k_values=self._ndcg_k_values)
                distortion = hierarchy.distortion(emb_dists, tree_dists)
        except RuntimeError as err:
            logger.warning('Skipping hierarchy metrics due to runtime error: %s', err)
            return None

        metrics: Dict[str, torch.Tensor] = {}
        cophenetic_corr = cast(torch.Tensor, cophenetic['correlation'])
        metrics['cophenetic_correlation'] = cophenetic_corr.detach()
        metrics['cophenetic_n_pairs'] = torch.tensor(
            float(cophenetic['n_pairs']), device=metrics['cophenetic_correlation'].device
        )
        spearman_corr = cast(torch.Tensor, spearman['correlation'])
        metrics['spearman_correlation'] = spearman_corr.detach()
        metrics['spearman_n_pairs'] = torch.tensor(
            float(spearman['n_pairs']), device=metrics['spearman_correlation'].device
        )

        for key, value in distortion.items():
            metrics[key] = cast(torch.Tensor, value).detach()

        for k in self._ndcg_k_values:
            ndcg_value = cast(torch.Tensor, ndcg[f'ndcg@{k}'])
            metrics[f'ndcg@{k}'] = ndcg_value.detach()
            n_queries = ndcg.get(f'ndcg@{k}_n_queries', 0)
            metrics[f'ndcg@{k}_n_queries'] = torch.tensor(
                float(n_queries), device=metrics[f'ndcg@{k}'].device
            )

        if (
            self.naics_hierarchy is not None and self.node_codes is not None and len(
                self.node_codes
            ) == embeddings.size(0)
        ):
            radius_metrics = compute_radius_structure_metrics(
                embeddings.detach(),
                self.node_codes,
                self.naics_hierarchy,
            )
            for name, value in radius_metrics.items():
                metrics[name] = torch.tensor(value, device=embeddings.device)

            retrieval_metrics = compute_hierarchy_retrieval_metrics(
                emb_dists,
                self.node_codes,
                self.naics_hierarchy,
                parent_top_k=self.parent_eval_top_k,
                child_top_k=self.child_eval_top_k,
            )
            for name, value in retrieval_metrics.items():
                metrics[name] = torch.tensor(value, device=embeddings.device)
        elif self.naics_hierarchy is not None:
            logger.warning(
                'Skipping hierarchy diagnostics: code count mismatch (%s codes, %s embeddings)',
                len(self.node_codes) if self.node_codes is not None else 0,
                embeddings.size(0),
            )

        return metrics

    def on_validation_epoch_start(self) -> None:
        self._val_epoch_metrics.clear()
        self._full_val_metrics.clear()
        self._full_eval_done = False

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        anchors = batch['anchor_idx'].to(self.device)
        positives = batch['positive_idx'].to(self.device)
        negatives = batch['negative_indices'].to(self.device)

        emb_upd = self.forward()
        metrics = compute_validation_metrics(
            emb_upd,
            anchors,
            positives,
            negatives,
            c=1.0,
            top_k=min(5, negatives.size(1)),
            as_tensors=True,
        )

        eval_curriculum = CurriculumState(name='eval', temperature=self.cfg.temperature_end)
        val_triplet_loss = self._compute_triplet_component(
            emb_upd,
            anchors,
            positives,
            negatives,
            eval_curriculum,
            negative_weights=None,
        )
        metrics_tensor = cast(Dict[str, torch.Tensor], metrics)
        metrics_tensor['triplet_loss'] = val_triplet_loss

        for name, value in metrics_tensor.items():
            self.log(
                f'val/{name}',
                value,
                on_step=False,
                on_epoch=True,
                prog_bar=(name == 'relation_accuracy'),
            )

        # metrics contains Tensors when as_tensors=True, but we defensively handle floats
        processed_metrics: Dict[str, float] = {}
        for key, value in metrics_tensor.items():
            if isinstance(value, torch.Tensor):
                processed_metrics[key] = float(value.detach().cpu())
            else:
                processed_metrics[key] = float(value)

        self._val_epoch_metrics.append(processed_metrics)

        if self._should_run_full_eval(batch_idx):
            full_metrics = self._compute_full_validation_metrics(emb_upd)
            if full_metrics:
                self._full_eval_done = True
                for name, value in full_metrics.items():
                    tensor_value: torch.Tensor = (
                        value if isinstance(value, torch.Tensor) else torch.tensor(
                            value, device=self.device
                        )
                    )
                    should_prog_bar = (
                        name == 'cophenetic_correlation' or name == f'ndcg@{self._primary_ndcg}'
                    )
                    self.log(
                        f'val/{name}',
                        tensor_value,
                        on_step=False,
                        on_epoch=True,
                        prog_bar=should_prog_bar,
                    )
                    scalar_value = float(tensor_value.detach().cpu())
                    self._full_val_metrics[f'val_{name}'] = scalar_value

    def on_train_epoch_end(self) -> None:
        if not self._train_losses:
            return

        epoch_stats: Dict[str, Any] = {
            'epoch': int(self.current_epoch + 1),
            'loss': float(np.mean(self._train_losses)),
            'triplet_loss': float(np.mean(self._train_triplet)),
            'level_loss': float(np.mean(self._train_level)),
            'temperature': self._current_temperature(),
            'lr': self._current_lr(),
        }

        if self.model.learnable_loss_weights and self._train_w_triplet:
            epoch_stats['w_triplet'] = float(np.mean(self._train_w_triplet))
            epoch_stats['w_level'] = float(np.mean(self._train_w_level))

        self.history.append(epoch_stats)

    def on_validation_epoch_end(self) -> None:
        if not self._val_epoch_metrics or not self.history:
            return

        avg_metrics = {
            f'val_{k}': float(np.mean([m[k] for m in self._val_epoch_metrics]))
            for k in self._val_epoch_metrics[0].keys()
        }
        self.history[-1].update(avg_metrics)
        if self._full_val_metrics:
            self.history[-1].update(self._full_val_metrics)

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            [
                {
                    'params': self.model.parameters(),
                    'lr': self.cfg.lr
                },
                {
                    'params': [self.embeddings],
                    'lr': self.cfg.lr * 0.1
                },
            ],
            weight_decay=self.cfg.weight_decay,
        )
        self._optimizer = optimizer

        warmup_epochs = int(self.cfg.num_epochs * self.cfg.warmup_ratio)

        def lr_lambda(epoch: int) -> float:
            if warmup_epochs > 0 and epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs

            denom = max(1, self.cfg.num_epochs - warmup_epochs)
            progress = (epoch - warmup_epochs) / denom
            return max(0.0, 0.5 * (1 + math.cos(math.pi * progress)))

        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
            },
        }

    def get_final_embeddings(self) -> torch.Tensor:
        with torch.no_grad():
            final_emb = self.forward()
        return final_emb.detach().cpu()

    def export_history(self) -> List[Dict[str, Any]]:
        return self.history

# -------------------------------------------------------------------------------------------------
# IO helpers
# -------------------------------------------------------------------------------------------------

def load_embeddings(parquet_path: str,
                    device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, pl.DataFrame]:
    '''
    Load hyperbolic embeddings from parquet file.

    Expects columns prefixed with 'hyp_e' (e.g., hyp_e0, hyp_e1, ...) for embeddings.
    '''
    df = pl.read_parquet(parquet_path)

    # Find embedding columns (hyp_e* pattern) and enforce a deterministic order
    embedding_cols = [col for col in df.columns if col.startswith('hyp_e')]
    if not embedding_cols:
        raise ValueError(f'No embedding columns found (expected hyp_e* pattern) in {parquet_path}')

    embedding_cols = sorted(
        embedding_cols,
        key=lambda name: int(name.replace('hyp_e', ''))
        if name.replace('hyp_e', '').isdigit() else name,
    )

    emb = df.select(embedding_cols).to_torch(dtype=pl.Float32).to(device)

    levels = df.get_column('level').to_torch().long().to(device)

    return emb, levels, df

def load_edge_index(
    relations_path: str,
    device: torch.device,
    allowed_relations: Optional[List[str]] = None,
    relation_weights: Optional[Dict[str, float]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
    df_rel = pl.read_parquet(relations_path).select(
        pl.col('idx_i'),
        pl.col('idx_j'),
        pl.col('relation'),
        pl.col('relation_id'),
    )

    if allowed_relations:
        df_rel = df_rel.filter(pl.col('relation').is_in(allowed_relations))

    if df_rel.is_empty():
        raise ValueError('No edges found for the provided relation filters.')

    idx_i = df_rel['idx_i'].to_numpy()
    idx_j = df_rel['idx_j'].to_numpy()
    relation_names = df_rel['relation'].to_list()
    relation_ids = df_rel['relation_id'].to_numpy()

    unique_relations = sorted(set(relation_names))
    type_to_id = {name: idx for idx, name in enumerate(unique_relations)}

    base_edge_types = np.array([type_to_id[name] for name in relation_names], dtype=np.int64)
    base_weights: List[float] = []
    for name, rel_id in zip(relation_names, relation_ids):
        base = relation_weights.get(name, 1.0) if relation_weights else 1.0
        distance_scale = 1.0 / (1.0 + max(0.0, float(rel_id) - 1) * 0.2)
        weight = max(0.05, float(base) * distance_scale)
        base_weights.append(weight)

    base_weights_arr = np.array(base_weights, dtype=np.float32)

    edges_np = np.stack([idx_i, idx_j], axis=0)
    # Make bidirectional
    edges_bidirectional = np.concatenate([edges_np, edges_np[::-1]], axis=1)
    edge_types = np.concatenate([base_edge_types, base_edge_types], axis=0)
    edge_weights = np.concatenate([base_weights_arr, base_weights_arr], axis=0)

    num_nodes = int(edges_bidirectional.max()) + 1
    loop_type_name = '__self_loop__'
    loop_type_id = type_to_id.setdefault(loop_type_name, len(type_to_id))
    loop_edges = np.stack([np.arange(num_nodes), np.arange(num_nodes)], axis=0)
    edges_with_loops = np.concatenate([edges_bidirectional, loop_edges], axis=1)
    edge_types = np.concatenate(
        [edge_types, np.full(num_nodes, loop_type_id, dtype=np.int64)], axis=0
    )
    edge_weights = np.concatenate([edge_weights, np.ones(num_nodes, dtype=np.float32)], axis=0)

    edge_index = torch.from_numpy(edges_with_loops).long().to(device)
    edge_type_tensor = torch.from_numpy(edge_types).long().to(device)
    edge_weight_tensor = torch.from_numpy(edge_weights).float().to(device)

    metadata = {
        'type_mapping': type_to_id,
        'edge_type_count': len(type_to_id),
        'sibling_type_id': type_to_id.get('sibling'),
    }

    return edge_index, edge_type_tensor, edge_weight_tensor, metadata

# -------------------------------------------------------------------------------------------------
# Save results
# -------------------------------------------------------------------------------------------------

def save_outputs(
    outdir: str,
    final_emb: torch.Tensor,
    orig_df: pl.DataFrame,
    cfg: Config,
    model: HGCN,
    log: List[Dict[str, Any]],
) -> pl.DataFrame:
    emb_np = final_emb.detach().cpu().numpy()

    base = orig_df.select('index', 'level', 'code')

    emb_schema = {f'hgcn_e{i}': pl.Float64 for i in range(emb_np.shape[1])}
    emb_df = pl.DataFrame(emb_np, schema=emb_schema)

    result_df = base.hstack(emb_df)

    (result_df.write_parquet(cfg.output_parquet))

    # Save final model state
    save_dict = {
        'state_dict': model.state_dict(),
        'embeddings': final_emb.detach().cpu(),
        'config': cfg.model_dump(),
    }

    if model.learnable_loss_weights:
        save_dict['final_log_var_triplet'] = float(
            model.log_var_triplet.item()
        )  # type: ignore[attr-defined]
        save_dict['final_log_var_level'] = float(
            model.log_var_level.item()
        )  # type: ignore[attr-defined]

    torch.save(save_dict, f'{outdir}/hgcn_model_final.pt')

    with open(f'{outdir}/training_log.json', 'w') as f:
        json.dump(log, f, indent=2)

    with open(f'{outdir}/config.json', 'w') as f:
        json.dump(cfg.model_dump(), f, indent=2)

    print(f'\n{"=" * 80}')
    print('TRAINING COMPLETE')
    print(f'{"=" * 80}')
    if log:
        print(f'Best loss: {min(x["loss"] for x in log):.4f}')
        print(f'Final loss: {log[-1]["loss"]:.4f}')
    else:
        print('No training history recorded.')

    if model.learnable_loss_weights and log:
        final_w_t = log[-1].get('w_triplet', 'N/A')
        final_w_l = log[-1].get('w_level', 'N/A')
        if isinstance(final_w_t, float):
            print(f'Final weights: triplet={final_w_t:.4f}, level={final_w_l:.4f}')

    print('\nOutputs:')
    print(f'  Embeddings: {cfg.output_parquet}')
    outdir_path = Path(outdir)
    print(f'  Final model: {outdir_path / "hgcn_model_final.pt"}')
    print(f'  Training log: {outdir_path / "training_log.json"}')

    return result_df

# -------------------------------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------------------------------

def main(config_file: str = 'conf/config.yaml') -> None:
    '''Main entry point for single-stage HGCN training via PyTorch Lightning.'''
    base_cfg = GraphConfig.from_yaml(config_file)
    outdir = setup_directory(base_cfg.output_dir)

    print('=' * 80)
    print('HGCN TRAINING (PyTorch Lightning)')
    print('=' * 80)
    print(f'Config: {config_file}')
    print(f'Output directory: {outdir}')

    seed_everything(base_cfg.seed, workers=True)
    np.random.seed(base_cfg.seed)

    emb, levels, df = load_embeddings(base_cfg.encodings_parquet, torch.device('cpu'))
    edge_index, edge_types, edge_weights, edge_meta = load_edge_index(
        base_cfg.relations_parquet,
        torch.device('cpu'),
        allowed_relations=base_cfg.edge_relations,
        relation_weights=base_cfg.edge_relation_weights,
    )

    print(f'Loaded embeddings: N={emb.size(0)}, dim={emb.size(1)}')
    print(f'Graph edges: {edge_index.size(1)}')

    datamodule = HGCNDataModule(base_cfg)
    lit_module = HGCNLightningModule(
        base_cfg, emb, levels, edge_index, edge_types, edge_weights, edge_meta, node_metadata=df
    )

    accelerator = base_cfg.device if base_cfg.device != 'auto' else 'auto'
    trainer = pyl.Trainer(
        max_epochs=base_cfg.num_epochs,
        accelerator=accelerator,
        devices=1,
        gradient_clip_val=base_cfg.gradient_clip_norm,
        log_every_n_steps=max(1, base_cfg.epoch_every),
        enable_checkpointing=False,
    )

    trainer.fit(lit_module, datamodule=datamodule)

    final_emb = lit_module.get_final_embeddings()
    save_outputs(
        str(outdir), final_emb, df, base_cfg, lit_module.model, lit_module.export_history()
    )

if __name__ == '__main__':
    main()
