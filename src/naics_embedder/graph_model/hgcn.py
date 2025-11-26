# -------------------------------------------------------------------------------------------------
# Imports
# -------------------------------------------------------------------------------------------------

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops

from naics_embedder.graph_model.dataloader.hgcn_datamodule import Config as LoaderCfg
from naics_embedder.graph_model.dataloader.hgcn_datamodule import create_dataloader
from naics_embedder.graph_model.evaluation import compute_validation_metrics
from naics_embedder.text_model.hyperbolic import LorentzOps
from naics_embedder.utils.config import GraphConfig
from naics_embedder.utils.utilities import pick_device, setup_directory

# -------------------------------------------------------------------------------------------------
# Config
# -------------------------------------------------------------------------------------------------

# Config is now imported from naics_embedder.utils.config as GraphConfig
# Using GraphConfig for type hints
Config = GraphConfig


# -------------------------------------------------------------------------------------------------
# PyTorch Geometric hyperbolic graph convolution (hgcn)
# -------------------------------------------------------------------------------------------------


class HyperbolicConvolution(MessagePassing):
    '''
    Hyperbolic graph convolution in Lorentz model using PyG's MessagePassing.
    Tangent-space linear -> aggregate -> residual -> layernorm -> exp map.
    Aggregation = mean over neighbors (propagate with 'mean').
    '''

    def __init__(self, dim: int, dropout: float = 0.1, learnable_curvature: bool = True):
        super().__init__(aggr='mean')
        self.dim = dim
        self.lin = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(dim)
        if learnable_curvature:
            self.curvature = nn.Parameter(torch.tensor(1.0))
        else:
            self.register_buffer('curvature', torch.tensor(1.0))

    def forward(self, x_hyp: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # Clamp curvature to safe range
        c_tensor = torch.clamp(self.curvature, min=0.1, max=10.0)
        c = float(c_tensor.item())

        # Issue #10: Fix gradient blocking - ensure gradients flow through all operations
        # Map from hyperboloid to tangent (requires_grad=True maintained)
        x_tan = LorentzOps.log_map_zero(x_hyp, c=c)
        x_lin = self.lin(x_tan)

        # Propagate: message passing on tangent features
        x_agg = self.propagate(edge_index, x=x_lin)

        # Issue #10: Use in-place operations carefully to preserve gradients
        # Residual in tangent space, then LN
        x_tan_out = x_tan + self.dropout(x_agg)
        x_tan_out = self.ln(x_tan_out)

        # Map back to hyperboloid (gradients flow through exp_map)
        return LorentzOps.exp_map_zero(x_tan_out, c=c)

    def message(self, x_j: torch.Tensor) -> torch.Tensor:
        # Messages are neighbor features (already linear-transformed)
        return x_j


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
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                HyperbolicConvolution(tangent_dim, dropout, learnable_curvature)
                for _ in range(n_layers)
            ]
        )

        # Learnable loss weights with uncertainty weighting
        if learnable_loss_weights:
            self.log_var_triplet = nn.Parameter(torch.zeros(1))
            self.log_var_level = nn.Parameter(torch.zeros(1))
            self.learnable_loss_weights = True
        else:
            self.learnable_loss_weights = False

    def forward(self, x_hyp: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x_hyp = layer(x_hyp, edge_index)

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
    margin: float,
    temperature: float = 1.0,
    c: float = 1.0,
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

    # Apply temperature scaling
    loss = F.relu((d_ap - d_an + margin) / temperature)

    return loss.mean()


def level_radius_loss(emb: torch.Tensor, idx: torch.Tensor, levels: torch.Tensor) -> torch.Tensor:
    # radial distance on Lorentz model: sqrt(x0^2 - 1)
    radial = torch.sqrt(emb[idx, 0] ** 2 - 1)
    target = (levels[idx].float() - 2) * 0.5

    # Use Huber loss (smooth_l1) instead of MSE
    # More robust to outliers and prevents loss explosion
    # Quadratic for small errors, linear for large errors
    return F.smooth_l1_loss(radial, target)


# -------------------------------------------------------------------------------------------------
# Learning Rate Scheduler with Warmup
# -------------------------------------------------------------------------------------------------


class WarmupCosineScheduler:
    '''
    Learning rate scheduler with linear warmup followed by cosine annealing.
    '''

    def __init__(
        self, optimizer: optim.Optimizer, warmup_epochs: int, total_epochs: int, min_lr: float = 0.0
    ):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.current_epoch = 0

    def step(self):
        self.current_epoch += 1

        if self.current_epoch <= self.warmup_epochs:
            # Linear warmup
            lr_scale = self.current_epoch / self.warmup_epochs
        else:
            # Cosine annealing
            progress = (self.current_epoch - self.warmup_epochs) / (
                self.total_epochs - self.warmup_epochs
            )
            lr_scale = 0.5 * (1 + np.cos(np.pi * progress))
            lr_scale = max(lr_scale, self.min_lr / self.base_lrs[0])

        for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            param_group['lr'] = base_lr * lr_scale

    def get_last_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]


# -------------------------------------------------------------------------------------------------
# IO helpers
# -------------------------------------------------------------------------------------------------


def load_embeddings(
    parquet_path: str, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, pl.DataFrame]:
    '''
    Load hyperbolic embeddings from parquet file.

    Expects columns prefixed with 'hyp_e' (e.g., hyp_e0, hyp_e1, ...) for embeddings.
    '''
    df = pl.read_parquet(parquet_path)

    # Find embedding columns (hyp_e* pattern)
    embedding_cols = [col for col in df.columns if col.startswith('hyp_e')]
    if not embedding_cols:
        raise ValueError(f'No embedding columns found (expected hyp_e* pattern) in {parquet_path}')

    emb = df.select(embedding_cols).to_torch(dtype=pl.Float32).to(device)

    levels = df.get_column('level').to_torch().long().to(device)

    return emb, levels, df


def load_edge_index(relations_path: str, device: torch.device) -> torch.Tensor:
    df_rel = pl.read_parquet(relations_path)

    edges = df_rel.filter(pl.col('relationship').eq('child')).select('idx_i', 'idx_j').to_numpy()

    edge_index = (
        torch.from_numpy(
            # make bidirectional
            np.concatenate([edges.T, edges[:, ::-1].T], axis=1)
        )
        .long()
        .to(device)
    )

    # Optionally add self loops (often stabilizes message passing)
    edge_index, _ = add_self_loops(edge_index, num_nodes=int(edge_index.max().item()) + 1)

    return edge_index


def create_contrastive_dataloader(cfg: Config) -> torch.utils.data.DataLoader:
    '''
    Create contrastive dataloader from GraphConfig.

    Uses curriculum configuration if available, otherwise uses base config values.
    '''
    # Get curriculum config if available
    curriculum_dict = {}
    # Note: GraphConfig doesn't have curriculum attribute, using base config values

    # Create loader config, merging base config with curriculum overrides
    # Derive descriptions_parquet path from encodings_parquet path
    descriptions_path = str(
        Path(cfg.encodings_parquet).parent.parent / 'naics_descriptions.parquet'
    )
    if not Path(descriptions_path).exists():
        # Fallback to default
        descriptions_path = './data/naics_descriptions.parquet'

    loader_cfg = LoaderCfg(
        training_pairs_path=cfg.training_pairs_path,
        descriptions_parquet=descriptions_path,
        batch_size=cfg.batch_size,
        shuffle=cfg.shuffle,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        n_positive_samples=curriculum_dict.get('n_positives', cfg.n_positive_samples),
        n_negatives=curriculum_dict.get('n_negatives', cfg.k_total),
        anchor_level=curriculum_dict.get('anchor_level'),
        relation_margin=curriculum_dict.get('relation_margin'),
        distance_margin=curriculum_dict.get('distance_margin'),
        positive_level=curriculum_dict.get('positive_level'),
        positive_relation=curriculum_dict.get('positive_relation'),
        positive_distance=curriculum_dict.get('positive_distance'),
        negative_level=curriculum_dict.get('negative_level'),
        negative_relation=curriculum_dict.get('negative_relation'),
        negative_distance=curriculum_dict.get('negative_distance'),
        seed=cfg.seed,
    )

    return create_dataloader(loader_cfg)


# -------------------------------------------------------------------------------------------------
# Training loop for a single curriculum stage
# -------------------------------------------------------------------------------------------------


def train_stage(
    stage: int,
    model: nn.Module,
    embeddings: nn.Parameter,
    data: Data,
    levels: torch.Tensor,
    loader: torch.utils.data.DataLoader,
    cfg: Config,
) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
    '''
    Train the model for one curriculum stage.
    '''

    opt = optim.AdamW(
        [
            {'params': model.parameters(), 'lr': cfg.lr},
            {'params': [embeddings], 'lr': cfg.lr * 0.1},
        ],
        weight_decay=cfg.weight_decay,
    )

    # Use warmup + cosine scheduler
    warmup_epochs = int(cfg.num_epochs * cfg.warmup_ratio)
    sched = WarmupCosineScheduler(opt, warmup_epochs=warmup_epochs, total_epochs=cfg.num_epochs)

    # Temperature schedule
    temp_schedule = np.linspace(cfg.temperature_start, cfg.temperature_end, cfg.num_epochs)

    log: List[Dict[str, Any]] = []
    best: float = float('inf')

    print(f'\n{"=" * 80}')
    print(f'CURRICULUM STAGE {stage}')
    print(f'{"=" * 80}')
    print(f'Epochs: {cfg.num_epochs} | Warmup: {warmup_epochs} | LR: {cfg.lr:.2e}')
    print(f'Temperature: {cfg.temperature_start:.3f} -> {cfg.temperature_end:.3f}')
    print(f'Samples: {cfg.n_positive_samples} | K: {cfg.k_total}')
    print(f'Hard: {cfg.pct_hard:.0%} | Medium: {cfg.pct_medium:.0%} | Easy: {cfg.pct_easy:.0%}')
    if cfg.allowed_relations:
        print(f'Relations: {cfg.allowed_relations}')
    if cfg.min_code_level is not None:
        print(f'Code levels: {cfg.min_code_level}-{cfg.max_code_level}')

    if model.learnable_loss_weights:
        print('Using learnable loss weights (uncertainty weighting)')
    else:
        print(f'Using fixed weights: triplet={cfg.w_triplet}, level={cfg.w_per_level}')
    print()

    for epoch in range(1, cfg.num_epochs + 1):
        model.train()

        temperature = temp_schedule[epoch - 1]
        losses, l_trips, l_lvls = [], [], []
        w_trip_log, w_lvl_log = [], []

        for batch in loader:
            anchors = batch['anchor_idx'].to(embeddings.device)
            positives = batch['positive_idx'].to(embeddings.device)
            negatives = batch['negative_indices'].to(embeddings.device)

            opt.zero_grad()

            emb_upd = model(embeddings, data.edge_index)

            all_idx = torch.cat([anchors, positives, negatives.view(-1)]).unique()

            l_trip = triplet_loss_hyp(
                emb_upd,
                anchors,
                positives,
                negatives,
                cfg.triplet_margin,
                temperature=temperature,
                c=1.0,
            )
            l_lvl = level_radius_loss(emb_upd, all_idx, levels)

            # Compute weighted loss
            if model.learnable_loss_weights:
                w_trip, w_lvl, reg = model.get_loss_weights()  # type: ignore[misc]
                loss = (w_trip * l_trip) + (w_lvl * l_lvl) + reg
                w_trip_log.append(w_trip.item())
                w_lvl_log.append(w_lvl.item())
            else:
                loss = (cfg.w_triplet * l_trip) + (cfg.w_per_level * l_lvl)

            loss.backward()

            nn.utils.clip_grad_norm_(
                list(model.parameters()) + [embeddings], cfg.gradient_clip_norm
            )

            opt.step()

            losses.append(loss.item())
            l_trips.append(l_trip.item())
            l_lvls.append(l_lvl.item())

        sched.step()

        avg = float(np.mean(losses))
        avg_t = float(np.mean(l_trips))
        avg_l = float(np.mean(l_lvls))

        lr = sched.get_last_lr()[0]

        # Compute validation metrics
        model.eval()
        with torch.no_grad():
            emb_for_val = model(embeddings, data.edge_index)

            # Sample a batch for validation metrics
            val_batch = next(iter(loader))
            val_anchors = val_batch['anchor_idx'].to(embeddings.device)
            val_positives = val_batch['positive_idx'].to(embeddings.device)
            val_negatives = val_batch['negative_indices'].to(embeddings.device)

            # Compute validation metrics
            val_metrics = compute_validation_metrics(
                emb_for_val, val_anchors, val_positives, val_negatives, c=1.0, top_k=1
            )
        model.train()

        if epoch in [1, cfg.num_epochs] or (epoch % cfg.epoch_every == 0):
            if model.learnable_loss_weights and w_trip_log:
                avg_w_t = float(np.mean(w_trip_log))
                avg_w_l = float(np.mean(w_lvl_log))
                print(
                    f'[Epoch {epoch:02d}/{cfg.num_epochs}]: L {avg:.4f} | '
                    f'L_t {avg_t:.2f} ({avg_w_t:.2f}) | L_lvl {avg_l:.4f} ({avg_w_l:.2f}) | '
                    f'T {temperature:.3f} | LR {lr:.2e} | '
                    f'D_p {val_metrics["avg_positive_dist"]:.3f} | '
                    f'D_n {val_metrics["avg_negative_dist"]:.3f} | '
                    f'D_s {val_metrics["distance_spread"]:.3f} | '
                    f'Acc {val_metrics["relation_accuracy"]:.2%}'
                )
            else:
                print(
                    f'[Epoch {epoch:02d}/{cfg.num_epochs}]: L {avg:.4f} | '
                    f'L_t {avg_t:.2f} | L_lvl | '
                    f'T {temperature:.3f} | LR {lr:.2e} | '
                    f'D_p {val_metrics["avg_positive_dist"]:.3f} | '
                    f'D_n {val_metrics["avg_negative_dist"]:.3f} | '
                    f'D_s {val_metrics["distance_spread"]:.3f} | '
                    f'Acc {val_metrics["relation_accuracy"]:.2%}'
                )

        log_entry = {
            'stage': stage,
            'epoch': epoch,
            'loss': avg,
            'triplet_loss': avg_t,
            'level_loss': avg_l,
            'temperature': float(temperature),
            'lr': float(lr),
            # Validation metrics
            'val_avg_positive_dist': val_metrics['avg_positive_dist'],
            'val_avg_negative_dist': val_metrics['avg_negative_dist'],
            'val_distance_spread': val_metrics['distance_spread'],
            'val_relation_accuracy': val_metrics['relation_accuracy'],
            'val_mean_positive_rank': val_metrics['mean_positive_rank'],
        }

        if model.learnable_loss_weights and w_trip_log:
            log_entry['w_triplet'] = float(np.mean(w_trip_log))
            log_entry['w_level'] = float(np.mean(w_lvl_log))

        log.append(log_entry)
        best = min(best, avg)

    print(f'\nStage {stage} complete. Best loss: {best:.4f}')

    model.eval()
    with torch.no_grad():
        final_emb = model(embeddings, data.edge_index)

    return final_emb, log


# -------------------------------------------------------------------------------------------------
# Curriculum training
# -------------------------------------------------------------------------------------------------


def train_curriculum(
    curriculum_stages: List[str],
    model: nn.Module,
    embeddings: nn.Parameter,
    data: Data,
    levels: torch.Tensor,
    base_cfg: Config,
    device: torch.device,
    config_file: str = 'conf/config.yaml',
) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
    '''
    Train through all curriculum stages sequentially.

    Args:
        curriculum_stages: List of curriculum stage names (e.g., ['01_graph', '02_graph', ...])
        model: HGCN model
        embeddings: Initial embeddings parameter
        data: PyG Data object with edge_index
        levels: Hierarchy levels tensor
        base_cfg: Base GraphConfig
        device: Device to train on
        config_file: Path to base config file
    '''

    # Issue #9: Decouple config loading from training loop - load all configs upfront
    # For now, use base config for all stages (curriculum configs not yet implemented)
    stage_configs = [base_cfg] * len(curriculum_stages)

    all_logs = []
    current_embeddings = embeddings

    for stage_idx, stage_name in enumerate(curriculum_stages, 1):
        # Use pre-loaded config
        stage_cfg = stage_configs[stage_idx - 1]

        # Create dataloader for this stage
        loader = create_contrastive_dataloader(stage_cfg)
        # Issue #8: Reduced verbose logging
        try:
            dataset_size = len(loader.dataset)  # type: ignore[arg-type]
        except (TypeError, AttributeError):
            dataset_size = 'unknown'
        print(
            f'\nStage {stage_idx}/{len(curriculum_stages)} ({stage_name}): '
            f'{dataset_size} pairs, batch={stage_cfg.batch_size}'
        )

        # Train this stage
        final_emb, stage_log = train_stage(
            stage_idx, model, current_embeddings, data, levels, loader, stage_cfg
        )

        # Update embeddings for next stage
        current_embeddings = nn.Parameter(final_emb.clone())

        # Save stage checkpoint
        outdir = Path(base_cfg.output_dir)
        outdir.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'stage': stage_idx,
            'stage_name': stage_name,
            'state_dict': model.state_dict(),
            'embeddings': final_emb.detach().cpu(),
            'config': stage_cfg.model_dump(),
            'log': stage_log,
        }

        if model.learnable_loss_weights:
            checkpoint['final_log_var_triplet'] = float(model.log_var_triplet.item())  # type: ignore[attr-defined]
            checkpoint['final_log_var_level'] = float(model.log_var_level.item())  # type: ignore[attr-defined]

        torch.save(checkpoint, f'{outdir}/stage_{stage_idx:02d}_{stage_name}_checkpoint.pt')
        print(f'Saved checkpoint: {outdir}/stage_{stage_idx:02d}_{stage_name}_checkpoint.pt')

        all_logs.extend(stage_log)

    return current_embeddings, all_logs


# -------------------------------------------------------------------------------------------------
# Save results
# -------------------------------------------------------------------------------------------------


def save_outputs(
    outdir: str,
    final_emb: torch.Tensor,
    orig_df: pl.DataFrame,
    cfg: Config,
    model: nn.Module,
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
        save_dict['final_log_var_triplet'] = float(model.log_var_triplet.item())  # type: ignore[attr-defined]
        save_dict['final_log_var_level'] = float(model.log_var_level.item())  # type: ignore[attr-defined]

    torch.save(save_dict, f'{outdir}/hgcn_model_final.pt')

    with open(f'{outdir}/training_log.json', 'w') as f:
        json.dump(log, f, indent=2)

    with open(f'{outdir}/config.json', 'w') as f:
        json.dump(cfg.model_dump(), f, indent=2)

    print(f'\n{"=" * 80}')
    print('TRAINING COMPLETE')
    print(f'{"=" * 80}')
    print(f'Best loss: {min(x["loss"] for x in log):.4f}')
    print(f'Final loss: {log[-1]["loss"]:.4f}')

    if model.learnable_loss_weights:
        final_w_t = log[-1].get('w_triplet', 'N/A')
        final_w_l = log[-1].get('w_level', 'N/A')
        if isinstance(final_w_t, float):
            print(f'Final weights: triplet={final_w_t:.4f}, level={final_w_l:.4f}')

    print('\nOutputs:')
    print(f'  Embeddings: {cfg.output_parquet}')
    outdir_path = Path(outdir)
    print(f'  Final model: {outdir_path / "hgcn_model_final.pt"}')
    print(f'  Training log: {outdir_path / "training_log.json"}')
    print(f'  Stage checkpoints: {outdir_path / "stage_*_checkpoint.pt"}')

    return result_df


# -------------------------------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------------------------------


def main(
    config_file: str = 'conf/config.yaml',
    chain_file: Optional[str] = None,
    curriculum_stages: Optional[List[str]] = None,
):
    '''
    Main entry point for curriculum-based HGCN training.

    Args:
        config_file: Path to base config YAML file
        chain_file: Path to chain config file (e.g., 'chain_graph') - overrides curriculum_stages
        curriculum_stages: List of curriculum stage names (e.g., ['01_graph', '02_graph', ...])
    '''
    # Load base configuration
    base_cfg = GraphConfig.from_yaml(config_file)

    # Output directory cleanup / setup
    outdir = setup_directory(base_cfg.output_dir)

    # Determine curriculum stages
    if chain_file:
        # Chain config not yet implemented, fall back to default
        curriculum_stages = ['01_graph', '02_graph', '03_graph', '04_graph', '05_graph', '06_graph']
        print('=' * 80)
        print('HGCN CURRICULUM TRAINING')
        print('=' * 80)
        print('Chain file specified but not yet implemented, using default stages')
        print(f'Curriculum stages: {len(curriculum_stages)}')
    elif curriculum_stages:
        print('=' * 80)
        print('HGCN CURRICULUM TRAINING')
        print('=' * 80)
        print(f'Curriculum stages: {len(curriculum_stages)}')
    else:
        # Default: use all graph curriculum stages
        curriculum_stages = ['01_graph', '02_graph', '03_graph', '04_graph', '05_graph', '06_graph']
        print('=' * 80)
        print('HGCN CURRICULUM TRAINING')
        print('=' * 80)
        print(f'Using default curriculum stages: {len(curriculum_stages)}')

    # Ensure curriculum_stages is a list for type safety
    if curriculum_stages is None:
        curriculum_stages = ['01_graph', '02_graph', '03_graph', '04_graph', '05_graph', '06_graph']

    print(f'Stages: {", ".join(curriculum_stages)}')
    print(f'Output directory: {outdir}')
    print()

    # Set seeds
    torch.manual_seed(base_cfg.seed)
    np.random.seed(base_cfg.seed)

    device = pick_device(base_cfg.device)

    # Load embeddings + levels
    emb, levels, df = load_embeddings(base_cfg.encodings_parquet, device)

    print(f'Loaded embeddings: N={emb.size(0)}, dim={emb.size(1)}')

    # Load relations -> edge_index
    edge_index = load_edge_index(base_cfg.relations_parquet, device)
    data = Data(edge_index=edge_index)

    print(f'Graph edges: {edge_index.size(1)}')

    # Build model
    model = HGCN(
        base_cfg.tangent_dim,
        base_cfg.n_hgcn_layers,
        base_cfg.dropout,
        base_cfg.learnable_curvature,
        base_cfg.learnable_loss_weights,
    ).to(device)

    # Learnable embeddings
    embeddings = nn.Parameter(emb.clone())

    # Train through curriculum
    final_emb, full_log = train_curriculum(
        curriculum_stages, model, embeddings, data, levels, base_cfg, device, config_file
    )

    # Save final outputs
    save_outputs(str(outdir), final_emb, df, base_cfg, model, full_log)


if __name__ == '__main__':
    main()
