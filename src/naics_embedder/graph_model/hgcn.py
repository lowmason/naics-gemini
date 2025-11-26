# -------------------------------------------------------------------------------------------------
# Imports
# -------------------------------------------------------------------------------------------------

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import polars as pl
import pytorch_lightning as pyl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pytorch_lightning import seed_everything
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops

from naics_embedder.graph_model.dataloader.hgcn_datamodule import HGCNDataModule
from naics_embedder.graph_model.evaluation import compute_validation_metrics
from naics_embedder.text_model.hyperbolic import LorentzOps
from naics_embedder.utils.config import GraphConfig
from naics_embedder.utils.utilities import setup_directory

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
    radial = torch.sqrt(torch.clamp(emb[idx, 0].pow(2) - 1, min=1e-8))
    target = (levels[idx].float() - 2) * 0.5

    # Use Huber loss (smooth_l1) instead of MSE
    # More robust to outliers and prevents loss explosion
    # Quadratic for small errors, linear for large errors
    return F.smooth_l1_loss(radial, target)

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
    ):
        super().__init__()
        self.cfg = cfg
        self.model = HGCN(
            cfg.tangent_dim,
            cfg.n_hgcn_layers,
            cfg.dropout,
            cfg.learnable_curvature,
            cfg.learnable_loss_weights,
        )
        self.embeddings = nn.Parameter(embeddings.clone())
        self.register_buffer('edge_index', edge_index)
        self.register_buffer('levels', levels)
        self._optimizer: Optional[optim.Optimizer] = None
        self._train_losses: List[float] = []
        self._train_triplet: List[float] = []
        self._train_level: List[float] = []
        self._train_w_triplet: List[float] = []
        self._train_w_level: List[float] = []
        self._val_epoch_metrics: List[Dict[str, float]] = []
        self.history: List[Dict[str, Any]] = []
        self.save_hyperparameters(ignore=['embeddings', 'levels', 'edge_index'])

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
        return self.model(self.embeddings, self.edge_index)

    def on_train_epoch_start(self) -> None:
        self._train_losses.clear()
        self._train_triplet.clear()
        self._train_level.clear()
        self._train_w_triplet.clear()
        self._train_w_level.clear()

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        anchors = batch['anchor_idx'].to(self.device)
        positives = batch['positive_idx'].to(self.device)
        negatives = batch['negative_indices'].to(self.device)

        emb_upd = self.forward()
        temperature = self._current_temperature()

        l_trip = triplet_loss_hyp(
            emb_upd,
            anchors,
            positives,
            negatives,
            self.cfg.triplet_margin,
            temperature=temperature,
            c=1.0,
        )

        all_idx = torch.cat([anchors, positives, negatives.view(-1)]).unique()
        l_lvl = level_radius_loss(emb_upd, all_idx, self.levels)

        if self.model.learnable_loss_weights:
            w_trip, w_lvl, reg = self.model.get_loss_weights()  # type: ignore[misc]
            loss = (w_trip * l_trip) + (w_lvl * l_lvl) + reg
            self._train_w_triplet.append(float(w_trip.detach().cpu()))
            self._train_w_level.append(float(w_lvl.detach().cpu()))
        else:
            loss = (self.cfg.w_triplet * l_trip) + (self.cfg.w_per_level * l_lvl)

        self._train_losses.append(float(loss.detach().cpu()))
        self._train_triplet.append(float(l_trip.detach().cpu()))
        self._train_level.append(float(l_lvl.detach().cpu()))

        self.log('train/loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train/triplet_loss', l_trip, on_step=False, on_epoch=True)
        self.log('train/level_loss', l_lvl, on_step=False, on_epoch=True)
        self.log('train/temperature', temperature, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def on_validation_epoch_start(self) -> None:
        self._val_epoch_metrics.clear()

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

        for name, value in metrics.items():
            self.log(
                f'val/{name}',
                value,
                on_step=False,
                on_epoch=True,
                prog_bar=(name == 'relation_accuracy'),
            )

        self._val_epoch_metrics.append({k: float(v.detach().cpu()) for k, v in metrics.items()})

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
        key=lambda name: int(name.replace('hyp_e', '')) if name.replace('hyp_e', '').isdigit() else name,
    )

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
        ).long().to(device)
    )

    # Optionally add self loops (often stabilizes message passing)
    edge_index, _ = add_self_loops(edge_index, num_nodes=int(edge_index.max().item()) + 1)

    return edge_index

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
    edge_index = load_edge_index(base_cfg.relations_parquet, torch.device('cpu'))

    print(f'Loaded embeddings: N={emb.size(0)}, dim={emb.size(1)}')
    print(f'Graph edges: {edge_index.size(1)}')

    datamodule = HGCNDataModule(base_cfg)
    lit_module = HGCNLightningModule(base_cfg, emb, levels, edge_index)

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
