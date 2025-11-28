import polars as pl
import torch

from naics_embedder.graph_model.hgcn import HGCNLightningModule
from naics_embedder.utils.config import GraphConfig

def _lorentz_points(spatial):
    spatial_tensor = torch.tensor(spatial, dtype=torch.float32)
    time = torch.sqrt(1.0 + torch.sum(spatial_tensor**2, dim=1, keepdim=True))
    return torch.cat([time, spatial_tensor], dim=1)

def test_hgcn_full_eval_metrics(tmp_path):
    codes = ['11', '111', '112']
    matrix = torch.tensor(
        [
            [0.0, 1.0, 2.0],
            [1.0, 0.0, 1.0],
            [2.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    ).numpy()
    columns = {}
    for idx, code in enumerate(codes):
        columns[f'idx_{idx}-code_{code}'] = matrix[:, idx]
    distance_df = pl.DataFrame(columns)
    distance_path = tmp_path / 'distance.parquet'
    distance_df.write_parquet(distance_path)

    cfg = GraphConfig(
        distance_matrix_parquet=str(distance_path),
        ndcg_k_values=[2],
        full_eval_frequency=1,
        tangent_dim=3,
        n_hgcn_layers=1,
        dropout=0.0,
        learnable_curvature=False,
        learnable_loss_weights=False,
        k_total=1,
        n_positive_samples=1,
        batch_size=1,
    )

    embeddings = _lorentz_points([[0.0, 0.0], [0.1, 0.0], [0.0, 0.1]])
    levels = torch.tensor([2, 3, 3], dtype=torch.long)
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    edge_types = torch.zeros(edge_index.size(1), dtype=torch.long)
    edge_weights = torch.ones(edge_index.size(1), dtype=torch.float32)
    edge_meta = {'edge_type_count': 1, 'sibling_type_id': None}
    node_metadata = pl.DataFrame({'code': codes})

    module = HGCNLightningModule(
        cfg, embeddings.clone(), levels, edge_index, edge_types, edge_weights, edge_meta,
        node_metadata
    )

    assert module.tree_distances is not None
    assert module._should_run_full_eval(batch_idx=0) is True

    metrics = module._compute_full_validation_metrics(module.forward())
    assert metrics is not None
    assert 'cophenetic_correlation' in metrics
    assert 'ndcg@2' in metrics
