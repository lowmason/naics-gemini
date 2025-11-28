from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import polars as pl
import torch

from naics_embedder.graph_model.hgcn import load_embeddings
from naics_embedder.text_model.evaluation import EmbeddingEvaluator, HierarchyMetrics
from naics_embedder.utils.distance_matrix import load_distance_submatrix

logger = logging.getLogger(__name__)

@dataclass
class Stage4VerificationConfig:
    '''
    Thresholds for verifying that HGCN refinement preserves global structure.
    '''

    max_cophenetic_degradation: float = 0.02
    max_ndcg_degradation: float = 0.01
    min_local_improvement: float = 0.05
    ndcg_k: int = 10
    parent_top_k: int = 1

def _load_parent_pairs(relations_path: Path, code_to_idx: Dict[str, int]) -> List[Tuple[int, int]]:
    df = pl.read_parquet(relations_path).filter(pl.col('relation') == 'child'
                                                ).select('code_i', 'code_j')
    pairs: Dict[int, int] = {}
    for row in df.iter_rows(named=True):
        parent_code = row['code_i']
        child_code = row['code_j']
        if parent_code not in code_to_idx or child_code not in code_to_idx:
            continue
        child_idx = code_to_idx[child_code]
        parent_idx = code_to_idx[parent_code]
        pairs[child_idx] = parent_idx
    return list(pairs.items())

def _parent_retrieval_accuracy(
    dist_matrix: torch.Tensor,
    parent_pairs: Sequence[Tuple[int, int]],
    top_k: int = 1,
) -> float:
    if not parent_pairs:
        return 0.0

    hits = 0
    for child_idx, parent_idx in parent_pairs:
        row = dist_matrix[child_idx].clone()
        row[child_idx] = float('inf')
        k = min(top_k, row.shape[0] - 1)
        top_indices = torch.topk(row, k=k, largest=False).indices
        if int(parent_idx) in top_indices.tolist():
            hits += 1
    return hits / len(parent_pairs)

def _to_float(value: float | torch.Tensor) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu())
    return float(value)

def _compute_global_metrics(
    embeddings: torch.Tensor,
    tree_distances: torch.Tensor,
    ndcg_k: int,
    parent_pairs: Sequence[Tuple[int, int]],
    evaluator: EmbeddingEvaluator,
    hierarchy: HierarchyMetrics,
    top_k: int,
) -> Dict[str, float]:
    with torch.no_grad():
        emb_dists = evaluator.compute_pairwise_distances(
            embeddings, metric='lorentz', curvature=1.0
        )

    cophenetic = hierarchy.cophenetic_correlation(emb_dists, tree_distances)
    ndcg = hierarchy.ndcg_ranking(emb_dists, tree_distances, k_values=[ndcg_k])
    parent_retrieval = _parent_retrieval_accuracy(emb_dists, parent_pairs, top_k=top_k)

    metrics = {
        'cophenetic_correlation': _to_float(cophenetic['correlation']),
        f'ndcg@{ndcg_k}': _to_float(ndcg[f'ndcg@{ndcg_k}']),
        f'parent_retrieval@{top_k}': parent_retrieval,
    }
    return metrics

def _load_embeddings_with_codes(parquet_path: Path,
                                ) -> Tuple[torch.Tensor, List[str], pl.DataFrame]:
    embeddings, _levels, df = load_embeddings(str(parquet_path), torch.device('cpu'))
    codes = df['code'].to_list()
    return embeddings, codes, df

def _align_embeddings(
    embeddings: torch.Tensor,
    codes: List[str],
    reference_codes: List[str],
) -> torch.Tensor:
    if codes == reference_codes:
        return embeddings

    code_to_idx = {code: idx for idx, code in enumerate(codes)}
    order = [code_to_idx[code] for code in reference_codes]
    return embeddings[torch.tensor(order, dtype=torch.long)]

def verify_stage4(
    stage3_parquet: Path,
    stage4_parquet: Path,
    distance_matrix: Path,
    relations_parquet: Path,
    config: Stage4VerificationConfig,
) -> Dict:
    stage3_parquet = stage3_parquet.expanduser()
    stage4_parquet = stage4_parquet.expanduser()
    distance_matrix = distance_matrix.expanduser()
    relations_parquet = relations_parquet.expanduser()

    if not stage3_parquet.exists():
        raise FileNotFoundError(f'Stage 3 embeddings not found: {stage3_parquet}')
    if not stage4_parquet.exists():
        raise FileNotFoundError(f'Stage 4 embeddings not found: {stage4_parquet}')

    emb_stage3, codes_pre, df_pre = _load_embeddings_with_codes(stage3_parquet)
    emb_stage4, codes_post, df_post = _load_embeddings_with_codes(stage4_parquet)

    if df_pre.shape[0] != df_post.shape[0]:
        raise ValueError('Stage 3 and Stage 4 embeddings have different row counts')

    emb_stage4 = _align_embeddings(emb_stage4, codes_post, codes_pre)
    codes = codes_pre

    tree_distances = load_distance_submatrix(distance_matrix, codes)
    code_to_idx = {code: idx for idx, code in enumerate(codes)}
    parent_pairs = _load_parent_pairs(relations_parquet, code_to_idx)

    evaluator = EmbeddingEvaluator()
    hierarchy = HierarchyMetrics()

    pre_metrics = _compute_global_metrics(
        emb_stage3,
        tree_distances,
        config.ndcg_k,
        parent_pairs,
        evaluator,
        hierarchy,
        config.parent_top_k,
    )
    post_metrics = _compute_global_metrics(
        emb_stage4,
        tree_distances,
        config.ndcg_k,
        parent_pairs,
        evaluator,
        hierarchy,
        config.parent_top_k,
    )

    ndcg_key = f'ndcg@{config.ndcg_k}'
    parent_key = f'parent_retrieval@{config.parent_top_k}'

    delta = {
        'cophenetic_correlation':
        post_metrics['cophenetic_correlation'] - pre_metrics['cophenetic_correlation'],
        ndcg_key:
        post_metrics[ndcg_key] - pre_metrics[ndcg_key],
        parent_key:
        post_metrics[parent_key] - pre_metrics[parent_key],
    }

    checks = {
        'cophenetic': delta['cophenetic_correlation'] >= -config.max_cophenetic_degradation,
        'ndcg': delta[ndcg_key] >= -config.max_ndcg_degradation,
        'local_improvement': delta[parent_key] >= config.min_local_improvement,
    }

    return {
        'pre': pre_metrics,
        'post': post_metrics,
        'delta': delta,
        'checks': checks,
        'thresholds': asdict(config),
        'passed': all(checks.values()),
        'codes': codes,
    }
