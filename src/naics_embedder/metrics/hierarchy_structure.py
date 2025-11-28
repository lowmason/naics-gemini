from __future__ import annotations

from collections import defaultdict
from typing import Dict, Optional, Sequence

import torch

from naics_embedder.text_model.hyperbolic import compute_hyperbolic_radii
from naics_embedder.utils.naics_hierarchy import NaicsHierarchy

def compute_radius_structure_metrics(
    embeddings: torch.Tensor,
    codes: Sequence[str],
    hierarchy: Optional[NaicsHierarchy] = None,
) -> Dict[str, float]:
    '''
    Compute per-level radius statistics and hierarchy-aware diagnostics.
    '''
    if embeddings.numel() == 0 or not codes:
        return {}

    radii = compute_hyperbolic_radii(embeddings).detach().cpu()
    per_level = defaultdict(list)
    code_to_radius: Dict[str, float] = {}

    for code, radius in zip(codes, radii):
        if not code:
            continue
        level = len(code)
        radius_value = float(radius.item())
        per_level[level].append(radius_value)
        code_to_radius[code] = radius_value

    metrics: Dict[str, float] = {}
    for level, values in per_level.items():
        tensor_vals = torch.tensor(values, dtype=torch.float32)
        if tensor_vals.numel() == 0:
            continue
        metrics[f'radius_mean_level_{level}'] = float(tensor_vals.mean().item())

    # Level separation (adjacent levels should be well spaced in radius space).
    if len(per_level) >= 2:
        separation_scores = []
        sorted_levels = sorted(per_level.keys())
        for lhs, rhs in zip(sorted_levels, sorted_levels[1:]):
            first = torch.tensor(per_level[lhs], dtype=torch.float32)
            second = torch.tensor(per_level[rhs], dtype=torch.float32)
            if first.numel() == 0 or second.numel() == 0:
                continue
            mean_gap = second.mean() - first.mean()
            p75 = torch.quantile(first, 0.75) if first.numel() > 1 else first.mean()
            p25 = torch.quantile(second, 0.25) if second.numel() > 1 else second.mean()
            overlap = torch.clamp(p75 - p25, min=0.0)
            separation_scores.append(mean_gap - overlap)
        if separation_scores:
            stacked = torch.stack(separation_scores)
            metrics['radius_level_separation'] = float(stacked.mean().item())

    # Radius monotonicity along parent-child paths.
    if hierarchy is not None:
        comparisons = 0
        violations = 0
        for parent, child in hierarchy.parent_child_pairs:
            parent_radius = code_to_radius.get(parent)
            child_radius = code_to_radius.get(child)
            if parent_radius is None or child_radius is None:
                continue
            comparisons += 1
            if child_radius <= parent_radius:
                violations += 1
        if comparisons > 0:
            metrics['radius_monotonicity'] = 1.0 - (violations / comparisons)

    return metrics

def compute_hierarchy_retrieval_metrics(
    distance_matrix: torch.Tensor,
    codes: Sequence[str],
    hierarchy: Optional[NaicsHierarchy],
    *,
    parent_top_k: int = 1,
    child_top_k: int = 5,
) -> Dict[str, float]:
    '''
    Evaluate how well embeddings recover hierarchical neighbors (parents/children/siblings).
    '''
    if hierarchy is None or distance_matrix.numel() == 0 or not codes:
        return {}

    if distance_matrix.device.type != 'cpu':
        distances = distance_matrix.detach().cpu()
    else:
        distances = distance_matrix.detach()

    n = distances.shape[0]
    if n != len(codes):
        raise ValueError('distance matrix size must match number of codes')

    code_to_idx = {code: idx for idx, code in enumerate(codes)}
    metrics: Dict[str, float] = {}

    # Parent retrieval (child -> parent).
    parent_pairs = [
        (code_to_idx[parent], code_to_idx[child]) for parent, child in hierarchy.parent_child_pairs
        if parent in code_to_idx and child in code_to_idx
    ]
    if parent_pairs and parent_top_k > 0:
        hits = 0
        valid = 0
        for parent_idx, child_idx in parent_pairs:
            row = distances[child_idx].clone()
            row[child_idx] = float('inf')
            k = min(parent_top_k, n - 1)
            if k <= 0:
                continue
            top_indices = torch.topk(row, k=k, largest=False).indices.tolist()
            valid += 1
            if parent_idx in top_indices:
                hits += 1
        if valid > 0:
            metrics[f'parent_retrieval@{parent_top_k}'] = hits / valid

    # Child retrieval (parent -> children recall@k).
    if child_top_k > 0:
        per_parent_scores = []
        for parent_code, children in hierarchy.children_by_parent.items():
            parent_idx = code_to_idx.get(parent_code)
            if parent_idx is None:
                continue
            child_indices = [code_to_idx[c] for c in children if c in code_to_idx]
            if not child_indices:
                continue
            row = distances[parent_idx].clone()
            row[parent_idx] = float('inf')
            k = min(child_top_k, n - 1)
            if k <= 0:
                continue
            top_indices = torch.topk(row, k=k, largest=False).indices.tolist()
            hits = sum(1 for idx in child_indices if idx in top_indices)
            per_parent_scores.append(hits / len(child_indices))
        if per_parent_scores:
            metrics[f'child_retrieval@{child_top_k}'] = float(
                sum(per_parent_scores) / len(per_parent_scores)
            )

    # Sibling confusion: nearest neighbor is a sibling instead of parent/ancestor.
    sibling_events = []
    for code, idx in code_to_idx.items():
        siblings = hierarchy.get_siblings(code)
        sibling_indices = [code_to_idx[c] for c in siblings if c in code_to_idx]
        if not sibling_indices:
            continue
        row = distances[idx].clone()
        row[idx] = float('inf')
        nearest = torch.argmin(row).item()
        sibling_events.append(1.0 if nearest in sibling_indices else 0.0)
    if sibling_events:
        metrics['sibling_confusion_rate'] = float(sum(sibling_events) / len(sibling_events))

    return metrics
