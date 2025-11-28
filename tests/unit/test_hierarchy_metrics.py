import pytest
import torch

from naics_embedder.metrics.hierarchy_structure import (
    compute_hierarchy_retrieval_metrics,
    compute_radius_structure_metrics,
)
from naics_embedder.utils.naics_hierarchy import NaicsHierarchy

def test_radius_metrics_capture_level_statistics():
    hierarchy = NaicsHierarchy([('31', '311'), ('311', '3111')])
    embeddings = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [1.2, 0.0, 0.0],
            [1.4, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    codes = ['31', '311', '3111']

    metrics = compute_radius_structure_metrics(embeddings, codes, hierarchy)
    assert pytest.approx(metrics['radius_mean_level_2'], rel=1e-6) == 1.0
    assert pytest.approx(metrics['radius_mean_level_3'], rel=1e-6) == 1.2
    assert pytest.approx(metrics['radius_mean_level_4'], rel=1e-6) == 1.4
    assert metrics['radius_monotonicity'] == pytest.approx(1.0)
    assert 'radius_level_separation' in metrics

def test_hierarchy_retrieval_metrics_detect_confusion():
    hierarchy = NaicsHierarchy([
        ('31', '311'),
        ('311', '3111'),
        ('311', '3112'),
    ])
    codes = ['31', '311', '3111', '3112']
    distance_matrix = torch.tensor(
        [
            [0.0, 0.2, 0.4, 0.45],
            [0.2, 0.0, 0.25, 0.3],
            [0.4, 0.25, 0.0, 0.1],
            [0.45, 0.3, 0.1, 0.0],
        ],
        dtype=torch.float32,
    )

    metrics = compute_hierarchy_retrieval_metrics(
        distance_matrix,
        codes,
        hierarchy,
        parent_top_k=2,
        child_top_k=2,
    )

    assert metrics['parent_retrieval@2'] == pytest.approx(1.0)
    assert metrics['child_retrieval@2'] == pytest.approx(0.75)
    assert metrics['sibling_confusion_rate'] == pytest.approx(1.0)
