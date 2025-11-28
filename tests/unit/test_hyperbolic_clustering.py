import math
from typing import Tuple

import numpy as np
import pytest
import torch

from naics_embedder.text_model.hyperbolic import LorentzOps, check_lorentz_manifold_validity
from naics_embedder.text_model.hyperbolic_clustering import HyperbolicKMeans

CURVATURE = 1.0

def _generate_hyperbolic_clusters(
    num_clusters: int,
    points_per_cluster: int,
    spatial_dim: int = 3,
    noise: float = 0.02,
    curvature: float = CURVATURE,
    radius: float = 1.5,
    seed: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    '''
    Create synthetic hyperbolic embeddings grouped around evenly spaced centers.
    '''
    device = torch.device('cpu')
    dtype = torch.float32
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    embeddings = []
    labels = []
    denom = max(num_clusters, 1)

    for cluster_idx in range(num_clusters):
        angle = (2.0 * math.pi * cluster_idx) / denom

        center = torch.zeros(spatial_dim, dtype=dtype, device=device)
        center[0] = radius * math.cos(angle)
        if spatial_dim > 1:
            center[1] = radius * math.sin(angle)
        if spatial_dim > 2:
            center[2] = 0.5 * radius * math.cos(2 * angle)

        tangent = torch.zeros(points_per_cluster, spatial_dim + 1, dtype=dtype, device=device)
        noise_sample = noise * torch.randn(
            (points_per_cluster, spatial_dim), generator=generator, dtype=dtype, device=device
        )
        tangent[:, 1:] = center + noise_sample

        embeddings.append(LorentzOps.exp_map_zero(tangent, c=curvature))
        labels.append(
            torch.full((points_per_cluster, ), cluster_idx, dtype=torch.long, device=device)
        )

    return torch.cat(embeddings, dim=0), torch.cat(labels, dim=0)

def _assign_with_lorentz_distance(model: HyperbolicKMeans, embeddings: torch.Tensor) -> np.ndarray:
    '''
    Assign each embedding to the closest cluster center using Lorentz distance.
    '''
    if model.cluster_centers_ is None:
        raise AssertionError('Model must be fitted before computing assignments.')

    assignments = []
    for point in embeddings:
        distances = torch.tensor(
            [
                model.lorentz_distance(point.unsqueeze(0), center.unsqueeze(0)).item()
                for center in model.cluster_centers_
            ]
        )
        assignments.append(int(torch.argmin(distances).item()))
    return np.array(assignments)

@pytest.mark.unit
class TestHyperbolicKMeans:

    def test_initialization_on_hyperboloid(self):
        '''Test cluster centers initialized on Lorentz manifold.'''
        torch.manual_seed(0)
        tangent_vectors = torch.randn(120, 5)
        tangent_vectors[:, 0] = 0.0  # time component unused at origin
        embeddings = LorentzOps.exp_map_zero(tangent_vectors, c=CURVATURE)

        kmeans = HyperbolicKMeans(
            n_clusters=5, curvature=CURVATURE, max_iter=30, tol=1e-4, random_state=42
        )
        kmeans.fit(embeddings)

        assert kmeans.cluster_centers_ is not None
        is_valid, _, violations = check_lorentz_manifold_validity(
            kmeans.cluster_centers_, curvature=CURVATURE
        )
        assert is_valid, f'Centers not on manifold: max violation = {violations.max().item()}'

    def test_cluster_assignment_uses_lorentzian_distance(self):
        '''Test points assigned to nearest cluster by Lorentzian distance.'''
        embeddings, _ = _generate_hyperbolic_clusters(
            num_clusters=3,
            points_per_cluster=25,
            noise=0.01,
            seed=21,
        )

        kmeans = HyperbolicKMeans(
            n_clusters=3,
            curvature=CURVATURE,
            max_iter=40,
            tol=1e-4,
            random_state=7,
        )
        kmeans.fit(embeddings)

        assert kmeans.cluster_centers_ is not None
        assert kmeans.labels_ is not None
        manual_labels = _assign_with_lorentz_distance(kmeans, embeddings)
        assert np.array_equal(manual_labels, kmeans.labels_)

    def test_convergence_on_synthetic_data(self):
        '''Test clustering converges within max_iter.'''
        embeddings, _ = _generate_hyperbolic_clusters(
            num_clusters=4,
            points_per_cluster=35,
            noise=0.015,
            seed=11,
        )

        kmeans = HyperbolicKMeans(
            n_clusters=4, curvature=CURVATURE, max_iter=50, tol=1e-3, random_state=3
        )
        kmeans.fit(embeddings)

        assert kmeans.inertia_ is not None
        assert kmeans.n_iter_ < kmeans.max_iter
        assert kmeans.inertia_ is not None and kmeans.inertia_ >= 0.0

    @pytest.mark.parametrize('n_clusters', [1, 3, 10])
    def test_different_cluster_counts(self, n_clusters):
        '''Test with different cluster counts.'''
        embeddings, _ = _generate_hyperbolic_clusters(
            num_clusters=n_clusters,
            points_per_cluster=20,
            noise=0.01,
            seed=5 + n_clusters,
        )

        kmeans = HyperbolicKMeans(
            n_clusters=n_clusters,
            curvature=CURVATURE,
            max_iter=60,
            tol=1e-3,
            random_state=17,
        )
        kmeans.fit(embeddings)

        assert kmeans.cluster_centers_ is not None
        assert kmeans.labels_ is not None
        assert kmeans.cluster_centers_.shape[0] == n_clusters
        assert kmeans.labels_.shape[0] == embeddings.shape[0]
        assert len(np.unique(kmeans.labels_)) == n_clusters

        is_valid, _, _ = check_lorentz_manifold_validity(
            kmeans.cluster_centers_, curvature=CURVATURE
        )
        assert is_valid
