# -------------------------------------------------------------------------------------------------
# Hyperbolic K-Means Clustering for Lorentz Model
# Implements K-Means clustering directly in hyperbolic space using Lorentzian distances
# -------------------------------------------------------------------------------------------------

import logging
from typing import Optional

import numpy as np
import torch

from naics_embedder.text_model.hyperbolic import LorentzDistance

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------------------------------
# Hyperbolic K-Means Clustering
# -------------------------------------------------------------------------------------------------

class HyperbolicKMeans:
    '''
    K-Means clustering algorithm compatible with the Lorentz model of hyperbolic space.

    This implementation performs clustering directly in hyperbolic space using:
    - Lorentzian distances for assignment
    - Fréchet mean (hyperbolic centroid) for center updates

    Key differences from Euclidean K-Means:
    1. Distances computed using Lorentzian geodesic distance
    2. Centroids computed using Fréchet mean on hyperboloid
    3. Initialization ensures centers are on the hyperboloid
    '''

    def __init__(
        self,
        n_clusters: int,
        curvature: float = 1.0,
        max_iter: int = 100,
        tol: float = 1e-4,
        random_state: Optional[int] = None,
        verbose: bool = False,
    ):
        '''
        Initialize Hyperbolic K-Means.

        Args:
            n_clusters: Number of clusters
            curvature: Curvature parameter for Lorentz model (default: 1.0)
            max_iter: Maximum number of iterations (default: 100)
            tol: Convergence tolerance (default: 1e-4)
            random_state: Random seed for initialization (default: None)
            verbose: Whether to print convergence messages (default: False)
        '''
        self.n_clusters = n_clusters
        self.curvature = curvature
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.verbose = verbose

        # Will be set during fit
        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None
        self.n_iter_ = 0

        # Initialize distance computation
        self.lorentz_distance = LorentzDistance(curvature=curvature)

    def _initialize_centers(self, embeddings: torch.Tensor, n_clusters: int) -> torch.Tensor:
        '''
        Initialize cluster centers on the hyperboloid.

        Uses k-means++ style initialization adapted for hyperbolic space:
        1. Select first center randomly
        2. Select subsequent centers with probability proportional to squared distance
           to nearest existing center

        Args:
            embeddings: Hyperbolic embeddings (N, D+1) on hyperboloid
            n_clusters: Number of clusters

        Returns:
            Initial cluster centers (n_clusters, D+1) on hyperboloid
        '''
        device = embeddings.device
        n_samples, embedding_dim = embeddings.shape

        # Initialize random number generator
        if self.random_state is not None:
            torch.manual_seed(self.random_state)

        # Select first center randomly
        centers = [embeddings[torch.randint(0, n_samples, (1, ), device=device)]]

        # Select remaining centers using k-means++ strategy
        for _ in range(n_clusters - 1):
            # Compute distances from all points to nearest existing center
            distances_to_centers = []
            for point in embeddings:
                # Compute distances to all existing centers
                point_distances = []
                for center in centers:
                    center_point = center if center.dim() == 1 else center.squeeze(0)
                    dist = self.lorentz_distance(point.unsqueeze(0), center_point.unsqueeze(0))
                    point_distances.append(dist.item())
                # Distance to nearest center
                min_dist = min(point_distances)
                distances_to_centers.append(min_dist**2)  # Squared distance

            # Convert to probabilities (proportional to squared distance)
            distances_to_centers = torch.tensor(distances_to_centers, device=device)
            probabilities = distances_to_centers / distances_to_centers.sum()

            # Sample next center
            idx = torch.multinomial(probabilities, 1)
            centers.append(embeddings[idx])

        return torch.cat(centers, dim=0)

    def _frechet_mean(
        self,
        points: torch.Tensor,
        initial_guess: Optional[torch.Tensor] = None,
        max_iter: int = 10
    ) -> torch.Tensor:
        '''
        Compute Fréchet mean (hyperbolic centroid) of points on hyperboloid.

        The Fréchet mean minimizes the sum of squared Lorentzian distances.
        We use an iterative algorithm:
        1. Project points to tangent space at current estimate
        2. Compute Euclidean mean in tangent space
        3. Project back to hyperboloid via exponential map
        4. Repeat until convergence

        Args:
            points: Points on hyperboloid (N, D+1)
            initial_guess: Initial guess for mean (default: None, uses first point)
            max_iter: Maximum iterations for Fréchet mean computation (default: 10)

        Returns:
            Fréchet mean on hyperboloid (1, D+1)
        '''
        if points.shape[0] == 1:
            return points

        # Initialize with first point or provided guess
        if initial_guess is not None:
            current_mean = initial_guess.clone()
        else:
            current_mean = points[0:1].clone()

        # Iterative Fréchet mean computation
        for _ in range(max_iter):
            # Project all points to tangent space at current mean
            # Use log_map from current_mean to each point
            tangent_vectors = []
            for point in points:
                # Compute log map from current_mean to point
                # This requires parallel transport and log map
                # Simplified: use log_map_zero and adjust
                # For more accuracy, we can use the full parallel transport
                # For now, use a simpler approximation
                tangent_vec = self._log_map(current_mean, point)
                tangent_vectors.append(tangent_vec)

            tangent_vectors = torch.cat(tangent_vectors, dim=0)

            # Compute Euclidean mean in tangent space
            tangent_mean = tangent_vectors.mean(dim=0, keepdim=True)

            # Project back to hyperboloid via exponential map
            new_mean = self._exp_map(current_mean, tangent_mean)

            # Check convergence
            dist = self.lorentz_distance(current_mean, new_mean).item()
            if dist < self.tol:
                break

            current_mean = new_mean

        return current_mean

    def _log_map(self, base: torch.Tensor, point: torch.Tensor) -> torch.Tensor:
        '''
        Logarithmic map from hyperboloid to tangent space at base point.

        Maps a point on the hyperboloid to the tangent space at the base point.
        This is a simplified version - for full accuracy, parallel transport is needed.

        Args:
            base: Base point on hyperboloid (1, D+1)
            point: Point on hyperboloid (1, D+1)

        Returns:
            Tangent vector (1, D+1)
        '''
        # Simplified log map: project to tangent space at origin, then parallel transport
        # For efficiency, we use an approximation
        # Full implementation would use parallel transport

        # Compute distance and direction
        dist = self.lorentz_distance(base, point)

        # Direction vector in embedding space
        direction = point - base

        # Project to tangent space (simplified - assumes base is near origin)
        # For more accuracy, we'd need proper parallel transport
        # This approximation works well when base points are not too far from origin
        direction_norm = torch.norm(direction, dim=1, keepdim=True) + 1e-8
        tangent_vec = direction * dist.unsqueeze(0) / direction_norm

        return tangent_vec

    def _exp_map(self, base: torch.Tensor, tangent_vec: torch.Tensor) -> torch.Tensor:
        '''
        Exponential map from tangent space at base point to hyperboloid.

        Maps a tangent vector at the base point to a point on the hyperboloid.

        Args:
            base: Base point on hyperboloid (1, D+1)
            tangent_vec: Tangent vector (1, D+1)

        Returns:
            Point on hyperboloid (1, D+1)
        '''
        # Simplified exp map: use exponential map at origin, then parallel transport
        # For efficiency, we use an approximation
        # Full implementation would use parallel transport

        # For now, use a simplified version that projects to origin and back
        # This works reasonably well for points not too far from origin
        norm_tangent = torch.norm(tangent_vec[:, 1:], dim=1, keepdim=True)
        norm_tangent = torch.clamp(norm_tangent, min=1e-8)

        sqrt_c = torch.sqrt(torch.tensor(self.curvature, device=base.device))

        # Exponential map at origin (simplified)
        x0 = torch.cosh(norm_tangent / sqrt_c)
        x_spatial = (torch.sinh(norm_tangent / sqrt_c) / norm_tangent) * tangent_vec[:, 1:]

        # Combine
        new_point = torch.cat([x0, x_spatial], dim=1)

        # Ensure it's on hyperboloid (project if needed)
        new_point = self._project_to_hyperboloid(new_point)

        return new_point

    def _project_to_hyperboloid(self, point: torch.Tensor) -> torch.Tensor:
        '''
        Project a point to the hyperboloid to satisfy Lorentz constraint.

        Ensures: -x₀² + x₁² + ... + xₙ² = -1/c

        Args:
            point: Point near hyperboloid (1, D+1)

        Returns:
            Projected point on hyperboloid (1, D+1)
        '''
        point[:, 0:1]
        x_spatial = point[:, 1:]

        # Compute spatial norm
        spatial_norm_sq = torch.sum(x_spatial**2, dim=1, keepdim=True)

        # Compute required time coordinate to satisfy constraint
        # -x₀² + ||x_spatial||² = -1/c
        # x₀² = ||x_spatial||² + 1/c
        # x₀ = sqrt(||x_spatial||² + 1/c)
        target_x0_sq = spatial_norm_sq + 1.0 / self.curvature
        target_x0 = torch.sqrt(torch.clamp(target_x0_sq, min=1.0 / self.curvature))

        # Normalize spatial coordinates to match
        # If spatial norm is too small, use default direction
        spatial_norm = torch.sqrt(torch.clamp(spatial_norm_sq, min=1e-8))
        x_spatial_normalized = x_spatial / spatial_norm

        # Scale spatial coordinates to match target x0
        # We need: -x₀² + ||x_spatial||² = -1/c
        # So: ||x_spatial||² = x₀² - 1/c
        target_spatial_norm_sq = target_x0**2 - 1.0 / self.curvature
        target_spatial_norm = torch.sqrt(torch.clamp(target_spatial_norm_sq, min=0.0))

        x_spatial_scaled = x_spatial_normalized * target_spatial_norm

        return torch.cat([target_x0, x_spatial_scaled], dim=1)

    def fit(self, embeddings: torch.Tensor) -> 'HyperbolicKMeans':
        '''
        Fit the K-Means model to hyperbolic embeddings.

        Args:
            embeddings: Hyperbolic embeddings (N, D+1) on hyperboloid

        Returns:
            self
        '''
        device = embeddings.device
        n_samples = embeddings.shape[0]

        if n_samples < self.n_clusters:
            raise ValueError(
                f'Number of samples ({n_samples}) must be >= n_clusters ({self.n_clusters})'
            )

        # Initialize centers
        centers = self._initialize_centers(embeddings, self.n_clusters)

        # Main K-Means loop
        for iteration in range(self.max_iter):
            # Assign points to nearest centers
            # Compute distances from all points to all centers
            distances = torch.zeros(n_samples, self.n_clusters, device=device)
            for i, point in enumerate(embeddings):
                for j, center in enumerate(centers):
                    dist = self.lorentz_distance(point.unsqueeze(0), center.unsqueeze(0))
                    distances[i, j] = dist.item()

            # Assign to nearest center
            labels = distances.argmin(dim=1)

            # Update centers using Fréchet mean
            new_centers = []
            for cluster_id in range(self.n_clusters):
                cluster_points = embeddings[labels == cluster_id]

                if len(cluster_points) == 0:
                    # Empty cluster: reinitialize randomly
                    new_centers.append(
                        embeddings[torch.randint(0, n_samples, (1, ), device=device)]
                    )
                else:
                    # Compute Fréchet mean
                    cluster_mean = self._frechet_mean(
                        cluster_points, initial_guess=centers[cluster_id:cluster_id + 1]
                    )
                    new_centers.append(cluster_mean)

            new_centers = torch.cat(new_centers, dim=0)

            # Check convergence
            center_shift = torch.tensor(0.0, device=device)
            for old_center, new_center in zip(centers, new_centers):
                shift = self.lorentz_distance(old_center.unsqueeze(0),
                                              new_center.unsqueeze(0)).item()
                center_shift = max(center_shift, shift)

            centers = new_centers

            if self.verbose:
                logger.info(
                    f'Hyperbolic K-Means iteration {iteration + 1}/{self.max_iter}: '
                    f'max center shift = {center_shift:.6f}'
                )

            if center_shift < self.tol:
                if self.verbose:
                    logger.info(f'Converged after {iteration + 1} iterations')
                break

        # Store results
        self.cluster_centers_ = centers
        self.labels_ = labels.cpu().numpy()
        self.n_iter_ = iteration + 1

        # Compute inertia (sum of squared distances to nearest center)
        inertia = 0.0
        for i, point in enumerate(embeddings):
            center = centers[labels[i]]
            dist = self.lorentz_distance(point.unsqueeze(0), center.unsqueeze(0)).item()
            inertia += dist**2
        self.inertia_ = inertia

        return self

    def fit_predict(self, embeddings: torch.Tensor) -> np.ndarray:
        '''
        Fit the model and return cluster labels.

        Args:
            embeddings: Hyperbolic embeddings (N, D+1) on hyperboloid

        Returns:
            Cluster labels (N,)
        '''
        self.fit(embeddings)
        if self.labels_ is None:
            raise RuntimeError('Labels not set after fit')
        return self.labels_

    def predict(self, embeddings: torch.Tensor) -> np.ndarray:
        '''
        Predict cluster labels for new embeddings.

        Args:
            embeddings: Hyperbolic embeddings (N, D+1) on hyperboloid

        Returns:
            Cluster labels (N,)
        '''
        if self.cluster_centers_ is None:
            raise ValueError('Model must be fitted before prediction')

        device = embeddings.device
        n_samples = embeddings.shape[0]

        # Compute distances to all centers
        distances = torch.zeros(n_samples, self.n_clusters, device=device)
        for i, point in enumerate(embeddings):
            for j, center in enumerate(self.cluster_centers_):
                dist = self.lorentz_distance(point.unsqueeze(0), center.unsqueeze(0))
                distances[i, j] = dist.item()

        # Assign to nearest center
        labels = distances.argmin(dim=1).cpu().numpy()

        return labels
