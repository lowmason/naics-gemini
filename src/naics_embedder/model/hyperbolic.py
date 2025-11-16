# -------------------------------------------------------------------------------------------------
# Hyperbolic Geometry Utilities
# Shared module for hyperbolic embeddings, distances, and manifold operations
# -------------------------------------------------------------------------------------------------

import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------------------------------------
# Hyperbolic Projection to Lorentz Model
# -------------------------------------------------------------------------------------------------

class HyperbolicProjection(nn.Module):
    '''
    Projects Euclidean embeddings to the Lorentz model of hyperbolic space.
    
    The Lorentz model represents points as (x₀, x₁, ..., xₙ) where:
    - x₀ is the time coordinate (hyperbolic radius)
    - x₁...xₙ are spatial coordinates
    - Constraint: -x₀² + x₁² + ... + xₙ² = -1/c (Lorentz inner product)
    '''
    
    def __init__(self, input_dim: int, curvature: float = 1.0):
        super().__init__()
        
        self.input_dim = input_dim
        self.c = curvature
        
        # Projection layer: maps Euclidean embedding to tangent space
        self.projection = nn.Linear(input_dim, input_dim + 1)
    
    def exp_map_zero(self, v: torch.Tensor) -> torch.Tensor:
        '''
        Exponential map from tangent space at origin to Lorentz hyperboloid.
        
        Args:
            v: Tangent vector of shape (batch_size, input_dim + 1)
        
        Returns:
            Point on Lorentz hyperboloid of shape (batch_size, input_dim + 1)
        '''
        sqrt_c = torch.sqrt(torch.tensor(self.c, device=v.device))
        norm_v = torch.norm(v, p=2, dim=1, keepdim=True)
        norm_v = torch.clamp(norm_v, min=1e-8)
        
        sinh_term = torch.sinh(norm_v / sqrt_c)
        
        # Time coordinate (hyperbolic radius)
        x0 = torch.cosh(norm_v / sqrt_c)
        # Spatial coordinates
        x_rest = (sinh_term * v) / norm_v
        
        return torch.cat([x0, x_rest], dim=1)
    
    def forward(self, euclidean_embedding: torch.Tensor) -> torch.Tensor:
        '''
        Project Euclidean embedding to Lorentz hyperboloid.
        
        Args:
            euclidean_embedding: Euclidean embedding of shape (batch_size, input_dim)
        
        Returns:
            Hyperbolic embedding in Lorentz model of shape (batch_size, input_dim + 1)
        '''
        tangent_vec = self.projection(euclidean_embedding)
        hyperbolic_embedding = self.exp_map_zero(tangent_vec)
        return hyperbolic_embedding


# -------------------------------------------------------------------------------------------------
# Lorentz Distance Computation
# -------------------------------------------------------------------------------------------------

class LorentzDistance(nn.Module):
    '''
    Computes distances in the Lorentz model of hyperbolic space.
    
    Distance between two points u, v on the hyperboloid:
    d(u, v) = √c * arccosh(-⟨u, v⟩_L)
    
    where ⟨u, v⟩_L = u₁v₁ + ... + uₙvₙ - u₀v₀ (Lorentz inner product)
    '''
    
    def __init__(self, curvature: float = 1.0):
        super().__init__()
        self.c = curvature
    
    def lorentz_dot(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        '''
        Compute Lorentz inner product: ⟨u, v⟩_L = Σᵢ uᵢvᵢ - u₀v₀
        
        Args:
            u: First point on hyperboloid, shape (batch_size, embedding_dim+1)
            v: Second point on hyperboloid, shape (batch_size, embedding_dim+1)
        
        Returns:
            Lorentz inner products, shape (batch_size,)
        '''
        uv = u * v
        return torch.sum(uv[:, 1:], dim=1) - uv[:, 0]
    
    def forward(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        '''
        Compute Lorentzian distance between two points.
        
        Args:
            u: First point on hyperboloid, shape (batch_size, embedding_dim+1)
            v: Second point on hyperboloid, shape (batch_size, embedding_dim+1)
        
        Returns:
            Distances, shape (batch_size,)
        '''
        dot_product = self.lorentz_dot(u, v)
        
        # Clamp to ensure valid arccosh argument
        clamped_dot = torch.clamp(dot_product, max=-1.0 - 1e-5)
        
        sqrt_c = torch.sqrt(torch.tensor(self.c, device=u.device))
        dist = sqrt_c * torch.acosh(-clamped_dot)
        
        return dist
    
    def batched_forward(
        self, 
        u: torch.Tensor, 
        v: torch.Tensor
    ) -> torch.Tensor:
        '''
        Batched Lorentz distance computation with broadcasting support.
        
        Args:
            u: Tensor of shape (batch_size, 1, embedding_dim+1) or (batch_size, embedding_dim+1)
            v: Tensor of shape (batch_size, k, embedding_dim+1)
        
        Returns:
            Tensor of shape (batch_size, k) with distances
        '''
        # Ensure u has the right shape for broadcasting
        if u.dim() == 2:
            u = u.unsqueeze(1)  # (batch_size, 1, embedding_dim+1)
        
        # Compute batched Lorentz dot product
        uv = u * v  # (batch_size, k, embedding_dim+1)
        
        # Lorentz dot: sum of spatial components - time component
        dot_product = torch.sum(uv[:, :, 1:], dim=2) - uv[:, :, 0]  # (batch_size, k)
        
        clamped_dot = torch.clamp(dot_product, max=-1.0 - 1e-5)
        
        sqrt_c = torch.sqrt(torch.tensor(self.c, device=u.device))
        dist = sqrt_c * torch.acosh(-clamped_dot)
        
        return dist


# -------------------------------------------------------------------------------------------------
# Hyperbolic Manifold Validation and Diagnostics
# -------------------------------------------------------------------------------------------------

def check_lorentz_manifold_validity(
    embeddings: torch.Tensor,
    curvature: float = 1.0,
    tolerance: float = 1e-3
) -> Tuple[bool, torch.Tensor, torch.Tensor]:
    '''
    Check if embeddings satisfy the Lorentz hyperboloid constraint.
    
    For valid points: -x₀² + x₁² + ... + xₙ² = -1/c
    
    Args:
        embeddings: Hyperbolic embeddings of shape (batch_size, embedding_dim+1)
        curvature: Curvature parameter c
        tolerance: Tolerance for constraint violation
    
    Returns:
        Tuple of:
            - is_valid: Boolean indicating if all points are valid
            - lorentz_norms: Lorentz inner product for each point (should be -1/c)
            - violations: Magnitude of constraint violations
    '''
    # Compute Lorentz inner product with itself: ⟨x, x⟩_L
    time_coord = embeddings[:, 0]  # x₀
    spatial_coords = embeddings[:, 1:]  # x₁...xₙ
    
    spatial_norm_sq = torch.sum(spatial_coords ** 2, dim=1)
    time_norm_sq = time_coord ** 2
    
    lorentz_norms = spatial_norm_sq - time_norm_sq  # Should be -1/c
    
    target_value = -1.0 / curvature
    violations = torch.abs(lorentz_norms - target_value)
    
    is_valid = torch.all(violations < tolerance).item()
    
    return is_valid, lorentz_norms, violations


def compute_hyperbolic_radii(embeddings: torch.Tensor) -> torch.Tensor:
    '''
    Extract hyperbolic radii (time coordinates) from Lorentz embeddings.
    
    The time coordinate x₀ represents the hyperbolic radius (distance from origin).
    
    Args:
        embeddings: Hyperbolic embeddings of shape (batch_size, embedding_dim+1)
    
    Returns:
        Hyperbolic radii of shape (batch_size,)
    '''
    return embeddings[:, 0]


def log_hyperbolic_diagnostics(
    embeddings: torch.Tensor,
    curvature: float = 1.0,
    level_labels: Optional[torch.Tensor] = None,
    logger_instance: Optional[logging.Logger] = None
) -> Dict[str, float]:
    '''
    Log comprehensive diagnostics for hyperbolic embeddings.
    
    Args:
        embeddings: Hyperbolic embeddings of shape (batch_size, embedding_dim+1)
        curvature: Curvature parameter c
        level_labels: Optional NAICS hierarchy level labels for grouped statistics
        logger_instance: Optional logger instance (uses module logger if None)
    
    Returns:
        Dictionary of diagnostic metrics
    '''
    if logger_instance is None:
        logger_instance = logger
    
    # Check manifold validity
    is_valid, lorentz_norms, violations = check_lorentz_manifold_validity(
        embeddings, curvature
    )
    
    # Compute hyperbolic radii
    radii = compute_hyperbolic_radii(embeddings)
    
    diagnostics = {
        'manifold_valid': is_valid,
        'lorentz_norm_mean': lorentz_norms.mean().item(),
        'lorentz_norm_std': lorentz_norms.std().item(),
        'lorentz_norm_min': lorentz_norms.min().item(),
        'lorentz_norm_max': lorentz_norms.max().item(),
        'violation_mean': violations.mean().item(),
        'violation_max': violations.max().item(),
        'radius_mean': radii.mean().item(),
        'radius_std': radii.std().item(),
        'radius_min': radii.min().item(),
        'radius_max': radii.max().item(),
    }
    
    # Log basic diagnostics
    logger_instance.info( 
        f'Hyperbolic Embedding Diagnostics:\n'
        f'  • Manifold valid: {is_valid}\n'
        f'  • Lorentz norm: {diagnostics["lorentz_norm_mean"]:.6f} ± {diagnostics["lorentz_norm_std"]:.6f} '
        f'(target: {-1.0/curvature:.6f})\n'
        f'  • Max violation: {diagnostics["violation_max"]:.6e}\n'
        f'  • Hyperbolic radius: {diagnostics["radius_mean"]:.4f} ± {diagnostics["radius_std"]:.4f}'
    )
    
    # Log per-level statistics if provided
    if level_labels is not None:
        unique_levels = torch.unique(level_labels)
        logger_instance.info('  • Radius by hierarchy level:')
        for level in unique_levels:
            level_mask = (level_labels == level)
            level_radii = radii[level_mask]
            logger_instance.info(
                f'    Level {level.item()}: {level_radii.mean().item():.4f} ± {level_radii.std().item():.4f}'
            )
    
    # Warn if manifold constraint is violated
    if not is_valid:
        logger_instance.warning(
            f'⚠️  Hyperbolic embeddings violate manifold constraint! '
            f'Max violation: {diagnostics["violation_max"]:.6e}'
        )
    
    return diagnostics


# -------------------------------------------------------------------------------------------------
# Lorentz Operations Utility Class
# -------------------------------------------------------------------------------------------------

class LorentzOps:
    '''
    Static utility class for Lorentz model operations.
    Provides functions for mapping between hyperboloid and tangent space, and computing distances.
    '''
    
    @staticmethod
    def log_map_zero(x_hyp: torch.Tensor, c: float = 1.0) -> torch.Tensor:
        '''
        Logarithmic map from hyperboloid to tangent space at origin.
        
        Maps a point on the Lorentz hyperboloid to the tangent space at the origin (1, 0, ..., 0).
        
        Args:
            x_hyp: Point on hyperboloid, shape (batch_size, embedding_dim+1)
            c: Curvature parameter (default: 1.0)
        
        Returns:
            Tangent vector, shape (batch_size, embedding_dim+1)
        '''
        sqrt_c = torch.sqrt(torch.tensor(c, device=x_hyp.device))
        
        # Time coordinate (x0) and spatial coordinates (x1...xn)
        x0 = x_hyp[:, 0:1]  # (batch_size, 1)
        x_spatial = x_hyp[:, 1:]  # (batch_size, embedding_dim)
        
        # Compute norm of spatial part
        norm_spatial = torch.norm(x_spatial, p=2, dim=1, keepdim=True)  # (batch_size, 1)
        norm_spatial = torch.clamp(norm_spatial, min=1e-8)
        
        # Compute distance from origin
        # For point (x0, x_spatial) on hyperboloid: x0^2 - ||x_spatial||^2 = 1/c
        # Distance: d = sqrt(c) * arccosh(sqrt(c) * x0)
        # For log map: v = (sqrt(c) * d / ||x_spatial||) * x_spatial
        d = sqrt_c * torch.acosh(torch.clamp(sqrt_c * x0, min=1.0 + 1e-5))
        
        # Scale factor
        scale = sqrt_c * d / norm_spatial
        scale = torch.where(norm_spatial > 1e-8, scale, torch.zeros_like(scale))
        
        # Tangent vector: time component is 0, spatial components are scaled
        v_spatial = scale * x_spatial
        v_time = torch.zeros_like(x0)
        
        return torch.cat([v_time, v_spatial], dim=1)
    
    @staticmethod
    def exp_map_zero(x_tan: torch.Tensor, c: float = 1.0) -> torch.Tensor:
        '''
        Exponential map from tangent space at origin to hyperboloid.
        
        Maps a tangent vector at the origin to a point on the Lorentz hyperboloid.
        
        Args:
            x_tan: Tangent vector, shape (batch_size, embedding_dim+1)
            c: Curvature parameter (default: 1.0)
        
        Returns:
            Point on hyperboloid, shape (batch_size, embedding_dim+1)
        '''
        sqrt_c = torch.sqrt(torch.tensor(c, device=x_tan.device))
        
        # Time component should be 0 for tangent at origin, but handle general case
        v_time = x_tan[:, 0:1]  # (batch_size, 1)
        v_spatial = x_tan[:, 1:]  # (batch_size, embedding_dim)
        
        # Compute norm of tangent vector (spatial part)
        norm_v = torch.norm(v_spatial, p=2, dim=1, keepdim=True)  # (batch_size, 1)
        norm_v = torch.clamp(norm_v, min=1e-8)
        
        # Exponential map formula
        # x0 = cosh(sqrt(c) * ||v|| / sqrt(c)) = cosh(||v||)
        # x_spatial = (sqrt(c) * sinh(||v||) / ||v||) * v
        x0 = torch.cosh(norm_v / sqrt_c)
        sinh_term = torch.sinh(norm_v / sqrt_c)
        x_spatial = (sinh_term / norm_v) * v_spatial
        
        return torch.cat([x0, x_spatial], dim=1)
    
    @staticmethod
    def lorentz_distance(u: torch.Tensor, v: torch.Tensor, c: float = 1.0) -> torch.Tensor:
        '''
        Compute Lorentzian distance between two points on the hyperboloid.
        
        Args:
            u: First point on hyperboloid, shape (batch_size, embedding_dim+1)
            v: Second point on hyperboloid, shape (batch_size, embedding_dim+1)
            c: Curvature parameter (default: 1.0)
        
        Returns:
            Distances, shape (batch_size,)
        '''
        # Compute Lorentz inner product: ⟨u, v⟩_L = Σᵢ uᵢvᵢ - u₀v₀
        uv = u * v
        dot_product = torch.sum(uv[:, 1:], dim=1) - uv[:, 0]
        
        # Clamp to ensure valid arccosh argument
        clamped_dot = torch.clamp(dot_product, max=-1.0 - 1e-5)
        
        sqrt_c = torch.sqrt(torch.tensor(c, device=u.device))
        dist = sqrt_c * torch.acosh(-clamped_dot)
        
        return dist

