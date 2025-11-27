# -------------------------------------------------------------------------------------------------
# Hyperbolic Geometry Utilities (#57)
# -------------------------------------------------------------------------------------------------
'''
Shared hyperbolic geometry utilities for Lorentz model operations.

This module provides:
  - ManifoldAdapter: Wrapper for consistent hyperbolic operations
  - Curvature management with phase-aware clamping
  - Manifold validation checks
  - Projection utilities for ensuring embeddings stay on hyperboloid

The Lorentz model represents hyperbolic space on a hyperboloid:
  L^n_c = {x ∈ R^{n+1} : ⟨x, x⟩_L = -1/c, x_0 > 0}

where ⟨x, y⟩_L = -x_0*y_0 + x_1*y_1 + ... + x_n*y_n (Minkowski inner product)
'''

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------------------------------
# Curvature Management
# -------------------------------------------------------------------------------------------------

@dataclass
class CurvatureConfig:
    '''Configuration for curvature management across curriculum phases.'''

    # Phase-specific curvature values
    phase1_curvature: float = 2.0  # High curvature for anchoring
    phase2_curvature: Optional[float] = None  # Learnable
    phase3_curvature: Optional[float] = None  # Learnable
    phase4_curvature: Optional[float] = None  # Learnable

    # Clamping bounds
    min_curvature: float = 0.1
    max_curvature: float = 10.0

    # Initial curvature for learnable phases
    initial_curvature: float = 1.0

class CurvatureManager(nn.Module):
    '''
    Manages curvature parameter with phase-aware behavior.

    In Phase 1: Fixed high curvature
    In Phases 2-4: Learnable curvature with clamping
    '''

    def __init__(self, config: CurvatureConfig) -> None:
        super().__init__()
        self.config = config

        # Learnable curvature parameter
        self._curvature = nn.Parameter(torch.tensor(config.initial_curvature, dtype=torch.float32))

        # Phase tracking
        self._current_phase: int = 1
        self._is_learnable: bool = False

    @property
    def curvature(self) -> Tensor:
        '''Get current curvature value (clamped to valid range).'''
        if self._is_learnable:
            return torch.clamp(
                self._curvature,
                self.config.min_curvature,
                self.config.max_curvature,
            )
        else:
            # Return phase-specific fixed curvature
            if self._current_phase == 1:
                return torch.tensor(
                    self.config.phase1_curvature,
                    device=self._curvature.device,
                    dtype=self._curvature.dtype,
                )
            return self._curvature.detach()

    @property
    def curvature_float(self) -> float:
        '''Get current curvature as float.'''
        return float(self.curvature.item())

    def set_phase(self, phase: int) -> None:
        '''Set current curriculum phase.'''
        self._current_phase = phase

        # Phase 1: Fixed curvature
        # Phases 2-4: Learnable curvature
        self._is_learnable = phase > 1

        if not self._is_learnable:
            # Freeze gradient
            self._curvature.requires_grad_(False)
        else:
            self._curvature.requires_grad_(True)

        logger.debug(
            f'Curvature phase={phase}, learnable={self._is_learnable}, value={self.curvature_float:.4f}'
        )

    def get_state(self) -> dict:
        '''Get state for checkpointing.'''
        return {
            'curvature': self._curvature.item(),
            'phase': self._current_phase,
            'is_learnable': self._is_learnable,
        }

    def load_state(self, state: dict) -> None:
        '''Load state from checkpoint.'''
        self._curvature.data.fill_(state['curvature'])
        self._current_phase = state['phase']
        self._is_learnable = state['is_learnable']

# -------------------------------------------------------------------------------------------------
# Lorentz Operations (Extended)
# -------------------------------------------------------------------------------------------------

class LorentzManifold:
    '''
    Extended Lorentz manifold operations with validation and projection.

    All operations maintain the Lorentz constraint: ⟨x, x⟩_L = -1/c
    '''

    @staticmethod
    def minkowski_dot(x: Tensor, y: Tensor) -> Tensor:
        '''
        Compute Minkowski inner product: ⟨x, y⟩_L = -x_0*y_0 + Σ x_i*y_i

        Args:
            x: First tensor [..., D+1]
            y: Second tensor [..., D+1]

        Returns:
            Inner product [...,]
        '''
        xy = x * y
        return torch.sum(xy[..., 1:], dim=-1) - xy[..., 0]

    @staticmethod
    def lorentz_norm_squared(x: Tensor) -> Tensor:
        '''
        Compute squared Lorentz norm: ⟨x, x⟩_L

        For valid hyperboloid points, this should equal -1/c.
        '''
        return LorentzManifold.minkowski_dot(x, x)

    @staticmethod
    def project_to_hyperboloid(x: Tensor, c: float = 1.0) -> Tensor:
        '''
        Project points onto the Lorentz hyperboloid.

        Ensures the constraint ⟨x, x⟩_L = -1/c is satisfied.

        Args:
            x: Points to project [..., D+1]
            c: Curvature parameter

        Returns:
            Projected points on hyperboloid
        '''
        # Compute spatial norm
        spatial = x[..., 1:]
        spatial_norm_sq = torch.sum(spatial**2, dim=-1, keepdim=True)

        # Compute required time coordinate: x_0 = sqrt(spatial_norm^2 + 1/c)
        x0_new = torch.sqrt(spatial_norm_sq + 1.0 / c)

        return torch.cat([x0_new, spatial], dim=-1)

    @staticmethod
    def check_on_manifold(x: Tensor, c: float = 1.0,
                          tolerance: float = 1e-4) -> Tuple[bool, Tensor]:
        '''
        Check if points lie on the Lorentz hyperboloid.

        Args:
            x: Points to check [..., D+1]
            c: Curvature parameter
            tolerance: Tolerance for constraint violation

        Returns:
            Tuple of (all_valid, violations)
        '''
        lorentz_norm = LorentzManifold.lorentz_norm_squared(x)
        target = -1.0 / c
        violations = torch.abs(lorentz_norm - target)
        all_valid = bool(torch.all(violations < tolerance).item())
        return all_valid, violations

    @staticmethod
    def exp_map_zero(v: Tensor, c: float = 1.0) -> Tensor:
        '''
        Exponential map from tangent space at origin to hyperboloid.

        Args:
            v: Tangent vectors [..., D+1] (time component ignored)
            c: Curvature parameter

        Returns:
            Points on hyperboloid [..., D+1]
        '''
        sqrt_c = torch.sqrt(torch.tensor(c, device=v.device, dtype=v.dtype))

        # Spatial components only
        v_spatial = v[..., 1:]
        norm_v = torch.norm(v_spatial, p=2, dim=-1, keepdim=True)
        norm_v = torch.clamp(norm_v, min=1e-8)

        # Clamp argument to avoid overflow
        theta = torch.clamp(sqrt_c * norm_v, max=40.0)

        # Exponential map formula
        x0 = torch.cosh(theta) / sqrt_c
        sinh_term = torch.sinh(theta) / sqrt_c
        x_spatial = (sinh_term / norm_v) * v_spatial

        return torch.cat([x0, x_spatial], dim=-1)

    @staticmethod
    def log_map_zero(x: Tensor, c: float = 1.0) -> Tensor:
        '''
        Logarithmic map from hyperboloid to tangent space at origin.

        Args:
            x: Points on hyperboloid [..., D+1]
            c: Curvature parameter

        Returns:
            Tangent vectors [..., D+1]
        '''
        sqrt_c = torch.sqrt(torch.tensor(c, device=x.device, dtype=x.dtype))

        x0 = x[..., 0:1]
        x_spatial = x[..., 1:]

        # Distance from origin
        theta = torch.acosh(torch.clamp(sqrt_c * x0, min=1.0 + 1e-5))

        # Scale factor
        sinh_theta = torch.sinh(theta)
        sinh_theta = torch.clamp(sinh_theta, min=1e-8)
        scale = theta / sinh_theta
        scale = torch.where(theta > 1e-8, scale, torch.ones_like(scale))

        # Tangent vector
        v_spatial = scale * x_spatial * sqrt_c
        v_time = torch.zeros_like(x0)

        return torch.cat([v_time, v_spatial], dim=-1)

    @staticmethod
    def distance(x: Tensor, y: Tensor, c: float = 1.0) -> Tensor:
        '''
        Compute geodesic distance on hyperboloid.

        d(x, y) = (1/√c) * arccosh(-c * ⟨x, y⟩_L)

        Args:
            x: First points [..., D+1]
            y: Second points [..., D+1]
            c: Curvature parameter

        Returns:
            Distances [...]
        '''
        dot = LorentzManifold.minkowski_dot(x, y)
        arccosh_arg = torch.clamp(-dot, min=1.0)
        sqrt_c = torch.sqrt(torch.tensor(c, device=x.device, dtype=x.dtype))
        return sqrt_c * torch.acosh(arccosh_arg)

    @staticmethod
    def parallel_transport(v: Tensor, x: Tensor, y: Tensor, c: float = 1.0) -> Tensor:
        '''
        Parallel transport tangent vector v from x to y.

        Args:
            v: Tangent vector at x
            x: Source point on hyperboloid
            y: Target point on hyperboloid
            c: Curvature parameter

        Returns:
            Transported tangent vector at y
        '''
        # Lorentz factor
        alpha = -c * LorentzManifold.minkowski_dot(x, y)
        alpha = torch.clamp(alpha, min=1.0)

        # Transport formula
        coef = c * LorentzManifold.minkowski_dot(y, v) / (alpha + 1)
        return v + coef.unsqueeze(-1) * (x + y)

# -------------------------------------------------------------------------------------------------
# Manifold Adapter
# -------------------------------------------------------------------------------------------------

class ManifoldAdapter(nn.Module):
    '''
    Adapter for consistent hyperbolic operations throughout the model.

    Wraps embeddings and operations to ensure they remain on the hyperboloid,
    with automatic projection and validation.
    '''

    def __init__(
        self,
        curvature_config: Optional[CurvatureConfig] = None,
        validate_manifold: bool = True,
        auto_project: bool = True,
    ) -> None:
        super().__init__()
        self.curvature_manager = CurvatureManager(curvature_config or CurvatureConfig())
        self.validate_manifold = validate_manifold
        self.auto_project = auto_project

        # Statistics tracking
        self._total_projections = 0
        self._total_violations = 0

    @property
    def c(self) -> float:
        '''Get current curvature.'''
        return self.curvature_manager.curvature_float

    def set_phase(self, phase: int) -> None:
        '''Set curriculum phase.'''
        self.curvature_manager.set_phase(phase)

    def to_hyperboloid(self, tangent_vectors: Tensor) -> Tensor:
        '''
        Map tangent vectors to hyperboloid via exponential map.

        Args:
            tangent_vectors: Vectors in tangent space at origin

        Returns:
            Points on hyperboloid
        '''
        c = self.curvature_manager.curvature_float
        x = LorentzManifold.exp_map_zero(tangent_vectors, c=c)

        if self.auto_project:
            x = self.ensure_on_manifold(x)

        return x

    def to_tangent(self, hyperboloid_points: Tensor) -> Tensor:
        '''
        Map hyperboloid points to tangent space via logarithmic map.

        Args:
            hyperboloid_points: Points on hyperboloid

        Returns:
            Vectors in tangent space at origin
        '''
        c = self.curvature_manager.curvature_float
        return LorentzManifold.log_map_zero(hyperboloid_points, c=c)

    def ensure_on_manifold(self, x: Tensor) -> Tensor:
        '''
        Ensure points lie on the hyperboloid, projecting if necessary.

        Args:
            x: Points that should be on hyperboloid

        Returns:
            Points guaranteed to be on hyperboloid
        '''
        c = self.curvature_manager.curvature_float

        if self.validate_manifold:
            is_valid, violations = LorentzManifold.check_on_manifold(x, c=c)

            if not is_valid:
                self._total_violations += int((violations > 1e-4).sum().item())

                if self.auto_project:
                    x = LorentzManifold.project_to_hyperboloid(x, c=c)
                    self._total_projections += 1

        return x

    def distance(self, x: Tensor, y: Tensor) -> Tensor:
        '''Compute geodesic distance.'''
        c = self.curvature_manager.curvature_float
        return LorentzManifold.distance(x, y, c=c)

    def get_diagnostics(self) -> dict:
        '''Get manifold diagnostics.'''
        return {
            'curvature': self.c,
            'total_projections': self._total_projections,
            'total_violations': self._total_violations,
            'phase': self.curvature_manager._current_phase,
            'curvature_learnable': self.curvature_manager._is_learnable,
        }

    def reset_statistics(self) -> None:
        '''Reset tracking statistics.'''
        self._total_projections = 0
        self._total_violations = 0

# -------------------------------------------------------------------------------------------------
# Validation Utilities
# -------------------------------------------------------------------------------------------------

def validate_hyperbolic_embeddings(
    embeddings: Tensor,
    c: float = 1.0,
    tolerance: float = 1e-3,
    raise_on_violation: bool = False,
) -> Tuple[bool, dict]:
    '''
    Validate that embeddings satisfy Lorentz hyperboloid constraints.

    Args:
        embeddings: Embeddings to validate [N, D+1]
        c: Curvature parameter
        tolerance: Tolerance for constraint violation
        raise_on_violation: Raise exception if violations found

    Returns:
        Tuple of (is_valid, diagnostics_dict)
    '''
    is_valid, violations = LorentzManifold.check_on_manifold(embeddings, c, tolerance)

    # Compute statistics
    time_coords = embeddings[:, 0]
    spatial_norms = torch.norm(embeddings[:, 1:], dim=-1)

    diagnostics = {
        'is_valid': is_valid,
        'num_violations': int((violations > tolerance).sum().item()),
        'max_violation': float(violations.max().item()),
        'mean_violation': float(violations.mean().item()),
        'time_coord_min': float(time_coords.min().item()),
        'time_coord_max': float(time_coords.max().item()),
        'time_coord_mean': float(time_coords.mean().item()),
        'spatial_norm_mean': float(spatial_norms.mean().item()),
        'curvature': c,
    }

    if not is_valid:
        msg = (
            f'Hyperbolic embeddings violate manifold constraint: '
            f'{diagnostics["num_violations"]} violations, '
            f'max={diagnostics["max_violation"]:.6f}'
        )
        if raise_on_violation:
            raise ValueError(msg)
        else:
            logger.warning(msg)

    return is_valid, diagnostics
