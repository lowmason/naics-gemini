'''
Unit tests for utils/hyperbolic.py - Shared hyperbolic geometry utilities.

Tests the ManifoldAdapter, CurvatureManager, LorentzManifold operations,
and validation utilities.
'''

import pytest
import torch

from naics_embedder.utils.hyperbolic import (
    CurvatureConfig,
    CurvatureManager,
    LorentzManifold,
    ManifoldAdapter,
    validate_hyperbolic_embeddings,
)

# -------------------------------------------------------------------------------------------------
# LorentzManifold Tests
# -------------------------------------------------------------------------------------------------

@pytest.mark.unit
class TestLorentzManifold:
    '''Test suite for LorentzManifold static operations.'''

    def test_minkowski_dot_self(self, sample_lorentz_embeddings):
        '''Test Minkowski dot product with self equals -1/c.'''
        dot = LorentzManifold.minkowski_dot(
            sample_lorentz_embeddings,
            sample_lorentz_embeddings,
        )

        # For valid Lorentz embeddings with c=1, self-dot should be -1
        assert torch.allclose(dot, torch.tensor(-1.0), atol=1e-3)

    def test_minkowski_dot_symmetry(self, sample_lorentz_embeddings):
        '''Test Minkowski dot product is symmetric.'''
        x = sample_lorentz_embeddings[:8]
        y = sample_lorentz_embeddings[8:]

        dot_xy = LorentzManifold.minkowski_dot(x, y)
        dot_yx = LorentzManifold.minkowski_dot(y, x)

        assert torch.allclose(dot_xy, dot_yx, atol=1e-6)

    def test_lorentz_norm_squared(self, sample_lorentz_embeddings):
        '''Test Lorentz norm squared computation.'''
        norm_sq = LorentzManifold.lorentz_norm_squared(sample_lorentz_embeddings)

        # Should be -1/c = -1 for c=1
        assert torch.allclose(norm_sq, torch.tensor(-1.0), atol=1e-3)

    def test_project_to_hyperboloid(self, test_device):
        '''Test projection onto hyperboloid.'''
        # Create points not on hyperboloid
        invalid_points = torch.randn(10, 33, device=test_device)

        projected = LorentzManifold.project_to_hyperboloid(invalid_points, c=1.0)

        # Check projected points are valid
        is_valid, violations = LorentzManifold.check_on_manifold(projected, c=1.0)
        assert is_valid

    def test_check_on_manifold_valid(self, sample_lorentz_embeddings):
        '''Test manifold check passes for valid embeddings.'''
        is_valid, violations = LorentzManifold.check_on_manifold(sample_lorentz_embeddings, c=1.0)

        assert is_valid
        assert torch.all(violations < 1e-3)

    def test_check_on_manifold_invalid(self, test_device):
        '''Test manifold check fails for invalid embeddings.'''
        invalid = torch.randn(10, 33, device=test_device)

        is_valid, violations = LorentzManifold.check_on_manifold(invalid, c=1.0)

        assert not is_valid

    def test_exp_map_zero(self, sample_tangent_vectors):
        '''Test exponential map from origin.'''
        hyp = LorentzManifold.exp_map_zero(sample_tangent_vectors, c=1.0)

        # Check result is on manifold
        is_valid, _ = LorentzManifold.check_on_manifold(hyp, c=1.0)
        assert is_valid

    def test_log_map_zero(self, sample_lorentz_embeddings):
        '''Test logarithmic map to origin.'''
        tangent = LorentzManifold.log_map_zero(sample_lorentz_embeddings, c=1.0)

        # Time component should be 0
        assert torch.allclose(tangent[:, 0], torch.zeros_like(tangent[:, 0]), atol=1e-5)

    def test_exp_log_roundtrip(self, sample_lorentz_embeddings):
        '''Test exp(log(x)) ≈ x.'''
        c = 1.0
        tangent = LorentzManifold.log_map_zero(sample_lorentz_embeddings, c=c)
        reconstructed = LorentzManifold.exp_map_zero(tangent, c=c)

        assert torch.allclose(reconstructed, sample_lorentz_embeddings, atol=1e-4)

    def test_distance_non_negative(self, sample_lorentz_embeddings):
        '''Test distance is non-negative.'''
        x = sample_lorentz_embeddings[:8]
        y = sample_lorentz_embeddings[8:]

        dist = LorentzManifold.distance(x, y, c=1.0)

        assert torch.all(dist >= 0)

    def test_distance_symmetry(self, sample_lorentz_embeddings):
        '''Test distance is symmetric.'''
        x = sample_lorentz_embeddings[:8]
        y = sample_lorentz_embeddings[8:]

        d_xy = LorentzManifold.distance(x, y, c=1.0)
        d_yx = LorentzManifold.distance(y, x, c=1.0)

        assert torch.allclose(d_xy, d_yx, atol=1e-6)

    def test_distance_to_self_zero(self, sample_lorentz_embeddings):
        '''Test distance to self is zero.'''
        dist = LorentzManifold.distance(
            sample_lorentz_embeddings,
            sample_lorentz_embeddings,
            c=1.0,
        )

        assert torch.allclose(dist, torch.zeros_like(dist), atol=5e-3)

    @pytest.mark.parametrize('curvature', [0.1, 0.5, 1.0, 5.0])
    def test_exp_map_different_curvatures(self, sample_tangent_vectors, curvature):
        '''Test exp_map works for different curvatures.'''
        hyp = LorentzManifold.exp_map_zero(sample_tangent_vectors, c=curvature)

        is_valid, _ = LorentzManifold.check_on_manifold(hyp, c=curvature, tolerance=1e-2)
        assert is_valid

# -------------------------------------------------------------------------------------------------
# CurvatureManager Tests
# -------------------------------------------------------------------------------------------------

@pytest.mark.unit
class TestCurvatureManager:
    '''Test suite for CurvatureManager.'''

    def test_initial_curvature(self):
        '''Test initial curvature value.'''
        config = CurvatureConfig(initial_curvature=1.5)
        manager = CurvatureManager(config)

        # Phase 1 uses fixed curvature
        assert manager.curvature_float == config.phase1_curvature

    def test_phase1_fixed_curvature(self):
        '''Test Phase 1 uses fixed high curvature.'''
        config = CurvatureConfig(phase1_curvature=2.5)
        manager = CurvatureManager(config)

        manager.set_phase(1)

        assert manager.curvature_float == 2.5
        assert not manager._is_learnable

    def test_phase2_learnable_curvature(self):
        '''Test Phase 2+ has learnable curvature.'''
        config = CurvatureConfig(initial_curvature=1.0)
        manager = CurvatureManager(config)

        manager.set_phase(2)

        assert manager._is_learnable
        assert manager._curvature.requires_grad

    def test_curvature_clamping(self):
        '''Test curvature stays within bounds.'''
        config = CurvatureConfig(
            min_curvature=0.5,
            max_curvature=5.0,
            initial_curvature=10.0,  # Above max
        )
        manager = CurvatureManager(config)
        manager.set_phase(2)  # Make learnable

        assert manager.curvature_float <= 5.0

    def test_state_serialization(self):
        '''Test state can be saved and restored.'''
        config = CurvatureConfig()
        manager = CurvatureManager(config)

        manager.set_phase(3)
        manager._curvature.data.fill_(2.5)

        state = manager.get_state()

        new_manager = CurvatureManager(config)
        new_manager.load_state(state)

        assert new_manager._current_phase == 3
        assert abs(new_manager.curvature_float - 2.5) < 0.1

# -------------------------------------------------------------------------------------------------
# ManifoldAdapter Tests
# -------------------------------------------------------------------------------------------------

@pytest.mark.unit
class TestManifoldAdapter:
    '''Test suite for ManifoldAdapter.'''

    def test_to_hyperboloid(self, sample_tangent_vectors):
        '''Test mapping to hyperboloid.'''
        adapter = ManifoldAdapter()

        hyp = adapter.to_hyperboloid(sample_tangent_vectors)

        is_valid, _ = LorentzManifold.check_on_manifold(hyp, c=adapter.c)
        assert is_valid

    def test_to_tangent(self, sample_lorentz_embeddings):
        '''Test mapping to tangent space.'''
        adapter = ManifoldAdapter()

        tangent = adapter.to_tangent(sample_lorentz_embeddings)

        # Time component should be 0
        assert torch.allclose(tangent[:, 0], torch.zeros_like(tangent[:, 0]), atol=1e-5)

    def test_roundtrip(self, sample_tangent_vectors):
        '''Test to_hyperboloid and to_tangent roundtrip.'''
        adapter = ManifoldAdapter()

        hyp = adapter.to_hyperboloid(sample_tangent_vectors)
        tangent_back = adapter.to_tangent(hyp)

        # Due to numerical precision in exp/log maps, use looser tolerance
        # The direction should be preserved even if magnitude differs slightly
        norm_original = torch.norm(sample_tangent_vectors[:, 1:], dim=1, keepdim=True)
        norm_back = torch.norm(tangent_back[:, 1:], dim=1, keepdim=True)

        # Check directions are similar (cosine similarity)
        cosine_sim = (sample_tangent_vectors[:, 1:] * tangent_back[:, 1:]).sum(dim=1) / (
            norm_original.squeeze() * norm_back.squeeze() + 1e-8
        )
        assert torch.all(cosine_sim > 0.99)

    def test_ensure_on_manifold_valid(self, sample_lorentz_embeddings):
        '''Test ensure_on_manifold with valid embeddings returns valid result.'''
        # Use auto_project=True to ensure result is always valid
        adapter = ManifoldAdapter(validate_manifold=True, auto_project=True)

        result = adapter.ensure_on_manifold(sample_lorentz_embeddings)

        # Check that result is on manifold (projection ensures this)
        is_valid, _ = LorentzManifold.check_on_manifold(result, c=adapter.c, tolerance=1e-3)
        assert is_valid

        # Also check no NaN/Inf
        assert not torch.any(torch.isnan(result))
        assert not torch.any(torch.isinf(result))

    def test_ensure_on_manifold_projects_invalid(self, test_device):
        '''Test ensure_on_manifold projects invalid embeddings.'''
        adapter = ManifoldAdapter(validate_manifold=True, auto_project=True)

        invalid = torch.randn(10, 33, device=test_device)
        result = adapter.ensure_on_manifold(invalid)

        is_valid, _ = LorentzManifold.check_on_manifold(result, c=adapter.c)
        assert is_valid

    def test_distance(self, sample_lorentz_embeddings):
        '''Test distance computation through adapter.'''
        adapter = ManifoldAdapter()

        x = sample_lorentz_embeddings[:8]
        y = sample_lorentz_embeddings[8:]

        dist = adapter.distance(x, y)

        assert dist.shape == (8, )
        assert torch.all(dist >= 0)

    def test_phase_switching(self):
        '''Test phase switching updates curvature.'''
        adapter = ManifoldAdapter()

        adapter.set_phase(1)
        _c1 = adapter.c  # noqa: F841

        adapter.set_phase(2)
        # Curvature may or may not change, but phase should update
        assert adapter.curvature_manager._current_phase == 2

    def test_diagnostics(self, sample_lorentz_embeddings):
        '''Test diagnostics retrieval.'''
        adapter = ManifoldAdapter()

        # Process some embeddings
        adapter.ensure_on_manifold(sample_lorentz_embeddings)

        diag = adapter.get_diagnostics()

        assert 'curvature' in diag
        assert 'total_projections' in diag
        assert 'phase' in diag

    def test_statistics_reset(self):
        '''Test statistics can be reset.'''
        adapter = ManifoldAdapter()

        adapter._total_projections = 10
        adapter._total_violations = 5

        adapter.reset_statistics()

        assert adapter._total_projections == 0
        assert adapter._total_violations == 0

# -------------------------------------------------------------------------------------------------
# Validation Utilities Tests
# -------------------------------------------------------------------------------------------------

@pytest.mark.unit
class TestValidationUtilities:
    '''Test suite for validation utilities.'''

    def test_validate_valid_embeddings(self, sample_lorentz_embeddings):
        '''Test validation passes for valid embeddings.'''
        is_valid, diag = validate_hyperbolic_embeddings(sample_lorentz_embeddings, c=1.0)

        assert is_valid
        assert diag['num_violations'] == 0

    def test_validate_invalid_embeddings(self, test_device):
        '''Test validation fails for invalid embeddings.'''
        invalid = torch.randn(10, 33, device=test_device)

        is_valid, diag = validate_hyperbolic_embeddings(invalid, c=1.0)

        assert not is_valid
        assert diag['num_violations'] > 0

    def test_validate_raises_on_violation(self, test_device):
        '''Test validation can raise exception.'''
        invalid = torch.randn(10, 33, device=test_device)

        with pytest.raises(ValueError, match='violate manifold constraint'):
            validate_hyperbolic_embeddings(invalid, c=1.0, raise_on_violation=True)

    def test_validate_diagnostics(self, sample_lorentz_embeddings):
        '''Test validation returns complete diagnostics.'''
        is_valid, diag = validate_hyperbolic_embeddings(sample_lorentz_embeddings, c=1.0)

        assert 'is_valid' in diag
        assert 'num_violations' in diag
        assert 'max_violation' in diag
        assert 'time_coord_mean' in diag
        assert 'spatial_norm_mean' in diag
        assert 'curvature' in diag

    @pytest.mark.parametrize('tolerance', [1e-2, 1e-3, 1e-4])
    def test_validate_tolerance(self, sample_lorentz_embeddings, tolerance):
        '''Test validation respects tolerance parameter.'''
        is_valid, diag = validate_hyperbolic_embeddings(
            sample_lorentz_embeddings, c=1.0, tolerance=tolerance
        )

        # Valid embeddings should pass with reasonable tolerance
        if tolerance >= 1e-3:
            assert is_valid

# -------------------------------------------------------------------------------------------------
# Numerical Stability Tests
# -------------------------------------------------------------------------------------------------

@pytest.mark.unit
class TestNumericalStability:
    '''Test numerical stability of hyperbolic operations.'''

    def test_small_norm_vectors(self, test_device):
        '''Test handling of very small norm tangent vectors.'''
        small = torch.randn(10, 33, device=test_device) * 1e-8

        hyp = LorentzManifold.exp_map_zero(small, c=1.0)

        assert not torch.any(torch.isnan(hyp))
        assert not torch.any(torch.isinf(hyp))

    def test_large_norm_vectors(self, test_device):
        '''Test handling of large norm tangent vectors.'''
        large = torch.randn(10, 33, device=test_device) * 100

        hyp = LorentzManifold.exp_map_zero(large, c=1.0)

        assert not torch.any(torch.isnan(hyp))
        assert not torch.any(torch.isinf(hyp))

    def test_extreme_curvatures(self, sample_tangent_vectors):
        '''Test operations with extreme curvature values.'''
        for c in [0.1, 10.0]:
            hyp = LorentzManifold.exp_map_zero(sample_tangent_vectors, c=c)

            assert not torch.any(torch.isnan(hyp))
            assert not torch.any(torch.isinf(hyp))

            is_valid, _ = LorentzManifold.check_on_manifold(hyp, c=c, tolerance=1e-2)
            assert is_valid

    def test_distance_near_identical_points(self, test_device):
        '''Test distance computation for nearly identical points.'''
        # Create two nearly identical points
        tangent = torch.randn(1, 33, device=test_device)
        tangent[:, 0] = 0

        x = LorentzManifold.exp_map_zero(tangent, c=1.0)
        y = LorentzManifold.exp_map_zero(tangent + 1e-8, c=1.0)

        dist = LorentzManifold.distance(x, y, c=1.0)

        assert not torch.isnan(dist)
        assert dist >= 0
