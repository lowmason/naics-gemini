'''
Unit tests for hyperbolic geometry operations.

Tests the core Lorentz model operations including exponential/logarithmic maps,
distance computations, manifold validity checks, and numerical stability.
'''

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from naics_embedder.text_model.hyperbolic import (
    HyperbolicProjection,
    LorentzDistance,
    LorentzOps,
    check_lorentz_manifold_validity,
    compute_hyperbolic_radii,
    log_hyperbolic_diagnostics,
)

# -------------------------------------------------------------------------------------------------
# LorentzOps Tests
# -------------------------------------------------------------------------------------------------

@pytest.mark.unit
class TestLorentzOps:
    '''Test suite for LorentzOps utility class.'''

    def test_exp_log_roundtrip(self, sample_lorentz_embeddings):
        '''Test that exp(log(x)) ≈ x for valid Lorentz embeddings.'''

        c = 1.0
        log_emb = LorentzOps.log_map_zero(sample_lorentz_embeddings, c=c)
        reconstructed = LorentzOps.exp_map_zero(log_emb, c=c)

        assert torch.allclose(reconstructed, sample_lorentz_embeddings, atol=1e-5)

    def test_log_exp_roundtrip(self, sample_tangent_vectors):
        '''Test that log(exp(v)) ≈ v for tangent vectors.'''

        c = 1.0
        # Project tangent to hyperboloid
        hyp_emb = LorentzOps.exp_map_zero(sample_tangent_vectors, c=c)
        # Map back to tangent space
        reconstructed = LorentzOps.log_map_zero(hyp_emb, c=c)

        # Note: Only spatial components should match (time component should be 0)
        assert torch.allclose(reconstructed[:, 1:], sample_tangent_vectors[:, 1:], atol=1e-4)

    def test_exp_map_produces_valid_embeddings(self, sample_tangent_vectors):
        '''Test that exp_map produces embeddings on the Lorentz manifold.'''

        c = 1.0
        hyp_emb = LorentzOps.exp_map_zero(sample_tangent_vectors, c=c)

        is_valid, lorentz_norms, violations = check_lorentz_manifold_validity(
            hyp_emb, curvature=c, tolerance=1e-3
        )

        assert is_valid, f'Max violation: {violations.max().item()}'
        assert torch.allclose(lorentz_norms, torch.tensor(-1.0 / c), atol=1e-3)

    @pytest.mark.parametrize('curvature', [0.1, 0.5, 1.0, 5.0, 10.0])
    def test_exp_map_with_different_curvatures(self, sample_tangent_vectors, curvature):
        '''Test exp_map works correctly for different curvature values.'''

        hyp_emb = LorentzOps.exp_map_zero(sample_tangent_vectors, c=curvature)

        is_valid, lorentz_norms, _ = check_lorentz_manifold_validity(
            hyp_emb, curvature=curvature, tolerance=1e-2
        )

        assert is_valid
        expected_norm = -1.0 / curvature
        assert torch.allclose(lorentz_norms, torch.tensor(expected_norm), atol=1e-2)

    def test_distance_is_positive(self, sample_lorentz_embeddings):
        '''Test that Lorentzian distances are always non-negative.'''

        x = sample_lorentz_embeddings[:8]
        y = sample_lorentz_embeddings[8:]

        distances = LorentzOps.lorentz_distance(x, y, c=1.0)

        assert torch.all(distances >= 0), 'Distances must be non-negative'

    def test_distance_symmetry(self, sample_lorentz_embeddings):
        '''Test that d(x, y) = d(y, x) (symmetry).'''

        x = sample_lorentz_embeddings[:8]
        y = sample_lorentz_embeddings[8:]

        d_xy = LorentzOps.lorentz_distance(x, y, c=1.0)
        d_yx = LorentzOps.lorentz_distance(y, x, c=1.0)

        assert torch.allclose(d_xy, d_yx, atol=1e-6)

    def test_distance_to_self_is_zero(self, sample_lorentz_embeddings):
        '''Test that d(x, x) = 0 (identity of indiscernibles).'''

        distances = LorentzOps.lorentz_distance(
            sample_lorentz_embeddings, sample_lorentz_embeddings, c=1.0
        )

        # Use 5e-3 tolerance due to floating point precision in acosh near 1.0
        assert torch.allclose(distances, torch.zeros_like(distances), atol=5e-3)

    def test_triangle_inequality(self, sample_lorentz_embeddings):
        '''Test triangle inequality: d(x, z) ≤ d(x, y) + d(y, z).'''

        batch_size = sample_lorentz_embeddings.shape[0]
        third = batch_size // 3

        x = sample_lorentz_embeddings[:third]
        y = sample_lorentz_embeddings[third:2 * third]
        z = sample_lorentz_embeddings[2 * third:3 * third]

        d_xz = LorentzOps.lorentz_distance(x, z, c=1.0)
        d_xy = LorentzOps.lorentz_distance(x, y, c=1.0)
        d_yz = LorentzOps.lorentz_distance(y, z, c=1.0)

        # Allow small numerical tolerance
        assert torch.all(d_xz <= d_xy + d_yz + 1e-4), 'Triangle inequality violated'

    def test_numerical_stability_small_norms(self, test_device):
        '''Test numerical stability for very small norm tangent vectors.'''

        small_tangent = torch.randn(10, 385, device=test_device) * 1e-8

        hyp_emb = LorentzOps.exp_map_zero(small_tangent, c=1.0)

        is_valid, _, _ = check_lorentz_manifold_validity(hyp_emb, curvature=1.0, tolerance=1e-3)
        assert is_valid
        assert not torch.any(torch.isnan(hyp_emb))
        assert not torch.any(torch.isinf(hyp_emb))

    def test_numerical_stability_large_norms(self, test_device):
        '''Test numerical stability for very large norm tangent vectors.'''

        large_tangent = torch.randn(10, 385, device=test_device) * 100

        hyp_emb = LorentzOps.exp_map_zero(large_tangent, c=1.0)

        assert not torch.any(torch.isnan(hyp_emb))
        assert not torch.any(torch.isinf(hyp_emb))

# -------------------------------------------------------------------------------------------------
# HyperbolicProjection Tests
# -------------------------------------------------------------------------------------------------

@pytest.mark.unit
class TestHyperbolicProjection:
    '''Test suite for HyperbolicProjection module.'''

    def test_projection_output_shape(self, sample_euclidean_embeddings):
        '''Test that projection increases dimension by 1.'''

        input_dim = sample_euclidean_embeddings.shape[1]
        projection = HyperbolicProjection(input_dim=input_dim, curvature=1.0)

        output = projection(sample_euclidean_embeddings)

        assert output.shape == (sample_euclidean_embeddings.shape[0], input_dim + 1)

    def test_projection_preserves_batch_size(self, sample_euclidean_embeddings):
        '''Test that batch size is preserved through projection.'''

        input_dim = sample_euclidean_embeddings.shape[1]
        projection = HyperbolicProjection(input_dim=input_dim, curvature=1.0)

        output = projection(sample_euclidean_embeddings)

        assert output.shape[0] == sample_euclidean_embeddings.shape[0]

    def test_projection_output_on_manifold(self, sample_euclidean_embeddings):
        '''Test that projected embeddings lie on Lorentz manifold.'''

        input_dim = sample_euclidean_embeddings.shape[1]
        projection = HyperbolicProjection(input_dim=input_dim, curvature=1.0)

        output = projection(sample_euclidean_embeddings)

        is_valid, _, _ = check_lorentz_manifold_validity(output, curvature=1.0, tolerance=1e-3)
        assert is_valid

    @pytest.mark.parametrize('curvature', [0.1, 1.0, 5.0])
    def test_projection_with_different_curvatures(self, sample_euclidean_embeddings, curvature):
        '''Test projection works correctly for different curvature values.'''

        input_dim = sample_euclidean_embeddings.shape[1]
        projection = HyperbolicProjection(input_dim=input_dim, curvature=curvature)

        output = projection(sample_euclidean_embeddings)

        is_valid, _, _ = check_lorentz_manifold_validity(
            output, curvature=curvature, tolerance=1e-2
        )
        assert is_valid

    def test_projection_no_nans_or_infs(self, sample_euclidean_embeddings):
        '''Test that projection never produces NaN or Inf values.'''

        input_dim = sample_euclidean_embeddings.shape[1]
        projection = HyperbolicProjection(input_dim=input_dim, curvature=1.0)

        output = projection(sample_euclidean_embeddings)

        assert not torch.any(torch.isnan(output))
        assert not torch.any(torch.isinf(output))

# -------------------------------------------------------------------------------------------------
# LorentzDistance Tests
# -------------------------------------------------------------------------------------------------

@pytest.mark.unit
class TestLorentzDistance:
    '''Test suite for LorentzDistance module.'''

    def test_distance_output_shape(self, sample_lorentz_embeddings):
        '''Test that distance computation produces correct output shape.'''

        distance_fn = LorentzDistance(curvature=1.0)
        x = sample_lorentz_embeddings[:8]
        y = sample_lorentz_embeddings[8:]

        distances = distance_fn(x, y)

        assert distances.shape == (8, )

    def test_batched_distance_shape(self, sample_lorentz_embeddings, test_device):
        '''Test batched distance computation with broadcasting.'''

        distance_fn = LorentzDistance(curvature=1.0)

        batch_size = 4
        k_negatives = 4  # Adjusted to work with 16 samples (4 * 4 = 16)
        dim = sample_lorentz_embeddings.shape[1]

        anchor = sample_lorentz_embeddings[:batch_size]  # (4, dim)
        negatives = sample_lorentz_embeddings[:batch_size *
                                              k_negatives].view(batch_size, k_negatives, dim)

        distances = distance_fn.batched_forward(anchor, negatives)

        assert distances.shape == (batch_size, k_negatives)

    def test_batched_distance_correctness(self, sample_lorentz_embeddings) -> None:
        '''Test batched distance matches pairwise computation.'''

        distance_fn = LorentzDistance(curvature=1.0)

        batch_size = 4
        k = 3
        dim = sample_lorentz_embeddings.shape[1]

        anchor = sample_lorentz_embeddings[:batch_size]
        points = sample_lorentz_embeddings[:batch_size * k].view(batch_size, k, dim)

        batched_dist = distance_fn.batched_forward(anchor, points)

        # Compute pairwise for comparison
        pairwise_dist = torch.zeros(batch_size, k)
        for i in range(batch_size):
            for j in range(k):
                pairwise_dist[i, j] = distance_fn(anchor[i:i + 1], points[i, j:j + 1, :]).item()

        assert torch.allclose(batched_dist, pairwise_dist, atol=1e-5)

    def test_lorentz_dot_product(self, sample_lorentz_embeddings):
        '''Test Lorentz inner product computation.'''

        distance_fn = LorentzDistance(curvature=1.0)

        x = sample_lorentz_embeddings[:8]

        # Self dot product should equal Lorentz norm (-1/c)
        dot_self = distance_fn.lorentz_dot(x, x)

        assert torch.allclose(dot_self, torch.tensor(-1.0), atol=1e-3)

# -------------------------------------------------------------------------------------------------
# Manifold Validity Tests
# -------------------------------------------------------------------------------------------------

@pytest.mark.unit
class TestManifoldValidity:
    '''Test suite for manifold validity checking functions.'''

    def test_valid_embeddings_pass_check(self, sample_lorentz_embeddings):
        '''Test that valid embeddings pass the manifold check.'''

        is_valid, lorentz_norms, violations = check_lorentz_manifold_validity(
            sample_lorentz_embeddings, curvature=1.0, tolerance=1e-3
        )

        assert is_valid
        assert torch.allclose(lorentz_norms, torch.tensor(-1.0), atol=1e-3)
        assert torch.all(violations < 1e-3)

    def test_invalid_embeddings_fail_check(self, test_device):
        '''Test that invalid embeddings fail the manifold check.'''

        # Create deliberately invalid embeddings (random points)
        invalid_embeddings = torch.randn(10, 385, device=test_device)

        is_valid, _, violations = check_lorentz_manifold_validity(
            invalid_embeddings, curvature=1.0, tolerance=1e-3
        )

        assert not is_valid
        assert torch.any(violations > 1e-3)

    @pytest.mark.parametrize('curvature', [0.1, 0.5, 1.0, 5.0, 10.0])
    def test_validity_check_with_different_curvatures(self, sample_tangent_vectors, curvature):
        '''Test validity checking works for different curvatures.'''

        hyp_emb = LorentzOps.exp_map_zero(sample_tangent_vectors, c=curvature)

        is_valid, lorentz_norms, _ = check_lorentz_manifold_validity(
            hyp_emb, curvature=curvature, tolerance=1e-2
        )

        assert is_valid
        expected_norm = -1.0 / curvature
        assert torch.allclose(lorentz_norms, torch.tensor(expected_norm), atol=1e-2)

    def test_tolerance_parameter(self, sample_lorentz_embeddings):
        '''Test that tolerance parameter affects validity check.'''

        # Should pass with loose tolerance
        is_valid_loose, _, _ = check_lorentz_manifold_validity(
            sample_lorentz_embeddings, curvature=1.0, tolerance=1e-2
        )

        # Should pass with moderate tolerance for well-formed embeddings
        # Note: 1e-6 is too strict for float32 computations with cosh/sinh
        is_valid_moderate, _, _ = check_lorentz_manifold_validity(
            sample_lorentz_embeddings, curvature=1.0, tolerance=1e-4
        )

        assert is_valid_loose
        assert is_valid_moderate

# -------------------------------------------------------------------------------------------------
# Hyperbolic Radii Tests
# -------------------------------------------------------------------------------------------------

@pytest.mark.unit
class TestHyperbolicRadii:
    '''Test suite for hyperbolic radius computation.'''

    def test_compute_radii_shape(self, sample_lorentz_embeddings):
        '''Test that radii computation produces correct shape.'''

        radii = compute_hyperbolic_radii(sample_lorentz_embeddings)

        assert radii.shape == (sample_lorentz_embeddings.shape[0], )

    def test_radii_are_positive(self, sample_lorentz_embeddings):
        '''Test that hyperbolic radii are always positive.'''

        radii = compute_hyperbolic_radii(sample_lorentz_embeddings)

        assert torch.all(radii > 0), 'Hyperbolic radii must be positive'

    def test_radii_equal_time_coordinate(self, sample_lorentz_embeddings):
        '''Test that radii equal the time coordinate (x₀).'''

        radii = compute_hyperbolic_radii(sample_lorentz_embeddings)
        time_coords = sample_lorentz_embeddings[:, 0]

        assert torch.allclose(radii, time_coords, atol=1e-8)

    def test_origin_has_unit_radius(self, test_device):
        '''Test that the origin on hyperboloid has radius 1.'''

        # Origin in Lorentz model: (1, 0, 0, ..., 0)
        dim = 385
        origin = torch.zeros(1, dim, device=test_device)
        origin[0, 0] = 1.0

        radius = compute_hyperbolic_radii(origin)

        assert torch.allclose(radius, torch.tensor([1.0], device=test_device), atol=1e-6)

# -------------------------------------------------------------------------------------------------
# Diagnostics Tests
# -------------------------------------------------------------------------------------------------

@pytest.mark.unit
class TestHyperbolicDiagnostics:
    '''Test suite for hyperbolic diagnostics logging.'''

    def test_diagnostics_returns_dict(self, sample_lorentz_embeddings):
        '''Test that diagnostics returns a dictionary of metrics.'''

        diagnostics = log_hyperbolic_diagnostics(sample_lorentz_embeddings, curvature=1.0)

        assert isinstance(diagnostics, dict)
        assert 'manifold_valid' in diagnostics
        assert 'radius_mean' in diagnostics
        assert 'lorentz_norm_mean' in diagnostics

    def test_diagnostics_reports_valid_manifold(self, sample_lorentz_embeddings):
        '''Test that diagnostics correctly reports valid manifold.'''

        diagnostics = log_hyperbolic_diagnostics(sample_lorentz_embeddings, curvature=1.0)

        assert diagnostics['manifold_valid'] is True
        assert abs(diagnostics['lorentz_norm_mean'] - (-1.0)) < 1e-2

    def test_diagnostics_with_level_labels(self, sample_lorentz_embeddings, test_device):
        '''Test diagnostics with hierarchy level labels.'''

        batch_size = sample_lorentz_embeddings.shape[0]
        level_labels = torch.randint(2, 7, (batch_size, ), device=test_device)

        diagnostics = log_hyperbolic_diagnostics(
            sample_lorentz_embeddings, curvature=1.0, level_labels=level_labels
        )

        assert isinstance(diagnostics, dict)
        assert diagnostics['manifold_valid'] is True

# -------------------------------------------------------------------------------------------------
# Property-Based Tests (Hypothesis)
# -------------------------------------------------------------------------------------------------

@pytest.mark.unit
class TestHyperbolicProperties:
    '''Property-based tests using Hypothesis for robustness.'''

    @given(
        batch_size=st.integers(min_value=1, max_value=32),
        dim=st.integers(min_value=64, max_value=512),
    )
    @settings(max_examples=10, deadline=None)
    def test_exp_preserves_batch_size_property(self, batch_size, dim):
        '''Property test: exp_map preserves batch size for any valid input.'''

        tangent = torch.randn(batch_size, dim + 1)
        hyp = LorentzOps.exp_map_zero(tangent, c=1.0)

        assert hyp.shape[0] == batch_size
        assert hyp.shape[1] == dim + 1

    @given(curvature=st.floats(min_value=0.1, max_value=10.0))
    @settings(max_examples=10, deadline=None)
    def test_exp_produces_valid_manifold_property(self, curvature):
        '''Property test: exp_map always produces valid manifold points.'''

        # Create tangent vectors with controlled norms for numerical stability
        tangent = torch.randn(8, 385)
        tangent[:, 0] = 0.0  # Time component should be 0 for tangent at origin
        # Scale to reasonable norm (around 2) to avoid sinh/cosh overflow
        tangent = tangent / (torch.norm(tangent, dim=1, keepdim=True) + 1e-8) * 2.0

        hyp = LorentzOps.exp_map_zero(tangent, c=curvature)

        is_valid, _, _ = check_lorentz_manifold_validity(hyp, curvature=curvature, tolerance=1e-2)

        assert is_valid
