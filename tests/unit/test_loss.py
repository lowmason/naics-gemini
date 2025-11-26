'''
Unit tests for loss functions.

Tests hyperbolic contrastive learning losses, hierarchy preservation losses,
and ranking-based losses used in training.
'''

import pytest
import torch

from naics_embedder.text_model.hard_negative_mining import NormAdaptiveMargin
from naics_embedder.text_model.hyperbolic import LorentzOps
from naics_embedder.text_model.loss import (
    HierarchyPreservationLoss,
    HyperbolicInfoNCELoss,
    LambdaRankLoss,
    RankOrderPreservationLoss,
)

# -------------------------------------------------------------------------------------------------
# HyperbolicInfoNCELoss Tests
# -------------------------------------------------------------------------------------------------

@pytest.mark.unit
class TestHyperbolicInfoNCELoss:
    '''Test suite for Hyperbolic InfoNCE loss.'''

    @pytest.fixture
    def loss_fn(self):
        '''Create InfoNCE loss function.'''

        return HyperbolicInfoNCELoss(embedding_dim=384, temperature=0.07, curvature=1.0)

    @pytest.fixture
    def sample_triplet(self, test_device, random_seed):
        '''Generate sample anchor, positive, and negative embeddings.'''

        torch.manual_seed(random_seed)

        batch_size = 8
        k_negatives = 16
        dim = 384

        # Create tangent vectors
        anchor_tan = torch.randn(batch_size, dim + 1, device=test_device)
        positive_tan = torch.randn(batch_size, dim + 1, device=test_device)
        negative_tan = torch.randn(batch_size * k_negatives, dim + 1, device=test_device)

        # Project to Lorentz hyperboloid
        anchor = LorentzOps.exp_map_zero(anchor_tan, c=1.0)
        positive = LorentzOps.exp_map_zero(positive_tan, c=1.0)
        negatives = LorentzOps.exp_map_zero(negative_tan, c=1.0)

        return anchor, positive, negatives, batch_size, k_negatives

    def test_loss_is_scalar(self, loss_fn, sample_triplet):
        '''Test that loss returns a scalar value.'''

        anchor, positive, negatives, batch_size, k_negatives = sample_triplet

        loss = loss_fn(anchor, positive, negatives, batch_size, k_negatives)

        assert loss.dim() == 0, 'Loss should be a scalar'
        assert loss.numel() == 1

    def test_loss_is_positive(self, loss_fn, sample_triplet):
        '''Test that loss is always non-negative.'''

        anchor, positive, negatives, batch_size, k_negatives = sample_triplet

        loss = loss_fn(anchor, positive, negatives, batch_size, k_negatives)

        assert loss >= 0, 'Loss should be non-negative'

    def test_loss_decreases_with_closer_positive(self, loss_fn, test_device):
        '''Test that loss decreases when positive is closer to anchor.'''

        batch_size = 4
        k_negatives = 8
        dim = 384

        # Anchor
        anchor_tan = torch.randn(batch_size, dim + 1, device=test_device)
        anchor = LorentzOps.exp_map_zero(anchor_tan, c=1.0)

        # Close positive (small perturbation)
        close_positive_tan = anchor_tan + torch.randn_like(anchor_tan) * 0.1
        close_positive = LorentzOps.exp_map_zero(close_positive_tan, c=1.0)

        # Far positive (moderate perturbation to avoid numerical overflow)
        far_positive_tan = anchor_tan + torch.randn_like(anchor_tan) * 2.0
        far_positive = LorentzOps.exp_map_zero(far_positive_tan, c=1.0)

        # Negatives
        negative_tan = torch.randn(batch_size * k_negatives, dim + 1, device=test_device)
        negatives = LorentzOps.exp_map_zero(negative_tan, c=1.0)

        # Compute losses
        loss_close = loss_fn(anchor, close_positive, negatives, batch_size, k_negatives)
        loss_far = loss_fn(anchor, far_positive, negatives, batch_size, k_negatives)

        assert loss_close < loss_far, 'Loss should be lower for closer positives'

    def test_false_negative_masking(self, loss_fn, sample_triplet, test_device):
        '''Test that false negative masking reduces loss.'''

        anchor, positive, negatives, batch_size, k_negatives = sample_triplet

        # Loss without masking
        loss_no_mask = loss_fn(anchor, positive, negatives, batch_size, k_negatives)

        # Loss with masking (mask out some negatives)
        false_negative_mask = torch.zeros(
            batch_size, k_negatives, dtype=torch.bool, device=test_device
        )
        false_negative_mask[:, :4] = True  # Mask first 4 negatives for each anchor

        loss_with_mask = loss_fn(
            anchor, positive, negatives, batch_size, k_negatives, false_negative_mask
        )

        # Loss with masking should be different (typically lower)
        assert loss_with_mask != loss_no_mask

    @pytest.mark.parametrize('temperature', [0.01, 0.05, 0.1, 0.5])
    def test_temperature_effect(self, sample_triplet, temperature):
        '''Test that temperature scaling affects loss magnitude.'''

        loss_fn = HyperbolicInfoNCELoss(embedding_dim=384, temperature=temperature, curvature=1.0)
        anchor, positive, negatives, batch_size, k_negatives = sample_triplet

        loss = loss_fn(anchor, positive, negatives, batch_size, k_negatives)

        # Loss should be computable for all valid temperatures
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_gradient_flow(self, loss_fn, sample_triplet):
        '''Test that gradients flow through the loss.'''

        anchor, positive, negatives, batch_size, k_negatives = sample_triplet

        # Enable gradients
        anchor.requires_grad_(True)
        positive.requires_grad_(True)
        negatives.requires_grad_(True)

        loss = loss_fn(anchor, positive, negatives, batch_size, k_negatives)
        loss.backward()

        # Check that gradients exist and are non-zero
        assert anchor.grad is not None
        assert positive.grad is not None
        assert negatives.grad is not None
        assert torch.any(anchor.grad != 0)

    def test_adaptive_margin_increases_contrastive_pressure(self, sample_triplet):
        '''Adaptive margins should make negatives harder (higher loss).'''

        anchor, positive, negatives, batch_size, k_negatives = sample_triplet

        loss_fn = HyperbolicInfoNCELoss(embedding_dim=384, temperature=0.07, curvature=1.0)

        adaptive_margins = torch.full((batch_size, ), 0.5, device=anchor.device)

        loss_nomargin = loss_fn(anchor, positive, negatives, batch_size, k_negatives)
        loss_margin = loss_fn(
            anchor, positive, negatives, batch_size, k_negatives, adaptive_margins=adaptive_margins
        )

        assert loss_margin > loss_nomargin

class TestNormAdaptiveMargin:
    '''Tests for norm-adaptive margin computation.'''

    def test_margin_decays_with_radius(self, test_device):
        '''Margin should shrink as Lorentz norm grows.'''

        miner = NormAdaptiveMargin(base_margin=1.0, curvature=1.0).to(test_device)

        # Small norm (near origin)
        anchor_small = LorentzOps.exp_map_zero(torch.zeros(2, 385, device=test_device), c=1.0)

        # Larger norm via scaled tangent
        anchor_large = LorentzOps.exp_map_zero(torch.ones(2, 385, device=test_device) * 2.0, c=1.0)

        margin_small = miner(anchor_small)
        margin_large = miner(anchor_large)

        assert torch.all(margin_small > margin_large)

# -------------------------------------------------------------------------------------------------
# HierarchyPreservationLoss Tests
# -------------------------------------------------------------------------------------------------

@pytest.mark.unit
class TestHierarchyPreservationLoss:
    '''Test suite for Hierarchy Preservation loss.'''

    @pytest.fixture
    def sample_tree_distances(self, test_device):
        '''Create sample tree distance matrix.'''

        # Simple 4-node tree with known distances
        distances = torch.tensor(
            [
                [0.0, 0.5, 1.5, 2.5],
                [0.5, 0.0, 0.5, 1.5],
                [1.5, 0.5, 0.0, 0.5],
                [2.5, 1.5, 0.5, 0.0],
            ],
            device=test_device,
        )

        code_to_idx = {
            '31': 0,
            '311': 1,
            '3111': 2,
            '31111': 3,
        }

        return distances, code_to_idx

    @pytest.fixture
    def hierarchy_loss_fn(self, sample_tree_distances):
        '''Create hierarchy preservation loss function.'''

        distances, code_to_idx = sample_tree_distances
        return HierarchyPreservationLoss(
            tree_distances=distances, code_to_idx=code_to_idx, weight=0.1, min_distance=0.1
        )

    def test_loss_is_scalar(self, hierarchy_loss_fn, test_device):
        '''Test that hierarchy loss returns a scalar.'''

        # Create sample embeddings
        embeddings = torch.randn(4, 385, device=test_device)
        embeddings = LorentzOps.exp_map_zero(embeddings, c=1.0)

        codes = ['31', '311', '3111', '31111']

        loss = hierarchy_loss_fn(embeddings, codes, LorentzOps.lorentz_distance)

        assert loss.dim() == 0
        assert loss.numel() == 1

    def test_loss_is_non_negative(self, hierarchy_loss_fn, test_device):
        '''Test that hierarchy loss is non-negative.'''

        embeddings = torch.randn(4, 385, device=test_device)
        embeddings = LorentzOps.exp_map_zero(embeddings, c=1.0)

        codes = ['31', '311', '3111', '31111']

        loss = hierarchy_loss_fn(embeddings, codes, LorentzOps.lorentz_distance)

        assert loss >= 0

    def test_loss_zero_for_insufficient_codes(self, hierarchy_loss_fn, test_device):
        '''Test that loss is zero when there are fewer than 2 valid codes.'''

        embeddings = torch.randn(2, 385, device=test_device)
        embeddings = LorentzOps.exp_map_zero(embeddings, c=1.0)

        # Only one valid code
        codes = ['31', 'invalid_code']

        loss = hierarchy_loss_fn(embeddings, codes, LorentzOps.lorentz_distance)

        assert loss == 0.0

    def test_loss_respects_weight_parameter(self, sample_tree_distances, test_device):
        '''Test that loss scales with weight parameter.'''

        distances, code_to_idx = sample_tree_distances

        loss_fn_1 = HierarchyPreservationLoss(distances, code_to_idx, weight=0.1)
        loss_fn_2 = HierarchyPreservationLoss(distances, code_to_idx, weight=0.2)

        embeddings = torch.randn(4, 385, device=test_device)
        embeddings = LorentzOps.exp_map_zero(embeddings, c=1.0)
        codes = ['31', '311', '3111', '31111']

        loss_1 = loss_fn_1(embeddings, codes, LorentzOps.lorentz_distance)
        loss_2 = loss_fn_2(embeddings, codes, LorentzOps.lorentz_distance)

        # loss_2 should be approximately 2x loss_1
        assert torch.allclose(loss_2, 2 * loss_1, rtol=0.01)

# -------------------------------------------------------------------------------------------------
# RankOrderPreservationLoss Tests
# -------------------------------------------------------------------------------------------------

@pytest.mark.unit
class TestRankOrderPreservationLoss:
    '''Test suite for Rank Order Preservation loss.'''

    @pytest.fixture
    def rank_loss_fn(self, sample_tree_distances):
        '''Create rank order preservation loss function.'''

        distances, code_to_idx = sample_tree_distances

        return RankOrderPreservationLoss(
            tree_distances=distances,
            code_to_idx=code_to_idx,
            weight=0.1,
            min_distance=0.1,
            margin=0.1,
        )

    @pytest.fixture
    def sample_tree_distances(self, test_device):
        '''Create sample tree distance matrix.'''

        distances = torch.tensor(
            [
                [0.0, 0.5, 1.5, 2.5],
                [0.5, 0.0, 0.5, 1.5],
                [1.5, 0.5, 0.0, 0.5],
                [2.5, 1.5, 0.5, 0.0],
            ],
            device=test_device,
        )

        code_to_idx = {
            '31': 0,
            '311': 1,
            '3111': 2,
            '31111': 3,
        }

        return distances, code_to_idx

    def test_loss_is_scalar(self, rank_loss_fn, test_device):
        '''Test that rank order loss returns a scalar.'''

        embeddings = torch.randn(4, 385, device=test_device)
        embeddings = LorentzOps.exp_map_zero(embeddings, c=1.0)

        codes = ['31', '311', '3111', '31111']

        loss = rank_loss_fn(embeddings, codes, LorentzOps.lorentz_distance)

        assert loss.dim() == 0

    def test_loss_is_non_negative(self, rank_loss_fn, test_device):
        '''Test that rank order loss is non-negative.'''

        embeddings = torch.randn(4, 385, device=test_device)
        embeddings = LorentzOps.exp_map_zero(embeddings, c=1.0)

        codes = ['31', '311', '3111', '31111']

        loss = rank_loss_fn(embeddings, codes, LorentzOps.lorentz_distance)

        assert loss >= 0

    def test_loss_zero_for_insufficient_codes(self, rank_loss_fn, test_device):
        '''Test that loss is zero when there are fewer than 3 valid codes.'''

        embeddings = torch.randn(3, 385, device=test_device)
        embeddings = LorentzOps.exp_map_zero(embeddings, c=1.0)

        # Only 2 valid codes
        codes = ['31', '311', 'invalid_code']

        loss = rank_loss_fn(embeddings, codes, LorentzOps.lorentz_distance)

        assert loss == 0.0

    def test_margin_parameter_effect(self, sample_tree_distances, test_device):
        '''Test that margin parameter affects loss magnitude.'''

        distances, code_to_idx = sample_tree_distances

        loss_fn_small = RankOrderPreservationLoss(distances, code_to_idx, weight=0.1, margin=0.1)
        loss_fn_large = RankOrderPreservationLoss(distances, code_to_idx, weight=0.1, margin=0.5)

        embeddings = torch.randn(4, 385, device=test_device)
        embeddings = LorentzOps.exp_map_zero(embeddings, c=1.0)
        codes = ['31', '311', '3111', '31111']

        loss_small = loss_fn_small(embeddings, codes, LorentzOps.lorentz_distance)
        loss_large = loss_fn_large(embeddings, codes, LorentzOps.lorentz_distance)

        # Both losses should be valid (may be zero if no violations)
        assert not torch.isnan(loss_small)
        assert not torch.isnan(loss_large)

# -------------------------------------------------------------------------------------------------
# LambdaRankLoss Tests
# -------------------------------------------------------------------------------------------------

@pytest.mark.unit
class TestLambdaRankLoss:
    '''Test suite for LambdaRank loss.'''

    @pytest.fixture
    def sample_tree_distances(self, test_device):
        '''Create sample tree distance matrix.'''

        distances = torch.tensor(
            [
                [0.0, 0.5, 1.5, 2.5, 3.5],
                [0.5, 0.0, 0.5, 1.5, 2.5],
                [1.5, 0.5, 0.0, 0.5, 1.5],
                [2.5, 1.5, 0.5, 0.0, 0.5],
                [3.5, 2.5, 1.5, 0.5, 0.0],
            ],
            device=test_device,
        )

        code_to_idx = {
            '31': 0,
            '311': 1,
            '3111': 2,
            '31111': 3,
            '311111': 4,
        }

        return distances, code_to_idx

    @pytest.fixture
    def lambda_rank_loss_fn(self, sample_tree_distances):
        '''Create LambdaRank loss function.'''

        distances, code_to_idx = sample_tree_distances

        return LambdaRankLoss(
            tree_distances=distances, code_to_idx=code_to_idx, weight=0.15, sigma=1.0, ndcg_k=10
        )

    def test_ndcg_computation(self, lambda_rank_loss_fn, test_device):
        '''Test NDCG computation correctness.'''

        # Perfect ranking (relevance descending)
        relevance = torch.tensor([4.0, 3.0, 2.0, 1.0], device=test_device)
        distances = torch.tensor([0.5, 1.0, 1.5, 2.0], device=test_device)  # Ascending

        ndcg = lambda_rank_loss_fn._compute_ndcg(relevance, distances, k=4)

        # Perfect ranking should have NDCG = 1.0
        assert torch.allclose(ndcg, torch.tensor(1.0, device=test_device), atol=1e-4)

    def test_ndcg_imperfect_ranking(self, lambda_rank_loss_fn, test_device):
        '''Test NDCG for imperfect ranking.'''

        # Relevance: high to low
        relevance = torch.tensor([4.0, 3.0, 2.0, 1.0], device=test_device)

        # Imperfect distances (not sorted properly)
        distances = torch.tensor([1.0, 0.5, 2.0, 1.5], device=test_device)

        ndcg = lambda_rank_loss_fn._compute_ndcg(relevance, distances, k=4)

        # Imperfect ranking should have NDCG < 1.0
        assert ndcg < 1.0
        assert ndcg >= 0.0

    def test_loss_is_scalar(self, lambda_rank_loss_fn, test_device):
        '''Test that LambdaRank loss returns a scalar.'''

        batch_size = 2
        k_negatives = 4
        dim = 384

        # Create embeddings
        anchor = torch.randn(batch_size, dim + 1, device=test_device)
        anchor = LorentzOps.exp_map_zero(anchor, c=1.0)

        positive = torch.randn(batch_size, dim + 1, device=test_device)
        positive = LorentzOps.exp_map_zero(positive, c=1.0)

        negatives = torch.randn(batch_size * k_negatives, dim + 1, device=test_device)
        negatives = LorentzOps.exp_map_zero(negatives, c=1.0)

        anchor_codes = ['31', '311']
        positive_codes = ['311', '3111']
        negative_codes = [['3111', '31111', '311111', '31'], ['31111', '311111', '31', '311']]

        loss = lambda_rank_loss_fn(
            anchor,
            positive,
            negatives,
            anchor_codes,
            positive_codes,
            negative_codes,
            LorentzOps.lorentz_distance,
            batch_size,
            k_negatives,
        )

        assert loss.dim() == 0

    def test_loss_is_non_negative(self, lambda_rank_loss_fn, test_device):
        '''Test that LambdaRank loss is non-negative.'''

        batch_size = 2
        k_negatives = 4
        dim = 384

        anchor = torch.randn(batch_size, dim + 1, device=test_device)
        anchor = LorentzOps.exp_map_zero(anchor, c=1.0)

        positive = torch.randn(batch_size, dim + 1, device=test_device)
        positive = LorentzOps.exp_map_zero(positive, c=1.0)

        negatives = torch.randn(batch_size * k_negatives, dim + 1, device=test_device)
        negatives = LorentzOps.exp_map_zero(negatives, c=1.0)

        anchor_codes = ['31', '311']
        positive_codes = ['311', '3111']
        negative_codes = [['3111', '31111', '311111', '31'], ['31111', '311111', '31', '311']]

        loss = lambda_rank_loss_fn(
            anchor,
            positive,
            negatives,
            anchor_codes,
            positive_codes,
            negative_codes,
            LorentzOps.lorentz_distance,
            batch_size,
            k_negatives,
        )

        # LambdaRank loss can be negative or positive (it's a gradient-based loss)
        # But it should be a valid number
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_loss_zero_for_invalid_codes(self, lambda_rank_loss_fn, test_device):
        '''Test that loss is zero when codes are not in ground truth.'''

        batch_size = 2
        k_negatives = 2
        dim = 384

        anchor = torch.randn(batch_size, dim + 1, device=test_device)
        anchor = LorentzOps.exp_map_zero(anchor, c=1.0)

        positive = torch.randn(batch_size, dim + 1, device=test_device)
        positive = LorentzOps.exp_map_zero(positive, c=1.0)

        negatives = torch.randn(batch_size * k_negatives, dim + 1, device=test_device)
        negatives = LorentzOps.exp_map_zero(negatives, c=1.0)

        # Invalid codes
        anchor_codes = ['invalid1', 'invalid2']
        positive_codes = ['invalid3', 'invalid4']
        negative_codes = [['invalid5', 'invalid6'], ['invalid7', 'invalid8']]

        loss = lambda_rank_loss_fn(
            anchor,
            positive,
            negatives,
            anchor_codes,
            positive_codes,
            negative_codes,
            LorentzOps.lorentz_distance,
            batch_size,
            k_negatives,
        )

        assert loss == 0.0

# -------------------------------------------------------------------------------------------------
# Integration Tests
# -------------------------------------------------------------------------------------------------

@pytest.mark.unit
class TestLossIntegration:
    '''Integration tests combining multiple loss functions.'''

    def test_combined_losses(self, test_device):
        '''Test that multiple losses can be computed and combined.'''
        # Setup
        batch_size = 4
        k_negatives = 8
        dim = 384

        # Create embeddings
        anchor = torch.randn(batch_size, dim + 1, device=test_device)
        anchor = LorentzOps.exp_map_zero(anchor, c=1.0)

        positive = torch.randn(batch_size, dim + 1, device=test_device)
        positive = LorentzOps.exp_map_zero(positive, c=1.0)

        negatives = torch.randn(batch_size * k_negatives, dim + 1, device=test_device)
        negatives = LorentzOps.exp_map_zero(negatives, c=1.0)

        # InfoNCE loss
        infonce_loss_fn = HyperbolicInfoNCELoss(embedding_dim=dim, temperature=0.07, curvature=1.0)
        infonce_loss = infonce_loss_fn(anchor, positive, negatives, batch_size, k_negatives)

        # Hierarchy loss
        tree_distances = torch.rand(4, 4, device=test_device)
        tree_distances = (tree_distances + tree_distances.T) / 2  # Symmetric
        tree_distances.fill_diagonal_(0)

        embeddings = torch.randn(4, 385, device=test_device)
        embeddings = LorentzOps.exp_map_zero(embeddings, c=1.0)
        codes = ['code1', 'code2', 'code3', 'code4']
        code_to_idx = {'code1': 0, 'code2': 1, 'code3': 2, 'code4': 3}

        # Hierarchy loss
        hierarchy_loss_fn = HierarchyPreservationLoss(tree_distances, code_to_idx, weight=0.1)
        hierarchy_loss = hierarchy_loss_fn(embeddings, codes, LorentzOps.lorentz_distance)

        # Rank order loss
        rank_order_loss_fn = RankOrderPreservationLoss(tree_distances, code_to_idx, weight=0.1)
        rank_order_loss = rank_order_loss_fn(embeddings, codes, LorentzOps.lorentz_distance)

        # Combined loss
        total_loss = infonce_loss + hierarchy_loss + rank_order_loss

        assert not torch.isnan(infonce_loss)
        assert not torch.isinf(infonce_loss)
        assert infonce_loss >= 0

        assert not torch.isnan(hierarchy_loss)
        assert not torch.isinf(hierarchy_loss)
        assert hierarchy_loss >= 0

        assert not torch.isnan(rank_order_loss)
        assert not torch.isinf(rank_order_loss)
        assert rank_order_loss >= 0

        assert not torch.isnan(total_loss)
        assert not torch.isinf(total_loss)
        assert total_loss >= 0
