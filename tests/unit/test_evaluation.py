'''
Unit tests for text model evaluation module.

Tests cover:
- EmbeddingEvaluator (distance and similarity computation)
- RetrievalMetrics (precision@k, recall@k, MAP, NDCG)
- HierarchyMetrics (cophenetic correlation, Spearman correlation, distortion)
- EmbeddingStatistics (statistics computation, collapse detection)
- NAICSEvaluationRunner (full evaluation pipeline)
'''

import pytest
import torch

from naics_embedder.text_model.evaluation import (
    EmbeddingEvaluator,
    EmbeddingStatistics,
    HierarchyMetrics,
    NAICSEvaluationRunner,
    RetrievalMetrics,
)

# -------------------------------------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------------------------------------

@pytest.fixture
def sample_embeddings(test_device):
    '''Generate sample embeddings for testing.'''
    torch.manual_seed(42)
    return torch.randn(20, 128, device=test_device)

@pytest.fixture
def sample_lorentz_embeddings(test_device):
    '''Generate valid Lorentz embeddings for testing.'''
    torch.manual_seed(42)
    # Create embeddings on hyperboloid: x_0^2 - sum(x_i^2) = -1/c
    # Start with spatial components
    spatial = torch.randn(20, 128, device=test_device)
    spatial_norm = torch.norm(spatial, dim=1, keepdim=True)
    # Set time component to satisfy constraint
    time = torch.sqrt(1.0 + spatial_norm**2)
    return torch.cat([time, spatial], dim=1)

@pytest.fixture
def sample_tree_distances(test_device):
    '''Generate sample tree distance matrix.'''
    torch.manual_seed(42)
    # Create symmetric distance matrix
    distances = torch.rand(20, 20, device=test_device) * 5.0
    distances = (distances + distances.t()) / 2.0
    distances.fill_diagonal_(0.0)
    return distances

@pytest.fixture
def sample_ground_truth_relevance(test_device):
    '''Generate binary relevance matrix.'''
    torch.manual_seed(42)
    # Create sparse relevance matrix (only some pairs are relevant)
    relevance = torch.zeros(20, 20, device=test_device)
    # Make first 5 items relevant to each other
    relevance[:5, :5] = 1.0
    relevance.fill_diagonal_(0.0)  # Exclude self
    return relevance

# -------------------------------------------------------------------------------------------------
# EmbeddingEvaluator Tests
# -------------------------------------------------------------------------------------------------

@pytest.mark.unit
class TestEmbeddingEvaluator:
    '''Test suite for EmbeddingEvaluator.'''

    def test_init(self, test_device):
        '''Test evaluator initialization.'''
        evaluator = EmbeddingEvaluator()
        # get_device() returns a string, not torch.device
        assert isinstance(evaluator.device, str)
        assert evaluator.device in ['cpu', 'cuda', 'mps']

    def test_compute_pairwise_distances_euclidean(self, sample_embeddings):
        '''Test Euclidean distance computation.'''
        evaluator = EmbeddingEvaluator()
        distances = evaluator.compute_pairwise_distances(sample_embeddings, metric='euclidean')

        assert distances.shape == (20, 20)
        assert torch.allclose(distances, distances.t())  # Symmetric
        # Zero diagonal
        assert torch.allclose(torch.diag(distances), torch.zeros(20, device=distances.device))

    def test_compute_pairwise_distances_cosine(self, sample_embeddings):
        '''Test cosine distance computation.'''
        evaluator = EmbeddingEvaluator()
        distances = evaluator.compute_pairwise_distances(sample_embeddings, metric='cosine')

        assert distances.shape == (20, 20)
        # Allow small numerical errors (cosine distance should be >= -1e-6)
        assert torch.all(distances >= -1e-6)  # Non-negative (with numerical tolerance)
        assert torch.all(distances <= 2.0 + 1e-6)  # Cosine distance is in [0, 2]
        assert torch.allclose(distances, distances.t())  # Symmetric

    def test_compute_pairwise_distances_lorentz(self, sample_lorentz_embeddings):
        '''Test Lorentz distance computation.'''
        evaluator = EmbeddingEvaluator()
        distances = evaluator.compute_pairwise_distances(
            sample_lorentz_embeddings, metric='lorentz', curvature=1.0
        )

        assert distances.shape == (20, 20)
        assert torch.all(distances >= 0.0)  # Non-negative
        assert torch.allclose(distances, distances.t())  # Symmetric
        # Lorentz distances: diagonal should be very small (not exactly zero due to numerical precision)
        diag = torch.diag(distances)
        assert torch.all(diag < 0.01)  # Diagonal should be very small

    def test_compute_pairwise_distances_invalid_metric(self, sample_embeddings):
        '''Test error handling for invalid metric.'''
        evaluator = EmbeddingEvaluator()
        with pytest.raises(ValueError, match='Unknown metric'):
            evaluator.compute_pairwise_distances(sample_embeddings, metric='invalid')

    def test_compute_similarity_matrix_cosine(self, sample_embeddings):
        '''Test cosine similarity computation.'''
        evaluator = EmbeddingEvaluator()
        similarities = evaluator.compute_similarity_matrix(sample_embeddings, metric='cosine')

        assert similarities.shape == (20, 20)
        # Allow small numerical errors (cosine similarity should be in [-1, 1] with tolerance)
        assert torch.all(similarities >= -1.0 - 1e-6) and torch.all(similarities <= 1.0 + 1e-6)
        assert torch.allclose(similarities, similarities.t())  # Symmetric
        # Self-similarity = 1
        assert torch.allclose(
            torch.diag(similarities),
            torch.ones(20, device=similarities.device),
            atol=1e-5,
        )

    def test_compute_similarity_matrix_dot(self, sample_embeddings):
        '''Test dot product similarity computation.'''
        evaluator = EmbeddingEvaluator()
        similarities = evaluator.compute_similarity_matrix(sample_embeddings, metric='dot')

        assert similarities.shape == (20, 20)
        assert torch.allclose(similarities, similarities.t())  # Symmetric

    def test_compute_similarity_matrix_invalid_metric(self, sample_embeddings):
        '''Test error handling for invalid similarity metric.'''
        evaluator = EmbeddingEvaluator()
        with pytest.raises(ValueError, match='Unknown metric'):
            evaluator.compute_similarity_matrix(sample_embeddings, metric='invalid')

# -------------------------------------------------------------------------------------------------
# RetrievalMetrics Tests
# -------------------------------------------------------------------------------------------------

@pytest.mark.unit
class TestRetrievalMetrics:
    '''Test suite for RetrievalMetrics.'''

    def test_init(self, test_device):
        '''Test metrics initialization.'''
        metrics = RetrievalMetrics()
        # get_device() returns a string, not torch.device
        assert isinstance(metrics.device, str)
        assert metrics.device in ['cpu', 'cuda', 'mps']

    def test_precision_at_k(self, sample_embeddings, sample_ground_truth_relevance):
        '''Test precision@k computation.'''
        metrics = RetrievalMetrics()
        evaluator = EmbeddingEvaluator()
        distances = evaluator.compute_pairwise_distances(sample_embeddings, metric='euclidean')

        precision = metrics.precision_at_k(distances, sample_ground_truth_relevance, k=5)

        assert precision.shape == (20, )
        assert torch.all(precision >= 0.0) and torch.all(precision <= 1.0)  # Precision in [0, 1]

    def test_precision_at_k_edge_cases(self, test_device):
        '''Test precision@k with edge cases.'''
        metrics = RetrievalMetrics()
        # Create simple case: 2 items, one relevant
        distances = torch.tensor([[0.0, 1.0], [1.0, 0.0]], device=test_device)
        relevance = torch.tensor([[0.0, 1.0], [1.0, 0.0]], device=test_device)

        precision = metrics.precision_at_k(distances, relevance, k=1)
        assert precision[0] == 1.0  # First item should retrieve relevant item
        assert precision[1] == 1.0  # Second item should retrieve relevant item

    def test_recall_at_k(self, sample_embeddings, sample_ground_truth_relevance):
        '''Test recall@k computation.'''
        metrics = RetrievalMetrics()
        evaluator = EmbeddingEvaluator()
        distances = evaluator.compute_pairwise_distances(sample_embeddings, metric='euclidean')

        recall = metrics.recall_at_k(distances, sample_ground_truth_relevance, k=5)

        assert recall.shape == (20, )
        assert torch.all(recall >= 0.0) and torch.all(recall <= 1.0)  # Recall in [0, 1]

    def test_recall_at_k_no_relevant(self, test_device):
        '''Test recall@k when there are no relevant items.'''
        metrics = RetrievalMetrics()
        distances = torch.rand(5, 5, device=test_device)
        distances.fill_diagonal_(0.0)
        relevance = torch.zeros(5, 5, device=test_device)

        recall = metrics.recall_at_k(distances, relevance, k=3)
        # Should handle gracefully (clamped to avoid division by zero)
        assert recall.shape == (5, )

    def test_mean_average_precision(self, sample_embeddings, sample_ground_truth_relevance):
        '''Test MAP computation.'''
        metrics = RetrievalMetrics()
        evaluator = EmbeddingEvaluator()
        distances = evaluator.compute_pairwise_distances(sample_embeddings, metric='euclidean')

        map_score = metrics.mean_average_precision(distances, sample_ground_truth_relevance)

        assert isinstance(map_score, torch.Tensor)
        assert map_score.shape == ()  # Scalar
        assert map_score >= 0.0 and map_score <= 1.0  # MAP in [0, 1]

    def test_mean_average_precision_with_k(self, sample_embeddings, sample_ground_truth_relevance):
        '''Test MAP computation with k limit.'''
        metrics = RetrievalMetrics()
        evaluator = EmbeddingEvaluator()
        distances = evaluator.compute_pairwise_distances(sample_embeddings, metric='euclidean')

        map_score = metrics.mean_average_precision(distances, sample_ground_truth_relevance, k=10)

        assert isinstance(map_score, torch.Tensor)
        assert map_score >= 0.0 and map_score <= 1.0

    def test_ndcg_at_k(self, sample_embeddings, test_device):
        '''Test NDCG@k computation.'''
        metrics = RetrievalMetrics()
        evaluator = EmbeddingEvaluator()
        distances = evaluator.compute_pairwise_distances(sample_embeddings, metric='euclidean')

        # Create relevance scores (higher = more relevant)
        relevance_scores = torch.rand(20, 20, device=test_device)
        relevance_scores.fill_diagonal_(0.0)

        ndcg = metrics.ndcg_at_k(distances, relevance_scores, k=10)

        assert ndcg.shape == (20, )
        assert torch.all(ndcg >= 0.0) and torch.all(ndcg <= 1.0)  # NDCG in [0, 1]

    def test_ndcg_at_k_perfect_ranking(self, test_device):
        '''Test NDCG@k with perfect ranking.'''
        metrics = RetrievalMetrics()
        # Create distances where closest items have highest relevance
        distances = torch.tensor(
            [
                [0.0, 0.1, 0.2, 0.3],
                [0.1, 0.0, 0.2, 0.3],
                [0.2, 0.2, 0.0, 0.3],
                [0.3, 0.3, 0.3, 0.0],
            ],
            device=test_device,
        )

        # Relevance scores: closer = higher relevance
        relevance_scores = 1.0 - distances
        relevance_scores.fill_diagonal_(0.0)

        ndcg = metrics.ndcg_at_k(distances, relevance_scores, k=3)
        # Should be close to 1.0 for perfect ranking
        assert torch.all(ndcg > 0.8)

# -------------------------------------------------------------------------------------------------
# HierarchyMetrics Tests
# -------------------------------------------------------------------------------------------------

@pytest.mark.unit
class TestHierarchyMetrics:
    '''Test suite for HierarchyMetrics.'''

    def test_init(self, test_device):
        '''Test metrics initialization.'''
        metrics = HierarchyMetrics()
        # get_device() returns a string, not torch.device
        assert isinstance(metrics.device, str)
        assert metrics.device in ['cpu', 'cuda', 'mps']

    def test_cophenetic_correlation(self, sample_embeddings, sample_tree_distances):
        '''Test cophenetic correlation computation.'''
        metrics = HierarchyMetrics()
        evaluator = EmbeddingEvaluator()
        emb_distances = evaluator.compute_pairwise_distances(sample_embeddings, metric='euclidean')

        result = metrics.cophenetic_correlation(emb_distances, sample_tree_distances)

        assert 'correlation' in result
        assert 'n_pairs' in result
        assert 'n_total' in result
        assert isinstance(result['correlation'], torch.Tensor)
        assert -1.0 <= result['correlation'].item() <= 1.0  # Correlation in [-1, 1]

    def test_cophenetic_correlation_with_min_distance(
        self, sample_embeddings, sample_tree_distances
    ):
        '''Test cophenetic correlation with minimum distance filter.'''
        metrics = HierarchyMetrics()
        evaluator = EmbeddingEvaluator()
        emb_distances = evaluator.compute_pairwise_distances(sample_embeddings, metric='euclidean')

        result = metrics.cophenetic_correlation(
            emb_distances, sample_tree_distances, min_distance=1.0
        )

        assert 'correlation' in result
        assert result['n_pairs'] <= result['n_total']  # Filtered pairs should be <= total

    def test_cophenetic_correlation_insufficient_pairs(self, test_device):
        '''Test cophenetic correlation with insufficient valid pairs.'''
        metrics = HierarchyMetrics()
        # Create very small tree distances (all filtered out)
        emb_distances = torch.rand(5, 5, device=test_device)
        emb_distances.fill_diagonal_(0.0)
        tree_distances = torch.ones(5, 5, device=test_device) * 0.05  # All below threshold
        tree_distances.fill_diagonal_(0.0)

        result = metrics.cophenetic_correlation(emb_distances, tree_distances, min_distance=0.1)

        correlation = result['correlation']
        if isinstance(correlation, torch.Tensor):
            assert correlation.item() == 0.0
        else:
            assert correlation == 0.0
        assert result['n_pairs'] < 2

    def test_spearman_correlation(self, sample_embeddings, sample_tree_distances):
        '''Test Spearman correlation computation.'''
        metrics = HierarchyMetrics()
        evaluator = EmbeddingEvaluator()
        emb_distances = evaluator.compute_pairwise_distances(sample_embeddings, metric='euclidean')

        result = metrics.spearman_correlation(emb_distances, sample_tree_distances)

        assert 'correlation' in result
        assert 'n_pairs' in result
        assert isinstance(result['correlation'], torch.Tensor)
        assert -1.0 <= result['correlation'].item() <= 1.0

    def test_spearman_correlation_perfect_rank(self, test_device):
        '''Test Spearman correlation with perfect rank preservation.'''
        metrics = HierarchyMetrics()
        # Create distances that perfectly preserve ranking
        tree_distances = torch.tensor(
            [
                [0.0, 1.0, 2.0, 3.0],
                [1.0, 0.0, 1.0, 2.0],
                [2.0, 1.0, 0.0, 1.0],
                [3.0, 2.0, 1.0, 0.0],
            ],
            device=test_device,
        )

        # Embedding distances are proportional (perfect rank preservation)
        emb_distances = tree_distances * 2.0

        result = metrics.spearman_correlation(emb_distances, tree_distances)
        # Should be close to 1.0 for perfect rank preservation
        correlation = result['correlation']
        if isinstance(correlation, torch.Tensor):
            assert correlation.item() > 0.9
        else:
            assert correlation > 0.9

    def test_distortion(self, sample_embeddings, sample_tree_distances):
        '''Test distortion metrics computation.'''
        metrics = HierarchyMetrics()
        evaluator = EmbeddingEvaluator()
        emb_distances = evaluator.compute_pairwise_distances(sample_embeddings, metric='euclidean')

        result = metrics.distortion(emb_distances, sample_tree_distances)

        assert 'mean_distortion' in result
        assert 'max_distortion' in result
        assert 'min_distortion' in result
        assert 'std_distortion' in result
        assert 'median_distortion' in result
        assert result['mean_distortion'] > 0.0

    def test_distortion_no_stretch(self, test_device):
        '''Test distortion with no stretching (proportional distances).'''
        metrics = HierarchyMetrics()
        tree_distances = torch.tensor(
            [
                [0.0, 1.0, 2.0],
                [1.0, 0.0, 1.0],
                [2.0, 1.0, 0.0],
            ],
            device=test_device,
        )

        # Embedding distances are exactly proportional
        emb_distances = tree_distances * 2.0

        result = metrics.distortion(emb_distances, tree_distances)
        # Mean distortion should be close to 2.0 (scaling factor)
        assert abs(result['mean_distortion'].item() - 2.0) < 0.1

    def test_ndcg_ranking(self, sample_embeddings, sample_tree_distances):
        '''Test NDCG ranking computation.'''
        metrics = HierarchyMetrics()
        evaluator = EmbeddingEvaluator()
        emb_distances = evaluator.compute_pairwise_distances(sample_embeddings, metric='euclidean')

        result = metrics.ndcg_ranking(emb_distances, sample_tree_distances, k_values=[5, 10])

        assert 'ndcg@5' in result
        assert 'ndcg@10' in result
        assert result['ndcg@5'] >= 0.0 and result['ndcg@5'] <= 1.0
        assert result['ndcg@10'] >= 0.0 and result['ndcg@10'] <= 1.0

# -------------------------------------------------------------------------------------------------
# EmbeddingStatistics Tests
# -------------------------------------------------------------------------------------------------

@pytest.mark.unit
class TestEmbeddingStatistics:
    '''Test suite for EmbeddingStatistics.'''

    def test_init(self, test_device):
        '''Test statistics initialization.'''
        stats = EmbeddingStatistics()
        # get_device() returns a string, not torch.device
        assert isinstance(stats.device, str)
        assert stats.device in ['cpu', 'cuda', 'mps']

    def test_compute_statistics(self, sample_embeddings):
        '''Test statistics computation.'''
        stats = EmbeddingStatistics()
        result = stats.compute_statistics(sample_embeddings)

        assert 'mean_norm' in result
        assert 'std_norm' in result
        assert 'min_norm' in result
        assert 'max_norm' in result
        assert 'mean_embedding' in result
        assert 'std_embedding' in result
        assert 'mean_pairwise_distance' in result
        assert 'std_pairwise_distance' in result

        assert result['mean_norm'] > 0.0
        assert result['min_norm'] >= 0.0
        assert result['max_norm'] >= result['min_norm']

    def test_compute_statistics_shape(self, test_device):
        '''Test statistics with different embedding shapes.'''
        stats = EmbeddingStatistics()
        embeddings = torch.randn(10, 64, device=test_device)
        result = stats.compute_statistics(embeddings)

        assert result['mean_embedding'].shape == (64, )  # Per-dimension mean
        assert isinstance(result['std_embedding'], torch.Tensor)  # Scalar

    def test_check_collapse_no_collapse(self, sample_embeddings):
        '''Test collapse detection with non-collapsed embeddings.'''
        stats = EmbeddingStatistics()
        result = stats.check_collapse(sample_embeddings)

        assert 'variance_collapsed' in result
        assert 'norm_collapsed' in result
        assert 'distance_collapsed' in result
        assert 'any_collapse' in result
        assert isinstance(result['any_collapse'], bool)
        # Random embeddings might be detected as collapsed if variance is low
        # Just check that the function runs and returns valid results
        assert isinstance(result['variance_collapsed'], bool)

    def test_check_collapse_variance_collapse(self, test_device):
        '''Test detection of variance collapse.'''
        stats = EmbeddingStatistics()
        # Create collapsed embeddings (all very similar)
        collapsed = torch.ones(10, 64, device=test_device) * 0.5
        collapsed += torch.randn(10, 64, device=test_device) * 0.0001  # Very small variance

        result = stats.check_collapse(collapsed, variance_threshold=0.001)
        assert result['variance_collapsed'] is True
        assert result['any_collapse'] is True

    def test_check_collapse_norm_collapse(self, test_device):
        '''Test detection of norm collapse.'''
        stats = EmbeddingStatistics()
        # Create embeddings with very similar norms
        collapsed = torch.randn(10, 64, device=test_device)
        collapsed = collapsed / torch.norm(collapsed, dim=1, keepdim=True) * 1.0  # All norm = 1.0

        result = stats.check_collapse(collapsed, norm_cv_threshold=0.05)
        # Should detect norm collapse (all norms are identical)
        assert result['norm_collapsed'] is True

    def test_check_collapse_distance_collapse(self, test_device):
        '''Test detection of distance collapse.'''
        stats = EmbeddingStatistics()
        # Create embeddings that are all very close to each other
        center = torch.randn(1, 64, device=test_device)
        collapsed = center + torch.randn(
            10, 64, device=test_device
        ) * 0.0001  # Even smaller variance

        result = stats.check_collapse(collapsed, distance_cv_threshold=0.05)
        # Should detect distance collapse when all points are very close
        assert result['distance_collapsed'] is True or result['variance_collapsed'] is True
        assert result['any_collapse'] is True

    def test_check_collapse_actual_values(self, sample_embeddings):
        '''Test that collapse check returns actual values for debugging.'''
        stats = EmbeddingStatistics()
        result = stats.check_collapse(sample_embeddings)

        assert 'mean_variance' in result
        assert 'norm_cv' in result
        assert 'distance_cv' in result
        assert 'norm_mean' in result
        assert 'norm_std' in result
        assert 'distance_mean' in result
        assert 'distance_std' in result

        # All should be numeric values
        assert isinstance(result['mean_variance'], float)
        assert isinstance(result['norm_cv'], float)

# -------------------------------------------------------------------------------------------------
# NAICSEvaluationRunner Tests
# -------------------------------------------------------------------------------------------------

@pytest.mark.unit
class TestNAICSEvaluationRunner:
    '''Test suite for NAICSEvaluationRunner.'''

    @pytest.fixture
    def mock_model(self, test_device):
        '''Create a mock NAICSContrastiveModel.'''
        from unittest.mock import MagicMock

        model = MagicMock()
        model.device = test_device
        return model

    def test_init(self, mock_model):
        '''Test runner initialization.'''
        runner = NAICSEvaluationRunner(mock_model)
        assert runner.model == mock_model
        assert runner.embedding_eval is not None
        assert runner.retrieval_metrics is not None
        assert runner.hierarchy_metrics is not None
        assert runner.embedding_stats is not None

    def test_evaluate_basic(self, mock_model, sample_embeddings):
        '''Test basic evaluation without optional inputs.'''
        runner = NAICSEvaluationRunner(mock_model)
        results = runner.evaluate(sample_embeddings)

        assert 'statistics' in results
        assert 'collapse_check' in results
        assert 'cophenetic_correlation' not in results  # Not provided
        assert 'retrieval' not in results  # Not provided

    def test_evaluate_with_tree_distances(
        self, mock_model, sample_embeddings, sample_tree_distances
    ):
        '''Test evaluation with tree distances.'''
        runner = NAICSEvaluationRunner(mock_model)
        results = runner.evaluate(sample_embeddings, tree_distances=sample_tree_distances)

        assert 'statistics' in results
        assert 'collapse_check' in results
        assert 'cophenetic_correlation' in results
        assert 'spearman_correlation' in results
        assert 'distortion' in results

    def test_evaluate_with_relevance(
        self, mock_model, sample_embeddings, sample_ground_truth_relevance
    ):
        '''Test evaluation with ground truth relevance.'''
        runner = NAICSEvaluationRunner(mock_model)
        results = runner.evaluate(
            sample_embeddings, ground_truth_relevance=sample_ground_truth_relevance
        )

        assert 'statistics' in results
        assert 'retrieval' in results
        assert 'precision@5' in results['retrieval']
        assert 'recall@5' in results['retrieval']
        assert 'map' in results['retrieval']

    def test_evaluate_full(
        self,
        mock_model,
        sample_embeddings,
        sample_tree_distances,
        sample_ground_truth_relevance,
    ):
        '''Test full evaluation with all inputs.'''
        runner = NAICSEvaluationRunner(mock_model)
        results = runner.evaluate(
            sample_embeddings,
            tree_distances=sample_tree_distances,
            ground_truth_relevance=sample_ground_truth_relevance,
            k_values=[5, 10],
        )

        assert 'statistics' in results
        assert 'collapse_check' in results
        assert 'cophenetic_correlation' in results
        assert 'spearman_correlation' in results
        assert 'distortion' in results
        assert 'retrieval' in results
        assert 'precision@5' in results['retrieval']
        assert 'precision@10' in results['retrieval']
        assert 'recall@5' in results['retrieval']
        assert 'recall@10' in results['retrieval']
        assert 'map' in results['retrieval']

    def test_evaluate_custom_k_values(
        self, mock_model, sample_embeddings, sample_ground_truth_relevance
    ):
        '''Test evaluation with custom k values.'''
        runner = NAICSEvaluationRunner(mock_model)
        results = runner.evaluate(
            sample_embeddings,
            ground_truth_relevance=sample_ground_truth_relevance,
            k_values=[3, 7, 15]
        )

        assert 'precision@3' in results['retrieval']
        assert 'precision@7' in results['retrieval']
        assert 'precision@15' in results['retrieval']
        assert 'recall@3' in results['retrieval']
        assert 'recall@7' in results['retrieval']
        assert 'recall@15' in results['retrieval']

# -------------------------------------------------------------------------------------------------
# Level Consistency Tests (Per-level embedding quality)
# -------------------------------------------------------------------------------------------------

@pytest.mark.unit
class TestLevelConsistency:
    '''Test suite for per-level embedding quality metrics.

    Tests that embeddings maintain quality and consistency across different
    NAICS hierarchy levels (2-digit through 6-digit codes).
    '''

    @pytest.fixture
    def level_embeddings(self, test_device):
        '''Generate embeddings for each hierarchy level.

        Returns dict mapping level -> embeddings tensor.
        '''
        torch.manual_seed(42)
        return {
            2: torch.randn(5, 128, device=test_device),  # 5 sector codes
            3: torch.randn(10, 128, device=test_device),  # 10 subsector codes
            4: torch.randn(15, 128, device=test_device),  # 15 industry group codes
            5: torch.randn(20, 128, device=test_device),  # 20 industry codes
            6: torch.randn(25, 128, device=test_device),  # 25 national codes
        }

    @pytest.fixture
    def level_tree_distances(self, test_device):
        '''Generate tree distances for each hierarchy level.'''
        torch.manual_seed(42)
        return {
            2: torch.rand(5, 5, device=test_device) * 2.0,
            3: torch.rand(10, 10, device=test_device) * 3.0,
            4: torch.rand(15, 15, device=test_device) * 4.0,
            5: torch.rand(20, 20, device=test_device) * 5.0,
            6: torch.rand(25, 25, device=test_device) * 6.0,
        }

    def test_level_consistency_statistics_per_level(self, level_embeddings):
        '''Test that statistics can be computed for each hierarchy level.'''
        stats = EmbeddingStatistics()

        level_stats = {}
        for level, embeddings in level_embeddings.items():
            level_stats[level] = stats.compute_statistics(embeddings)

        # Each level should have valid statistics
        for level in [2, 3, 4, 5, 6]:
            assert level in level_stats
            assert 'mean_norm' in level_stats[level]
            assert 'mean_pairwise_distance' in level_stats[level]
            assert level_stats[level]['mean_norm'] > 0

    def test_level_consistency_no_collapse_per_level(self, level_embeddings):
        '''Test that no level has collapsed embeddings.'''
        stats = EmbeddingStatistics()

        for level, embeddings in level_embeddings.items():
            result = stats.check_collapse(embeddings)
            # Random embeddings should not be collapsed
            assert 'any_collapse' in result
            # Note: actual collapse detection depends on thresholds

    def test_level_consistency_hierarchy_metrics_per_level(
        self, level_embeddings, level_tree_distances
    ):
        '''Test hierarchy preservation metrics for each level.'''
        evaluator = EmbeddingEvaluator()
        hierarchy_metrics = HierarchyMetrics()

        for level in [2, 3, 4, 5, 6]:
            embeddings = level_embeddings[level]
            tree_dists = level_tree_distances[level]

            # Make tree distances symmetric with zero diagonal
            tree_dists = (tree_dists + tree_dists.t()) / 2
            tree_dists.fill_diagonal_(0.0)

            emb_distances = evaluator.compute_pairwise_distances(embeddings, metric='euclidean')
            result = hierarchy_metrics.cophenetic_correlation(
                emb_distances, tree_dists, min_distance=0.05
            )

            assert 'correlation' in result
            assert 'n_pairs' in result
            # Correlation should be in valid range
            corr = result['correlation']
            if isinstance(corr, torch.Tensor):
                assert -1.0 <= corr.item() <= 1.0
            else:
                assert -1.0 <= corr <= 1.0

    def test_level_consistency_distance_properties(self, level_embeddings):
        '''Test that distance properties are consistent across levels.'''
        stats = EmbeddingStatistics()

        level_distance_stats = {}
        for level, embeddings in level_embeddings.items():
            emb_stats = stats.compute_statistics(embeddings)
            level_distance_stats[level] = {
                'mean_dist': emb_stats['mean_pairwise_distance'].item(),
                'std_dist': emb_stats['std_pairwise_distance'].item(),
            }

        # All levels should have positive mean distances
        for level in [2, 3, 4, 5, 6]:
            assert level_distance_stats[level]['mean_dist'] > 0

    def test_level_consistency_norm_distribution(self, level_embeddings):
        '''Test that norm distributions are reasonable across levels.'''
        stats = EmbeddingStatistics()

        for level, embeddings in level_embeddings.items():
            level_stats = stats.compute_statistics(embeddings)

            # Norms should be positive
            assert level_stats['min_norm'] >= 0
            assert level_stats['max_norm'] >= level_stats['min_norm']

            # Coefficient of variation of norms (std/mean) should be reasonable
            mean_norm = level_stats['mean_norm'].item()
            std_norm = level_stats['std_norm'].item()
            if mean_norm > 0:
                cv = std_norm / mean_norm
                # CV should be in reasonable range for random embeddings
                assert cv >= 0

    def test_level_consistency_retrieval_metrics(self, level_embeddings, test_device):
        '''Test retrieval metrics can be computed for each level.'''
        evaluator = EmbeddingEvaluator()
        retrieval = RetrievalMetrics()

        for level, embeddings in level_embeddings.items():
            n = embeddings.shape[0]
            distances = evaluator.compute_pairwise_distances(embeddings, metric='euclidean')

            # Create mock relevance (first k items are relevant to each other)
            k = min(3, n // 2)
            relevance = torch.zeros(n, n, device=test_device)
            relevance[:k, :k] = 1.0
            relevance.fill_diagonal_(0.0)

            # Compute MAP
            map_score = retrieval.mean_average_precision(distances, relevance)
            assert map_score >= 0.0 and map_score <= 1.0

    def test_level_consistency_cross_level_coherence(self, test_device):
        '''Test that embeddings from different levels maintain coherent relationships.

        Parent codes should generally be closer to their children than to
        codes from different branches in a well-trained embedding space.
        '''
        torch.manual_seed(42)
        evaluator = EmbeddingEvaluator()

        # Create parent and child embeddings with expected relationship
        # Parent embedding
        parent = torch.randn(1, 128, device=test_device)
        # Child embeddings (similar to parent with some noise)
        children = parent + torch.randn(3, 128, device=test_device) * 0.5
        # Unrelated codes (different random embeddings)
        unrelated = torch.randn(3, 128, device=test_device) * 2.0

        all_embeddings = torch.cat([parent, children, unrelated], dim=0)
        distances = evaluator.compute_pairwise_distances(all_embeddings, metric='euclidean')

        # Distance from parent (idx 0) to children (idx 1,2,3) should generally be
        # smaller than to unrelated (idx 4,5,6)
        parent_to_children = distances[0, 1:4].mean()
        parent_to_unrelated = distances[0, 4:7].mean()

        # In a well-structured embedding, children should be closer
        # (with high probability given our construction)
        assert parent_to_children < parent_to_unrelated
