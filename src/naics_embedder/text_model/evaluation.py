# -------------------------------------------------------------------------------------------------
# Imports and settings
# -------------------------------------------------------------------------------------------------

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import torch
import torch.nn.functional as F

from naics_embedder.utils.backend import get_device

if TYPE_CHECKING:
    from naics_embedder.text_model.naics_model import NAICSContrastiveModel

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------------------------------
# Embedding similarity and distance utilities
# -------------------------------------------------------------------------------------------------

class EmbeddingEvaluator:

    def __init__(self):
        '''
        Evaluator for embedding quality metrics.

        Device is automatically detected via get_device().
        '''

        self.device, _, _ = get_device()

    def compute_pairwise_distances(
        self, embeddings: torch.Tensor, metric: str = 'euclidean', curvature: float = 1.0
    ) -> torch.Tensor:
        '''
        Compute pairwise distances between embeddings.

        Args:
            embeddings: Tensor of shape (N, D) for Euclidean or (N, D+1) for Lorentz
            metric: Distance metric ('euclidean', 'cosine', or 'lorentz')
            curvature: Curvature parameter for Lorentz metric (default: 1.0)

        Returns:
            Distance matrix of shape (N, N)
        '''

        embeddings = embeddings.to(self.device)
        embeddings = embeddings.float()

        if metric == 'euclidean':
            # Euclidean distance: ||u - v||_2
            distances = torch.cdist(embeddings, embeddings, p=2)

        elif metric == 'cosine':
            # Cosine distance: 1 - cos(u, v)
            normalized = F.normalize(embeddings, p=2, dim=1)
            similarities = torch.mm(normalized, normalized.t())
            distances = 1.0 - similarities

        elif metric == 'lorentz':
            # Lorentzian distance on hyperboloid
            distances = self._lorentz_distance_matrix(embeddings, curvature=curvature)

        else:
            raise ValueError(f'Unknown metric: {metric}')

        return distances

    def _lorentz_distance_matrix(
        self, embeddings: torch.Tensor, curvature: float = 1.0
    ) -> torch.Tensor:
        '''
        Compute pairwise Lorentzian distances using fully vectorized operations.

        Args:
            embeddings: Hyperbolic embeddings of shape (N, D+1)
            curvature: Curvature parameter c

        Returns:
            Distance matrix of shape (N, N)
        '''
        embeddings.shape[0]
        embeddings = embeddings.to(self.device)

        # Vectorized Lorentz dot product computation
        # embeddings: (N, D+1)
        u = embeddings.unsqueeze(1)  # (N, 1, D+1)
        v = embeddings.unsqueeze(0)  # (1, N, D+1)

        # Compute Lorentz dot product: sum of spatial - time
        uv = u * v  # (N, N, D+1)
        dot_product = torch.sum(uv[:, :, 1:], dim=2) - uv[:, :, 0]  # (N, N)

        # Clamp and compute distance
        clamped_dot = torch.clamp(dot_product, max=-1.0 - 1e-5)
        sqrt_c = torch.sqrt(torch.tensor(curvature, device=embeddings.device))
        distances = sqrt_c * torch.acosh(-clamped_dot)

        return distances

    def compute_similarity_matrix(
        self, embeddings: torch.Tensor, metric: str = 'cosine'
    ) -> torch.Tensor:
        '''
        Compute pairwise similarities between embeddings.

        Args:
            embeddings: Tensor of shape (N, D)
            metric: Similarity metric ('cosine' or 'dot')

        Returns:
            Similarity matrix of shape (N, N)
        '''

        embeddings = embeddings.to(self.device)

        if metric == 'cosine':
            normalized = F.normalize(embeddings, p=2, dim=1)
            similarities = torch.mm(normalized, normalized.t())

        elif metric == 'dot':
            similarities = torch.mm(embeddings, embeddings.t())

        else:
            raise ValueError(f'Unknown metric: {metric}')

        return similarities

# -------------------------------------------------------------------------------------------------
# Retrieval metrics
# -------------------------------------------------------------------------------------------------

class RetrievalMetrics:

    def __init__(self):
        '''
        Metrics for evaluating retrieval quality.

        Device is automatically detected via get_device().
        '''

        self.device, _, _ = get_device()

    def precision_at_k(
        self, distances: torch.Tensor, ground_truth: torch.Tensor, k: int = 10
    ) -> torch.Tensor:
        '''
        Compute precision@k for retrieval.

        Args:
            distances: Distance matrix of shape (N, N)
            ground_truth: Binary relevance matrix of shape (N, N)
            k: Number of top results to consider

        Returns:
            Precision@k for each query (shape: N)
        '''

        distances = distances.to(self.device)
        ground_truth = ground_truth.to(self.device)

        N = distances.shape[0]

        # Get top-k nearest neighbors (excluding self)
        _, top_k_indices = torch.topk(distances, k + 1, dim=1, largest=False)
        top_k_indices = top_k_indices[:, 1:]  # Exclude self

        # Check which top-k are relevant
        batch_indices = torch.arange(N, device=self.device).unsqueeze(1).expand(-1, k)
        retrieved_relevance = ground_truth[batch_indices, top_k_indices]

        # Precision = relevant retrieved / k
        precision = retrieved_relevance.float().sum(dim=1) / k

        return precision

    def recall_at_k(
        self, distances: torch.Tensor, ground_truth: torch.Tensor, k: int = 10
    ) -> torch.Tensor:
        '''
        Compute recall@k for retrieval.

        Args:
            distances: Distance matrix of shape (N, N)
            ground_truth: Binary relevance matrix of shape (N, N)
            k: Number of top results to consider

        Returns:
            Recall@k for each query (shape: N)
        '''

        distances = distances.to(self.device)
        ground_truth = ground_truth.to(self.device)

        N = distances.shape[0]

        # Get top-k nearest neighbors (excluding self)
        _, top_k_indices = torch.topk(distances, k + 1, dim=1, largest=False)
        top_k_indices = top_k_indices[:, 1:]

        # Check which top-k are relevant
        batch_indices = torch.arange(N, device=self.device).unsqueeze(1).expand(-1, k)
        retrieved_relevance = ground_truth[batch_indices, top_k_indices]

        # Total relevant for each query
        total_relevant = ground_truth.sum(dim=1) - ground_truth.diagonal()  # Exclude self
        total_relevant = torch.clamp(total_relevant, min=1.0)  # Avoid division by zero

        # Recall = relevant retrieved / total relevant
        recall = retrieved_relevance.float().sum(dim=1) / total_relevant

        return recall

    def mean_average_precision(
        self, distances: torch.Tensor, ground_truth: torch.Tensor, k: Optional[int] = None
    ) -> torch.Tensor:
        '''
        Compute Mean Average Precision (MAP).

        Args:
            distances: Distance matrix of shape (N, N)
            ground_truth: Binary relevance matrix of shape (N, N)
            k: Maximum rank to consider (None = all)

        Returns:
            MAP score (scalar)
        '''

        distances = distances.to(self.device)
        ground_truth = ground_truth.to(self.device)

        N = distances.shape[0]
        k = k or N

        # Sort by distance (ascending)
        sorted_indices = torch.argsort(distances, dim=1)[:, 1:k + 1]  # Exclude self

        # Get relevance of sorted results
        batch_indices = torch.arange(N, device=self.device).unsqueeze(1)
        sorted_relevance = ground_truth[batch_indices, sorted_indices]

        # Compute average precision for each query
        cumsum = torch.cumsum(sorted_relevance.float(), dim=1)
        positions = torch.arange(1, k + 1, device=self.device).float()

        precision_at_positions = cumsum / positions
        average_precision = (precision_at_positions * sorted_relevance.float()).sum(dim=1)

        # Normalize by total relevant
        total_relevant = ground_truth.sum(dim=1) - ground_truth.diagonal()
        total_relevant = torch.clamp(total_relevant, min=1.0)
        average_precision = average_precision / total_relevant

        # Mean over all queries
        map_score = average_precision.mean()

        return map_score

    def ndcg_at_k(
        self, distances: torch.Tensor, relevance_scores: torch.Tensor, k: int = 10
    ) -> torch.Tensor:
        '''
        Compute Normalized Discounted Cumulative Gain (NDCG@k).

        Args:
            distances: Distance matrix of shape (N, N)
            relevance_scores: Relevance scores (higher = more relevant), shape (N, N)
            k: Number of top results to consider

        Returns:
            NDCG@k for each query (shape: N)
        '''

        distances = distances.to(self.device)
        relevance_scores = relevance_scores.to(self.device)

        N = distances.shape[0]

        # Get top-k by distance
        _, top_k_indices = torch.topk(distances, k + 1, dim=1, largest=False)
        top_k_indices = top_k_indices[:, 1:]  # Exclude self

        # Get relevance of retrieved items
        batch_indices = torch.arange(N, device=self.device).unsqueeze(1).expand(-1, k)
        retrieved_relevance = relevance_scores[batch_indices, top_k_indices]

        # Compute DCG
        positions = torch.arange(1, k + 1, device=self.device).float()
        discounts = 1.0 / torch.log2(positions + 1)
        dcg = (retrieved_relevance * discounts).sum(dim=1)

        # Compute ideal DCG (sort by relevance)
        ideal_relevance, _ = torch.topk(relevance_scores, k + 1, dim=1, largest=True)
        ideal_relevance = ideal_relevance[:, 1:]  # Exclude self
        idcg = (ideal_relevance * discounts).sum(dim=1)

        # NDCG
        idcg = torch.clamp(idcg, min=1e-10)  # Avoid division by zero
        ndcg = dcg / idcg

        return ndcg

# -------------------------------------------------------------------------------------------------
# Hierarchy preservation metrics
# -------------------------------------------------------------------------------------------------

class HierarchyMetrics:

    def __init__(self):
        '''
        Metrics for evaluating hierarchy preservation.

        Device is automatically detected via get_device().
        '''

        self.device, _, _ = get_device()

    def cophenetic_correlation(
        self,
        embedding_distances: torch.Tensor,
        tree_distances: torch.Tensor,
        min_distance: float = 0.1,
    ) -> Dict[str, Union[torch.Tensor, int]]:
        '''
        Compute cophenetic correlation coefficient with better handling.

        Measures how well embedding distances preserve hierarchical tree distances.
        Filters out pairs with very small tree distances to avoid noise.

        Args:
            embedding_distances: Distance matrix from embeddings (N, N)
            tree_distances: Ground truth tree distances (N, N)
            min_distance: Minimum tree distance to include (filters out same-level codes)

        Returns:
            Dictionary with correlation and metadata
        '''

        embedding_distances = embedding_distances.to(self.device)
        tree_distances = tree_distances.to(self.device)

        # Get upper triangular indices (excluding diagonal)
        N = embedding_distances.shape[0]
        triu_indices = torch.triu_indices(N, N, offset=1, device=self.device)

        # Extract upper triangular values
        emb_dists = embedding_distances[triu_indices[0], triu_indices[1]]
        tree_dists = tree_distances[triu_indices[0], triu_indices[1]]

        # Filter out pairs with very small tree distances
        valid_mask = tree_dists >= min_distance
        emb_dists_filtered = emb_dists[valid_mask]
        tree_dists_filtered = tree_dists[valid_mask]

        # Check if we have enough valid pairs
        if len(emb_dists_filtered) < 2:
            return {
                'correlation': torch.tensor(0.0, device=self.device),
                'n_pairs': len(emb_dists_filtered),
                'n_total': len(emb_dists),
                'mean_tree_dist': tree_dists.mean(),
                'mean_emb_dist': emb_dists.mean(),
            }

        # Compute Pearson correlation on filtered pairs
        correlation = self._pearson_correlation(emb_dists_filtered, tree_dists_filtered)

        return {
            'correlation': correlation,
            'n_pairs': len(emb_dists_filtered),
            'n_total': len(emb_dists),
            'mean_tree_dist': tree_dists_filtered.mean(),
            'mean_emb_dist': emb_dists_filtered.mean(),
        }

    def _pearson_correlation(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        '''
        Compute Pearson correlation coefficient.

        Args:
            x: First variable (shape: N)
            y: Second variable (shape: N)

        Returns:
            Correlation coefficient (scalar)
        '''

        x_mean = x.mean()
        y_mean = y.mean()

        x_centered = x - x_mean
        y_centered = y - y_mean

        numerator = (x_centered * y_centered).sum()
        denominator = torch.sqrt((x_centered**2).sum() * (y_centered**2).sum())

        denominator = torch.clamp(denominator, min=1e-10)
        correlation = numerator / denominator

        return correlation

    def spearman_correlation(
        self,
        embedding_distances: torch.Tensor,
        tree_distances: torch.Tensor,
        min_distance: float = 0.1,
    ) -> Dict[str, Union[torch.Tensor, int]]:
        '''
        Compute Spearman rank correlation coefficient with filtering.

        Args:
            embedding_distances: Distance matrix from embeddings (N, N)
            tree_distances: Ground truth tree distances (N, N)
            min_distance: Minimum tree distance to include

        Returns:
            Dictionary with correlation and metadata
        '''

        embedding_distances = embedding_distances.to(self.device)
        tree_distances = tree_distances.to(self.device)

        # Get upper triangular values
        N = embedding_distances.shape[0]
        triu_indices = torch.triu_indices(N, N, offset=1, device=self.device)

        emb_dists = embedding_distances[triu_indices[0], triu_indices[1]]
        tree_dists = tree_distances[triu_indices[0], triu_indices[1]]

        # Filter out pairs with very small tree distances
        valid_mask = tree_dists >= min_distance
        emb_dists_filtered = emb_dists[valid_mask]
        tree_dists_filtered = tree_dists[valid_mask]

        if len(emb_dists_filtered) < 2:
            return {
                'correlation': torch.tensor(0.0, device=self.device),
                'n_pairs': len(emb_dists_filtered),
                'n_total': len(emb_dists),
            }

        # Convert to ranks
        emb_ranks = self._rank_tensor(emb_dists_filtered)
        tree_ranks = self._rank_tensor(tree_dists_filtered)

        # Compute Pearson correlation of ranks
        correlation = self._pearson_correlation(emb_ranks, tree_ranks)

        return {
            'correlation': correlation,
            'n_pairs': len(emb_dists_filtered),
            'n_total': len(emb_dists),
        }

    def ndcg_ranking(
        self,
        embedding_distances: torch.Tensor,
        tree_distances: torch.Tensor,
        k_values: List[int] = [5, 10, 20],
        min_distance: float = 0.1,
    ) -> Dict[str, torch.Tensor]:
        '''
        Compute NDCG@k for ranking evaluation.

        For each anchor code, ranks all other codes by embedding distance and
        evaluates using NDCG based on tree distance relevance.

        Args:
            embedding_distances: Distance matrix from embeddings (N, N)
            tree_distances: Ground truth tree distances (N, N)
            k_values: List of k values for NDCG@k
            min_distance: Minimum tree distance to consider

        Returns:
            Dictionary with NDCG@k for each k value
        '''
        embedding_distances = embedding_distances.to(self.device)
        tree_distances = tree_distances.to(self.device)

        N = embedding_distances.shape[0]
        results = {}

        # For each anchor, compute NDCG
        all_ndcg_scores = {k: [] for k in k_values}

        for anchor_idx in range(N):
            # Get distances from anchor to all others
            anchor_emb_dists = embedding_distances[anchor_idx]  # (N,)
            anchor_tree_dists = tree_distances[anchor_idx]  # (N,)

            # Filter out self and very small tree distances
            not_self = torch.arange(N, device=self.device) != anchor_idx
            valid_mask = (anchor_tree_dists >= min_distance) & not_self

            if valid_mask.sum() < max(k_values):
                continue

            # Get valid indices
            valid_indices = torch.where(valid_mask)[0]

            # Get distances and tree distances for valid codes
            valid_emb_dists = anchor_emb_dists[valid_indices]
            valid_tree_dists = anchor_tree_dists[valid_indices]

            # Convert tree distances to relevance scores
            # Lower tree distance = higher relevance
            max_tree_dist = valid_tree_dists.max()
            relevance_scores = (max_tree_dist - valid_tree_dists + 1e-6) / (max_tree_dist + 1e-6)

            # Compute NDCG@k for each k
            for k in k_values:
                k_actual = min(k, len(valid_emb_dists))

                # Sort by embedding distance (ascending = most relevant first)
                _, sorted_indices = torch.sort(valid_emb_dists, descending=False)
                sorted_relevance = relevance_scores[sorted_indices[:k_actual]]

                # Compute DCG
                positions = torch.arange(1, k_actual + 1, dtype=torch.float32, device=self.device)
                dcg = torch.sum(sorted_relevance / torch.log2(positions + 1))

                # Compute ideal DCG (sort by relevance descending)
                ideal_relevance, _ = torch.sort(relevance_scores, descending=True)
                ideal_dcg = torch.sum(ideal_relevance[:k_actual] / torch.log2(positions + 1))

                # NDCG = DCG / IDCG
                if ideal_dcg > 0:
                    ndcg = dcg / ideal_dcg
                    all_ndcg_scores[k].append(ndcg)

        # Average NDCG across all anchors
        for k in k_values:
            if len(all_ndcg_scores[k]) > 0:
                results[f'ndcg@{k}'] = torch.stack(all_ndcg_scores[k]).mean()
                results[f'ndcg@{k}_n_queries'] = len(all_ndcg_scores[k])
            else:
                results[f'ndcg@{k}'] = torch.tensor(0.0, device=self.device)
                results[f'ndcg@{k}_n_queries'] = 0

        return results

    def _rank_tensor(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Convert values to ranks.

        Args:
            x: Values to rank (shape: N)

        Returns:
            Ranks (shape: N)
        '''

        _, indices = torch.sort(x)
        ranks = torch.zeros_like(indices, dtype=torch.float32)
        ranks[indices] = torch.arange(len(x), device=x.device, dtype=torch.float32)

        return ranks

    def distortion(self, embedding_distances: torch.Tensor,
                   tree_distances: torch.Tensor) -> Dict[str, torch.Tensor]:
        '''
        Compute distortion metrics (how much distances are stretched/compressed).

        Args:
            embedding_distances: Distance matrix from embeddings (N, N)
            tree_distances: Ground truth tree distances (N, N)

        Returns:
            Dictionary with distortion metrics
        '''

        embedding_distances = embedding_distances.to(self.device)
        tree_distances = tree_distances.to(self.device)

        # Get upper triangular values
        N = embedding_distances.shape[0]
        triu_indices = torch.triu_indices(N, N, offset=1, device=self.device)

        emb_dists = embedding_distances[triu_indices[0], triu_indices[1]]
        tree_dists = tree_distances[triu_indices[0], triu_indices[1]]

        # Avoid division by zero
        tree_dists = torch.clamp(tree_dists, min=1e-10)

        # Distortion ratio
        ratios = emb_dists / tree_dists

        return {
            'mean_distortion': ratios.mean(),
            'max_distortion': ratios.max(),
            'min_distortion': ratios.min(),
            'std_distortion': ratios.std(),
            'median_distortion': ratios.median(),
        }

# -------------------------------------------------------------------------------------------------
# Embedding space statistics
# -------------------------------------------------------------------------------------------------

class EmbeddingStatistics:

    def __init__(self):
        '''
        Statistics for analyzing embedding space.

        Device is automatically detected via get_device().
        '''

        self.device, _, _ = get_device()

    def compute_statistics(self, embeddings: torch.Tensor) -> Dict[str, torch.Tensor]:
        '''
        Compute comprehensive statistics about embeddings.

        Args:
            embeddings: Tensor of shape (N, D)

        Returns:
            Dictionary of statistics
        '''

        embeddings = embeddings.to(self.device)
        embeddings = embeddings.float()

        # Basic statistics
        mean = embeddings.mean(dim=0)
        std = embeddings.std(dim=0)

        # Norms
        norms = torch.norm(embeddings, p=2, dim=1)

        # Pairwise distances
        distances = torch.cdist(embeddings, embeddings, p=2)
        triu_indices = torch.triu_indices(
            embeddings.shape[0], embeddings.shape[0], offset=1, device=self.device
        )
        pairwise_dists = distances[triu_indices[0], triu_indices[1]]

        return {
            'mean_norm': norms.mean(),
            'std_norm': norms.std(),
            'min_norm': norms.min(),
            'max_norm': norms.max(),
            'mean_embedding': mean,
            'std_embedding': std.mean(),  # Average std across dimensions
            'mean_pairwise_distance': pairwise_dists.mean(),
            'std_pairwise_distance': pairwise_dists.std(),
            'min_pairwise_distance': pairwise_dists.min(),
            'max_pairwise_distance': pairwise_dists.max(),
        }

    def check_collapse(
        self,
        embeddings: torch.Tensor,
        variance_threshold: float = 0.001,
        norm_cv_threshold: float = 0.05,
        distance_cv_threshold: float = 0.05,
    ) -> Dict[str, Any]:
        '''
        Check if embeddings have collapsed (become too similar).
        Uses more informative metrics including actual values and ratios.

        Args:
            embeddings: Tensor of shape (N, D)
            variance_threshold: Threshold for mean variance per dimension
            norm_cv_threshold: Threshold for coefficient of variation in norms
            distance_cv_threshold: Threshold for coefficient of variation in distances

        Returns:
            Dictionary with collapse indicators and actual values
        '''

        embeddings = embeddings.to(self.device)
        embeddings = embeddings.float()

        # Variance collapse: low variance across dimensions
        var_per_dim = embeddings.var(dim=0)
        mean_var = var_per_dim.mean().item()
        variance_collapsed = mean_var < variance_threshold

        # Norm collapse: all norms very similar (use coefficient of variation)
        norms = torch.norm(embeddings, p=2, dim=1)
        norm_mean = norms.mean().item()
        norm_std = norms.std().item()
        norm_cv = norm_std / (norm_mean + 1e-10)  # Coefficient of variation
        norm_collapsed = norm_cv < norm_cv_threshold

        # Distance collapse: all pairwise distances very similar
        distances = torch.cdist(embeddings, embeddings, p=2)
        triu_indices = torch.triu_indices(
            embeddings.shape[0], embeddings.shape[0], offset=1, device=self.device
        )
        pairwise_dists = distances[triu_indices[0], triu_indices[1]]
        distance_mean = pairwise_dists.mean().item()
        distance_std = pairwise_dists.std().item()
        distance_cv = distance_std / (distance_mean + 1e-10)
        distance_collapsed = distance_cv < distance_cv_threshold

        any_collapse = variance_collapsed or norm_collapsed or distance_collapsed

        return {
            'variance_collapsed': variance_collapsed,
            'norm_collapsed': norm_collapsed,
            'distance_collapsed': distance_collapsed,
            'any_collapse': any_collapse,
            # Include actual values for debugging
            'mean_variance': mean_var,
            'norm_cv': norm_cv,
            'distance_cv': distance_cv,
            'norm_mean': norm_mean,
            'norm_std': norm_std,
            'distance_mean': distance_mean,
            'distance_std': distance_std,
        }

# -------------------------------------------------------------------------------------------------
# Evaluation runner
# -------------------------------------------------------------------------------------------------

class NAICSEvaluationRunner:

    def __init__(self, model: 'NAICSContrastiveModel'):
        '''
        Complete evaluation runner for NAICS embeddings.

        Args:
            model: Trained NAICSContrastiveModel instance

        Device is automatically detected via get_device().
        '''

        self.model = model
        self.device, _, _ = get_device()

        self.embedding_eval = EmbeddingEvaluator()
        self.retrieval_metrics = RetrievalMetrics()
        self.hierarchy_metrics = HierarchyMetrics()
        self.embedding_stats = EmbeddingStatistics()

    def evaluate(
        self,
        embeddings: torch.Tensor,
        tree_distances: Optional[torch.Tensor] = None,
        ground_truth_relevance: Optional[torch.Tensor] = None,
        k_values: List[int] = [5, 10, 20],
    ) -> Dict[str, Any]:
        '''
        Run comprehensive evaluation.

        Args:
            embeddings: Learned embeddings (N, D)
            tree_distances: Ground truth tree distances (N, N), optional
            ground_truth_relevance: Binary relevance matrix (N, N), optional
            k_values: k values for precision@k and recall@k

        Returns:
            Dictionary of all evaluation metrics
        '''

        results = {}

        # Embedding statistics
        logger.info('Computing embedding statistics...')
        results['statistics'] = self.embedding_stats.compute_statistics(embeddings)
        results['collapse_check'] = self.embedding_stats.check_collapse(embeddings)

        # Compute embedding distances
        logger.info('Computing pairwise distances...')
        emb_distances = self.embedding_eval.compute_pairwise_distances(
            embeddings, metric='euclidean'
        )

        # Hierarchy preservation
        if tree_distances is not None:
            logger.info('Evaluating hierarchy preservation...')
            results['cophenetic_correlation'] = self.hierarchy_metrics.cophenetic_correlation(
                emb_distances, tree_distances
            )
            results['spearman_correlation'] = self.hierarchy_metrics.spearman_correlation(
                emb_distances, tree_distances
            )
            results['distortion'] = self.hierarchy_metrics.distortion(emb_distances, tree_distances)

        # Retrieval metrics
        if ground_truth_relevance is not None:
            logger.info('Computing retrieval metrics...')
            results['retrieval'] = {}

            for k in k_values:
                precision = self.retrieval_metrics.precision_at_k(
                    emb_distances, ground_truth_relevance, k
                )
                recall = self.retrieval_metrics.recall_at_k(
                    emb_distances, ground_truth_relevance, k
                )

                results['retrieval'][f'precision@{k}'] = precision.mean()
                results['retrieval'][f'recall@{k}'] = recall.mean()

            results['retrieval']['map'] = self.retrieval_metrics.mean_average_precision(
                emb_distances, ground_truth_relevance
            )

        return results
