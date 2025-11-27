# -------------------------------------------------------------------------------------------------
# Difficulty Annotation Preprocessing for Curriculum Learning (#53)
# -------------------------------------------------------------------------------------------------
'''
Pre-processing pipeline for node and relation difficulty scoring.

Computes:
  - Node scores: degree centrality, PageRank, k-core decomposition
  - Relation cardinality: 1-1, 1-N, N-1, N-N classification
  - Distance-based difficulty thresholds for curriculum phases

Distance/Difficulty Metrics (from training_triplets):
  - `distance`: Tree distance between nodes (lower = closer = easier)
  - `relation_id`: Relation type ID (lower = closer = easier)
      - 1 = child, 2 = sibling, 3 = grandchild, etc.
  - `margin`: Composite metric for weighted sampling (HIGHER = closer)
      - Formula: margin = 1 / (1/3 * relation_id + 2/3 * distance)
      - Incorporates exclusions (excluded pairs get adjusted margin)
      - Used for weighted negative sampling

Training triplet columns:
  - `relation_margin`: Relation-based component
  - `distance_margin`: Distance-based component
  - `margin`: Combined margin for sampling weights

Outputs:
  - node_scores.pt: PyTorch tensor dict keyed by node index for O(1) lookup
  - relation_types.json: Per-relation cardinality classification
  - distance_thresholds.json: Phase-based difficulty thresholds
'''

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import polars as pl
import torch

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------------------------------
# Node Centrality Computation
# -------------------------------------------------------------------------------------------------

def compute_degree_centrality(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    '''
    Compute degree centrality for each node.

    Degree centrality = degree(node) / (num_nodes - 1)

    Args:
        edge_index: Edge index tensor of shape (2, num_edges)
        num_nodes: Total number of nodes in the graph

    Returns:
        Tensor of shape (num_nodes,) with degree centrality scores
    '''
    # Count edges per node (undirected: count both directions)
    degrees = torch.zeros(num_nodes, dtype=torch.float32)

    # Source nodes
    src_nodes = edge_index[0]
    for node in src_nodes:
        degrees[node] += 1

    # Normalize by max possible degree
    max_degree = num_nodes - 1 if num_nodes > 1 else 1
    centrality = degrees / max_degree

    return centrality

def compute_pagerank(
    edge_index: torch.Tensor,
    num_nodes: int,
    damping: float = 0.85,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> torch.Tensor:
    '''
    Compute PageRank scores using power iteration.

    Args:
        edge_index: Edge index tensor of shape (2, num_edges)
        num_nodes: Total number of nodes
        damping: Damping factor (default: 0.85)
        max_iter: Maximum iterations (default: 100)
        tol: Convergence tolerance (default: 1e-6)

    Returns:
        Tensor of shape (num_nodes,) with PageRank scores
    '''
    # Build adjacency list for efficient iteration
    adj_list: Dict[int, list] = defaultdict(list)
    out_degree = torch.zeros(num_nodes, dtype=torch.float32)

    src = edge_index[0].tolist()
    dst = edge_index[1].tolist()

    for s, d in zip(src, dst):
        adj_list[s].append(d)
        out_degree[s] += 1

    # Handle dangling nodes (no outgoing edges)
    out_degree = torch.where(out_degree == 0, torch.ones_like(out_degree), out_degree)

    # Initialize PageRank uniformly
    pr = torch.ones(num_nodes, dtype=torch.float32) / num_nodes
    teleport = (1 - damping) / num_nodes

    for _ in range(max_iter):
        pr_new = torch.full((num_nodes, ), teleport, dtype=torch.float32)

        # Distribute PageRank along edges
        for src_node in range(num_nodes):
            if src_node in adj_list:
                contribution = damping * pr[src_node] / out_degree[src_node]
                for dst_node in adj_list[src_node]:
                    pr_new[dst_node] += contribution

        # Check convergence
        diff = torch.abs(pr_new - pr).sum()
        pr = pr_new

        if diff < tol:
            break

    # Normalize to sum to 1
    pr = pr / pr.sum()

    return pr

def compute_kcore(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    '''
    Compute k-core decomposition numbers for each node.

    The k-core of a node is the largest k such that the node belongs to
    a subgraph where all nodes have degree >= k.

    Args:
        edge_index: Edge index tensor of shape (2, num_edges)
        num_nodes: Total number of nodes

    Returns:
        Tensor of shape (num_nodes,) with k-core numbers
    '''
    # Build adjacency set for each node
    neighbors: Dict[int, set] = defaultdict(set)

    src = edge_index[0].tolist()
    dst = edge_index[1].tolist()

    for s, d in zip(src, dst):
        neighbors[s].add(d)
        neighbors[d].add(s)  # Undirected

    # Initialize degrees
    degrees = {i: len(neighbors[i]) for i in range(num_nodes)}
    core_numbers = torch.zeros(num_nodes, dtype=torch.float32)

    # Process nodes in order of increasing degree
    remaining = set(range(num_nodes))
    current_k = 0

    while remaining:
        # Find nodes with minimum degree
        min_degree = min(degrees[n] for n in remaining)
        current_k = max(current_k, min_degree)

        # Remove all nodes with degree <= current_k
        to_remove = [n for n in remaining if degrees[n] <= current_k]

        for node in to_remove:
            core_numbers[node] = current_k
            remaining.remove(node)

            # Update degrees of neighbors
            for neighbor in neighbors[node]:
                if neighbor in remaining:
                    degrees[neighbor] -= 1

    return core_numbers

def compute_node_scores(
    descriptions_parquet: str,
    relations_parquet: str,
    output_path: Optional[str] = None,
) -> Dict[str, torch.Tensor]:
    """
    Compute all node difficulty scores and save to file.

    Args:
        descriptions_parquet: Path to descriptions parquet file
        relations_parquet: Path to relations parquet file
        output_path: Optional path to save node_scores.pt

    Returns:
        Dictionary with score tensors:
          - 'degree_centrality': Degree centrality scores
          - 'pagerank': PageRank scores
          - 'kcore': K-core decomposition numbers
          - 'level': Hierarchy level (2, 3, 4, 5, 6)
          - 'composite': Weighted composite difficulty score
    """
    logger.info('Computing node difficulty scores...')

    # Load data
    df_desc = pl.read_parquet(descriptions_parquet)
    df_rel = pl.read_parquet(relations_parquet)

    num_nodes = len(df_desc)
    logger.info(f'  • Number of nodes: {num_nodes}')

    # Build edge index from child relations (hierarchical edges)
    child_edges = df_rel.filter(pl.col('relation') == 'child')
    src = torch.tensor(child_edges['idx_i'].to_numpy(), dtype=torch.long)
    dst = torch.tensor(child_edges['idx_j'].to_numpy(), dtype=torch.long)

    # Make bidirectional for centrality computation
    edge_index = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])], dim=0)

    logger.info(f'  • Number of edges: {edge_index.shape[1]}')

    # Compute centrality metrics
    logger.info('  • Computing degree centrality...')
    degree_centrality = compute_degree_centrality(edge_index, num_nodes)

    logger.info('  • Computing PageRank...')
    pagerank = compute_pagerank(edge_index, num_nodes)

    logger.info('  • Computing k-core decomposition...')
    kcore = compute_kcore(edge_index, num_nodes)

    # Extract hierarchy levels
    levels = torch.tensor(df_desc['level'].to_numpy(), dtype=torch.float32)

    # Compute composite difficulty score
    # Higher score = easier (hub nodes)
    # Lower score = harder (tail nodes)
    # Normalize each metric to [0, 1] range
    def normalize(x: torch.Tensor) -> torch.Tensor:
        x_min, x_max = x.min(), x.max()
        if x_max - x_min > 1e-8:
            return (x - x_min) / (x_max - x_min)
        return torch.zeros_like(x)

    degree_norm = normalize(degree_centrality)
    pagerank_norm = normalize(pagerank)
    kcore_norm = normalize(kcore)

    # Composite: weighted average (higher = easier/hub-like)
    composite = 0.4 * degree_norm + 0.4 * pagerank_norm + 0.2 * kcore_norm

    scores = {
        'degree_centrality': degree_centrality,
        'pagerank': pagerank,
        'kcore': kcore,
        'level': levels,
        'composite': composite,
        'num_nodes': torch.tensor([num_nodes]),
    }

    # Log statistics
    logger.info('  • Score statistics:')
    logger.info(
        f'    - Degree centrality: mean={degree_centrality.mean():.4f}, '
        f'max={degree_centrality.max():.4f}'
    )
    logger.info(f'    - PageRank: mean={pagerank.mean():.6f}, max={pagerank.max():.6f}')
    logger.info(f'    - K-core: mean={kcore.mean():.2f}, max={kcore.max():.0f}')
    logger.info(f'    - Composite: mean={composite.mean():.4f}, std={composite.std():.4f}')

    # Save to file
    if output_path:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        torch.save(scores, output_file)
        logger.info(f'  • Saved node scores to {output_path}')

    return scores

# -------------------------------------------------------------------------------------------------
# Relation Cardinality Classification
# -------------------------------------------------------------------------------------------------

def compute_relation_cardinality(
    relations_parquet: str,
    output_path: Optional[str] = None,
    threshold: float = 1.5,
) -> Dict[str, Any]:
    """
    Classify relations by cardinality (1-1, 1-N, N-1, N-N) and difficulty.

    For each relation type:
      - Compute average heads-per-tail
      - Compute average tails-per-head
      - Classify based on thresholds
      - Use relation_id as distance metric (lower = closer = easier)

    Args:
        relations_parquet: Path to relations parquet file
        output_path: Optional path to save relation_types.json
        threshold: Threshold for classifying as "many" (default: 1.5)

    Returns:
        Dictionary with relation cardinality info:
          - 'relation_types': Dict mapping relation name to cardinality class
          - 'relation_difficulty': Dict mapping relation_id to difficulty tier
          - 'statistics': Dict with detailed statistics per relation
    """
    logger.info('Computing relation cardinality classifications...')

    df_rel = pl.read_parquet(relations_parquet)

    # Get unique relations with their IDs
    relations = df_rel.select('relation', 'relation_id').unique().to_dicts()
    logger.info(f'  • Found {len(relations)} relation types')

    relation_types: Dict[str, str] = {}
    relation_difficulty: Dict[int, str] = {}
    statistics: Dict[str, Dict[str, Any]] = {}

    # Relation_id serves as distance metric: lower = closer = easier
    # Phase 1: relation_id 1 (child) - easiest, direct parent-child
    # Phase 2: relation_id 2-4 (sibling, grandchild, great-grandchild) - medium
    # Phase 3: relation_id 5+ (nephew, cousin, etc.) - harder, more distant

    for rel_info in relations:
        rel = str(rel_info['relation'])
        rel_id = int(rel_info['relation_id'])

        rel_df = df_rel.filter(pl.col('relation') == rel)

        # Count unique heads per tail
        tails_per_head_series = rel_df.group_by('idx_i').agg(
            pl.col('idx_j').n_unique().alias('count')
        )['count']
        tails_per_head_val = tails_per_head_series.mean()
        tails_per_head: float = 0.0
        if tails_per_head_val is not None:
            tails_per_head = float(str(tails_per_head_val))

        # Count unique tails per head
        heads_per_tail_series = rel_df.group_by('idx_j').agg(
            pl.col('idx_i').n_unique().alias('count')
        )['count']
        heads_per_tail_val = heads_per_tail_series.mean()
        heads_per_tail: float = 0.0
        if heads_per_tail_val is not None:
            heads_per_tail = float(str(heads_per_tail_val))

        # Classify cardinality
        many_heads = heads_per_tail > threshold
        many_tails = tails_per_head > threshold

        if many_heads and many_tails:
            cardinality = 'N-N'
        elif many_heads:
            cardinality = 'N-1'
        elif many_tails:
            cardinality = '1-N'
        else:
            cardinality = '1-1'

        # Classify difficulty tier based on relation_id (distance metric)
        # Lower relation_id = closer relationship = easier
        if rel_id <= 1:
            difficulty_tier = 'easy'  # Phase 1: direct child
        elif rel_id <= 4:
            difficulty_tier = 'medium'  # Phase 2: sibling, grandchild, great-grandchild
        else:
            difficulty_tier = 'hard'  # Phase 3+: distant relations

        relation_types[rel] = cardinality
        relation_difficulty[rel_id] = difficulty_tier
        statistics[rel] = {
            'relation_id': rel_id,
            'heads_per_tail': heads_per_tail,
            'tails_per_head': tails_per_head,
            'cardinality': cardinality,
            'difficulty_tier': difficulty_tier,
            'num_edges': len(rel_df),
        }

        logger.info(
            f'    - {rel} (id={rel_id}): {cardinality}, {difficulty_tier} '
            f'(h/t={heads_per_tail:.2f}, t/h={tails_per_head:.2f})'
        )

    result = {
        'relation_types': relation_types,
        'relation_difficulty': relation_difficulty,
        'statistics': statistics,
        'threshold': threshold,
    }

    # Save to file
    if output_path:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        logger.info(f'  • Saved relation types to {output_path}')

    return result

# -------------------------------------------------------------------------------------------------
# Distance-Based Difficulty Thresholds
# -------------------------------------------------------------------------------------------------

def compute_difficulty_thresholds(
    distances_parquet: str,
    triplets_parquet: str,
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    '''
    Compute difficulty thresholds based on distance and margin distributions.

    Metrics:
      - distance: Tree distance (lower = closer = easier)
      - relation_id: Relation type (lower = closer = easier)
      - margin: Composite metric (HIGHER = closer = easier for sampling)
          Formula: margin = 1 / (1/3 * relation_id + 2/3 * distance)

    We compute percentile-based thresholds for curriculum phases.

    Args:
        distances_parquet: Path to distances parquet file
        triplets_parquet: Path to training triplets directory
        output_path: Optional path to save thresholds

    Returns:
        Dictionary with difficulty thresholds for each phase
    '''
    logger.info('Computing difficulty thresholds...')

    # Load distance data
    df_dist = pl.read_parquet(distances_parquet)
    distances = df_dist['distance'].to_numpy()

    # Load triplet data for margin distribution
    import glob

    triplet_files = glob.glob(f'{triplets_parquet}/**/*.parquet', recursive=True)

    if triplet_files:
        # Sample a subset for efficiency
        sample_files = triplet_files[:min(50, len(triplet_files))]
        df_triplets = pl.concat([pl.read_parquet(f) for f in sample_files])

        margins = df_triplets['margin'].to_numpy()

        # Margin thresholds (higher = closer, so we use upper percentiles for easy)
        margin_p80 = float(np.percentile(margins, 80))  # Top 20% = easiest
        margin_p40 = float(np.percentile(margins, 40))  # Top 60%
        margin_p10 = float(np.percentile(margins, 10))  # Top 90%

        margin_stats = {
            'min': float(margins.min()),
            'max': float(margins.max()),
            'mean': float(margins.mean()),
            'std': float(margins.std()),
            'p10': margin_p10,
            'p40': margin_p40,
            'p80': margin_p80,
        }
    else:
        logger.warning('No triplet files found, skipping margin thresholds')
        margin_p80 = margin_p40 = margin_p10 = 0.0
        margin_stats = {}

    # Distance thresholds (lower = closer = easier)
    # Phase 1 (easy): closest 20% of pairs (low distance)
    # Phase 2 (medium): 20-60% (medium distance)
    # Phase 3 (hard): 60-90% (high distance)
    # Phase 4 (all): full distribution

    dist_p20 = float(np.percentile(distances, 20))
    dist_p60 = float(np.percentile(distances, 60))
    dist_p90 = float(np.percentile(distances, 90))

    thresholds = {
        # Distance-based thresholds (filter by distance <= threshold)
        'phase1_max_distance': dist_p20,  # Easy: distance <= p20
        'phase2_max_distance': dist_p60,  # Medium: distance <= p60
        'phase3_max_distance': dist_p90,  # Hard: distance <= p90
        'phase4_max_distance': float(distances.max()),  # All
        # Margin-based thresholds (filter by margin >= threshold for easy samples)
        # Higher margin = closer relationship
        'phase1_min_margin': margin_p80,  # Easy: margin >= p80 (top 20%)
        'phase2_min_margin': margin_p40,  # Medium: margin >= p40 (top 60%)
        'phase3_min_margin': margin_p10,  # Hard: margin >= p10 (top 90%)
        'phase4_min_margin': 0.0,  # All
        # Relation-based thresholds (filter by relation_id <= threshold)
        'phase1_max_relation': 1,  # Easy: child only
        'phase2_max_relation': 4,  # Medium: up to great-grandchild
        'phase3_max_relation': 10,  # Hard: most relations
        'phase4_max_relation': 99,  # All
        'distance_statistics': {
            'min': float(distances.min()),
            'max': float(distances.max()),
            'mean': float(distances.mean()),
            'std': float(distances.std()),
            'p20': dist_p20,
            'p60': dist_p60,
            'p90': dist_p90,
        },
        'margin_statistics': margin_stats,
    }

    logger.info(f'  • Distance range: [{distances.min():.2f}, {distances.max():.2f}]')
    logger.info(
        f'  • Distance thresholds: P1<={dist_p20:.2f}, P2<={dist_p60:.2f}, P3<={dist_p90:.2f}'
    )
    if margin_stats:
        logger.info(f'  • Margin range: [{margins.min():.4f}, {margins.max():.4f}]')
        logger.info(
            f'  • Margin thresholds: P1>={margin_p80:.4f}, P2>={margin_p40:.4f}, P3>={margin_p10:.4f}'
        )

    if output_path:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(thresholds, f, indent=2)
        logger.info(f'  • Saved difficulty thresholds to {output_path}')

    return thresholds

# -------------------------------------------------------------------------------------------------
# Main Preprocessing Function
# -------------------------------------------------------------------------------------------------

def preprocess_curriculum_data(
    descriptions_parquet: str = './data/naics_descriptions.parquet',
    relations_parquet: str = './data/naics_relations.parquet',
    distances_parquet: str = './data/naics_distances.parquet',
    triplets_parquet: str = './data/naics_training_pairs',
    output_dir: str = './data/curriculum_cache',
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any], Dict[str, Any]]:
    '''
    Run full preprocessing pipeline for curriculum learning.

    Args:
        descriptions_parquet: Path to descriptions parquet file
        relations_parquet: Path to relations parquet file
        distances_parquet: Path to distances parquet file
        triplets_parquet: Path to training triplets directory
        output_dir: Directory to save output files

    Returns:
        Tuple of (node_scores, relation_types, difficulty_thresholds)
    '''
    logger.info('=' * 60)
    logger.info('Preprocessing Curriculum Data')
    logger.info('=' * 60)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Compute node scores
    node_scores = compute_node_scores(
        descriptions_parquet,
        relations_parquet,
        output_path=str(output_path / 'node_scores.pt'),
    )

    # Compute relation cardinality
    relation_types = compute_relation_cardinality(
        relations_parquet,
        output_path=str(output_path / 'relation_types.json'),
    )

    # Compute difficulty thresholds (distance + margin)
    difficulty_thresholds = compute_difficulty_thresholds(
        distances_parquet,
        triplets_parquet,
        output_path=str(output_path / 'difficulty_thresholds.json'),
    )

    logger.info('=' * 60)
    logger.info('Preprocessing complete!')
    logger.info('=' * 60)

    return node_scores, relation_types, difficulty_thresholds

def load_node_scores(path: str) -> Dict[str, torch.Tensor]:
    '''Load precomputed node scores from file.'''
    return torch.load(path, weights_only=True)

def load_relation_types(path: str) -> Dict[str, Any]:
    '''Load precomputed relation types from file.'''
    with open(path, 'r') as f:
        return json.load(f)

def load_distance_thresholds(path: str) -> Dict[str, Any]:
    '''Load precomputed distance thresholds from file.'''
    with open(path, 'r') as f:
        return json.load(f)

# -------------------------------------------------------------------------------------------------
# CLI Entry Point
# -------------------------------------------------------------------------------------------------

def main() -> None:
    '''Command-line entry point for preprocessing.'''
    import argparse

    parser = argparse.ArgumentParser(description='Preprocess curriculum difficulty annotations')
    parser.add_argument(
        '--descriptions',
        default='./data/naics_descriptions.parquet',
        help='Path to descriptions parquet',
    )
    parser.add_argument(
        '--relations',
        default='./data/naics_relations.parquet',
        help='Path to relations parquet',
    )
    parser.add_argument(
        '--distances',
        default='./data/naics_distances.parquet',
        help='Path to distances parquet',
    )
    parser.add_argument(
        '--triplets',
        default='./data/naics_training_pairs',
        help='Path to training triplets directory',
    )
    parser.add_argument(
        '--output-dir',
        default='./data/curriculum_cache',
        help='Output directory for cached files',
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(message)s')

    preprocess_curriculum_data(
        descriptions_parquet=args.descriptions,
        relations_parquet=args.relations,
        distances_parquet=args.distances,
        triplets_parquet=args.triplets,
        output_dir=args.output_dir,
    )

if __name__ == '__main__':
    main()
