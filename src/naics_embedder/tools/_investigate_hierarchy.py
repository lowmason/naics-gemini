'''
Investigate hierarchy preservation metrics.

This script helps diagnose why hierarchy correlations might be low.
'''

from pathlib import Path
from typing import Optional

import torch

try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


def analyze_ground_truth_distances(distance_matrix_path: Path):
    '''Analyze the ground truth distance matrix.'''
    
    print('=' * 80)
    print('GROUND TRUTH DISTANCE MATRIX ANALYSIS')
    print('=' * 80)
    
    if not distance_matrix_path.exists():
        print(f'❌ Distance matrix not found: {distance_matrix_path}')
        return None
    
    if not HAS_POLARS:
        print('\n⚠️  Cannot analyze distance matrix without polars.')
        print(f'   File exists: {distance_matrix_path}')
        print(f'   Size: {distance_matrix_path.stat().st_size / 1024 / 1024:.2f} MB')
        return None
    
    print(f'\n📁 Loading: {distance_matrix_path}')
    df = pl.read_parquet(distance_matrix_path)
    
    print(f'   Shape: {df.shape}')
    print(f'   Columns: {len(df.columns)}')
    
    # Convert to torch tensor
    distances = df.to_torch()
    
    # Get upper triangular values (excluding diagonal)
    n = distances.shape[0]
    triu_indices = torch.triu_indices(n, n, offset=1)
    upper_tri_values = distances[triu_indices[0], triu_indices[1]]
    
    print('\n📊 Distance Statistics:')
    print(f'   Mean:   {upper_tri_values.mean().item():.4f}')
    print(f'   Std:    {upper_tri_values.std().item():.4f}')
    print(f'   Min:    {upper_tri_values.min().item():.4f}')
    print(f'   Max:    {upper_tri_values.max().item():.4f}')
    print(f'   Median: {upper_tri_values.median().item():.4f}')
    
    # Check for zeros (self-distances should be 0, but pairs shouldn't)
    zero_pairs = (upper_tri_values == 0).sum().item()
    if zero_pairs > 0:
        print(f'\n⚠️  WARNING: Found {zero_pairs} zero-distance pairs (excluding diagonal)')
    else:
        print('\n✓ No zero-distance pairs found (good)')
    
    # Check distance distribution
    print('\n📈 Distance Distribution:')
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    for p in percentiles:
        val = torch.quantile(upper_tri_values, p/100).item()
        print(f'   {p:2d}th percentile: {val:.4f}')
    
    return distances


def check_evaluation_sample_size(config_path: Path):
    '''Check evaluation sample size configuration.'''
    
    print('\n' + '=' * 80)
    print('EVALUATION CONFIGURATION')
    print('=' * 80)
    
    if not HAS_YAML:
        print('⚠️  YAML not available. Cannot check config.')
        return None
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    eval_sample_size = config.get('model', {}).get('eval_sample_size', 500)
    print(f'\n📊 Evaluation Sample Size: {eval_sample_size}')
    
    if eval_sample_size < 100:
        print('   ⚠️  WARNING: Sample size is small. This may affect correlation accuracy.')
    elif eval_sample_size < 500:
        print('   ℹ️  INFO: Sample size is moderate. Consider increasing for more stable metrics.')
    else:
        print('   ✓ Sample size is reasonable.')
    
    return eval_sample_size


def analyze_correlation_issues():
    '''Provide analysis of potential correlation issues.'''
    
    print('\n' + '=' * 80)
    print('POTENTIAL ISSUES WITH LOW HIERARCHY CORRELATIONS')
    print('=' * 80)
    
    print("""
    Based on your metrics (Cophenetic ~0.22, Spearman ~0.00), here are potential causes:
    
    1. EARLY TRAINING STAGE
       - You're only at epoch 5 of 20
       - Hyperbolic embeddings need time to organize
       - The model is still learning the basic structure
       
    2. SAMPLE SIZE IN EVALUATION
       - Small evaluation samples can lead to noisy correlations
       - Check if eval_sample_size is sufficient (should be >= 500)
       
    3. HYPERBOLIC RADIUS GROWTH
       - Your radius is growing rapidly (2.23 → 16.29)
       - This suggests the model is exploring the space
       - Correlations may improve as the space stabilizes
       
    4. DISTANCE METRIC MISMATCH
       - Ground truth uses tree distances
       - Model uses Lorentzian distances in hyperbolic space
       - These may not align perfectly, especially early in training
       
    5. CURRICULUM STAGE DIFFICULTY
       - Stage 02 focuses on levels [4, 5, 6] and relations [1, 2, 3, 4]
       - This is more specific than stage 01
       - Model may need more epochs to learn this structure
       
    6. GROUND TRUTH QUALITY
       - Verify that ground truth distances are computed correctly
       - Check if distance matrix has proper structure
       - Ensure evaluation codes match ground truth codes
    """)


def main(project_root: Optional[Path] = None):
    '''Main entry point.'''
    if project_root is None:
        project_root = Path.cwd()
    
    # Check ground truth distances
    distance_matrix_path = project_root / 'data' / 'naics_distance_matrix.parquet'
    analyze_ground_truth_distances(distance_matrix_path)
    
    # Check evaluation config
    config_path = project_root / 'conf' / 'config.yaml'
    if config_path.exists():
        check_evaluation_sample_size(config_path)
    else:
        print(f'\n⚠️  Config file not found: {config_path}')
    
    # Provide analysis
    analyze_correlation_issues()
    
    # Recommendations
    print('\n' + '=' * 80)
    print('RECOMMENDATIONS')
    print('=' * 80)
    print("""
    1. CONTINUE TRAINING
       - You're only 25% through stage 02 (5/20 epochs)
       - Correlations often improve in later epochs
       - Monitor trends, not just absolute values
       
    2. MONITOR METRICS
       - Watch if cophenetic correlation starts increasing again
       - Check if hyperbolic radius stabilizes
       - Ensure no collapse occurs
       
    3. CHECK EVALUATION SETUP
       - Verify eval_sample_size is sufficient (>= 500)
       - Ensure ground truth distances are loaded correctly
       - Check that evaluation codes match ground truth
       
    4. CONSIDER ADJUSTMENTS (if metrics don't improve)
       - Reduce learning rate if correlations plateau
       - Check if loss is decreasing (good sign)
       - Verify training data quality
       
    5. COMPARE WITH STAGE 01
       - Check if stage 01 had better correlations
       - Stage 02 may be more difficult (more specific filtering)
       - This is expected in curriculum learning
    """)


if __name__ == '__main__':
    from pathlib import Path
    main(Path.cwd())

