'''
Metrics visualization and investigation tools.

Provides functions to visualize training metrics and investigate hierarchy correlations.
'''

import sys
from pathlib import Path
from typing import List, Dict, Optional

# Import matplotlib with non-interactive backend
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# Import the functions from the original scripts
# We'll import them directly since they're already in the package
try:
    from naics_embedder.tools._visualize_metrics import (
        parse_log_file,
        create_visualizations,
        print_analysis
    )
    HAS_VISUALIZE = True
except ImportError:
    HAS_VISUALIZE = False

try:
    from naics_embedder.tools._investigate_hierarchy import (
        analyze_ground_truth_distances,
        check_evaluation_sample_size,
        analyze_correlation_issues
    )
    HAS_INVESTIGATE = True
except ImportError:
    HAS_INVESTIGATE = False


def visualize_metrics(
    stage: str = '02_text',
    log_file: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    project_root: Optional[Path] = None
) -> Dict:
    '''
    Visualize training metrics from log files.
    
    Args:
        stage: Stage name to filter (e.g., '02_text')
        log_file: Path to log file (default: logs/train_sequential.log)
        output_dir: Output directory for plots (default: outputs/visualizations/)
        project_root: Project root directory (default: current working directory)
        
    Returns:
        Dictionary with metrics and output file path
    '''
    if project_root is None:
        project_root = Path.cwd()
    
    if log_file is None:
        log_file = project_root / 'logs' / 'train_sequential.log'
    
    if output_dir is None:
        output_dir = project_root / 'outputs' / 'visualizations'
    
    if not log_file.exists():
        raise FileNotFoundError(f"Log file not found: {log_file}")
    
    if not HAS_VISUALIZE:
        raise ImportError("Visualization tools not available. Missing dependencies.")
    
    # Parse metrics
    metrics = parse_log_file(log_file, stage=stage)
    
    if not metrics:
        raise ValueError(f"No metrics found for stage '{stage}' in log file!")
    
    # Create visualizations
    if HAS_MATPLOTLIB:
        create_visualizations(metrics, output_dir, stage)
        output_file = output_dir / f'{stage}_metrics.png'
    else:
        output_file = None
        print("⚠️  Matplotlib not available. Skipping visualization creation.")
    
    # Print analysis
    print_analysis(metrics, stage)
    
    # Print summary table
    print("\n" + "=" * 90)
    print("METRICS SUMMARY TABLE")
    print("=" * 90)
    print(f"{'Epoch':<8} {'Radius':<15} {'Cophenetic':<12} {'Spearman':<12} {'Dist CV':<10} {'Collapse':<10}")
    print("-" * 90)
    for m in metrics:
        epoch = m.get('epoch', 'N/A')
        radius = f"{m.get('radius_mean', 0):.2f}±{m.get('radius_std', 0):.2f}"
        cophenetic = f"{m.get('cophenetic', 0):.4f}" if 'cophenetic' in m else 'N/A'
        spearman = f"{m.get('spearman', 0):.4f}" if 'spearman' in m else 'N/A'
        dist_cv = f"{m.get('dist_cv', 0):.4f}" if 'dist_cv' in m else 'N/A'
        collapse = 'Yes' if m.get('collapse', False) else 'No'
        print(f"{epoch:<8} {radius:<15} {cophenetic:<12} {spearman:<12} {dist_cv:<10} {collapse:<10}")
    print()
    
    return {
        'metrics': metrics,
        'output_file': output_file,
        'stage': stage,
        'num_epochs': len(metrics)
    }


def investigate_hierarchy(
    distance_matrix_path: Optional[Path] = None,
    config_path: Optional[Path] = None,
    project_root: Optional[Path] = None
) -> Dict:
    '''
    Investigate why hierarchy preservation correlations might be low.
    
    Args:
        distance_matrix_path: Path to ground truth distance matrix
        config_path: Path to config file
        project_root: Project root directory
        
    Returns:
        Dictionary with investigation results
    '''
    if project_root is None:
        project_root = Path.cwd()
    
    if distance_matrix_path is None:
        distance_matrix_path = project_root / 'data' / 'naics_distance_matrix.parquet'
    
    if config_path is None:
        config_path = project_root / 'conf' / 'config.yaml'
    
    if not HAS_INVESTIGATE:
        raise ImportError("Investigation tools not available. Missing dependencies.")
    
    results = {}
    
    # Analyze ground truth distances
    distances = analyze_ground_truth_distances(distance_matrix_path)
    results['distance_matrix_analyzed'] = distances is not None
    
    # Check evaluation config
    if config_path.exists():
        eval_sample_size = check_evaluation_sample_size(config_path)
        results['eval_sample_size'] = eval_sample_size
    else:
        results['eval_sample_size'] = None
    
    # Provide analysis
    analyze_correlation_issues()
    
    return results

