"""
Visualize training metrics from log files.
"""

import re
from pathlib import Path
from typing import List, Dict, Optional

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None


def parse_log_file(log_file: Path, stage: Optional[str] = None) -> List[Dict]:
    """Parse training log file and extract evaluation metrics."""
    
    with open(log_file, 'r') as f:
        content = f.read()
    
    metrics = []
    lines = content.split('\n')
    
    in_target_stage = False
    current_epoch = None
    current_timestamp = None
    
    for i, line in enumerate(lines):
        # Check if we're in the target stage
        if stage and stage in line:
            if 'Starting training' in line or 'Stage' in line:
                in_target_stage = True
                continue
        
        if not in_target_stage and stage:
            continue
            
        # Extract epoch number
        epoch_match = re.search(r'Running evaluation metrics \(epoch (\d+)\)', line)
        if epoch_match:
            current_epoch = int(epoch_match.group(1))
            # Extract timestamp
            timestamp_match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
            if timestamp_match:
                current_timestamp = timestamp_match.group(1)
            continue
        
        # Extract hyperbolic radius
        radius_match = re.search(
            r'Hyperbolic radius:\s+([\d.]+)\s+±\s+([\d.]+)', 
            line
        )
        if radius_match and current_epoch is not None:
            if not any(m.get('epoch') == current_epoch for m in metrics):
                metrics.append({
                    'epoch': current_epoch,
                    'timestamp': current_timestamp,
                    'radius_mean': float(radius_match.group(1)),
                    'radius_std': float(radius_match.group(2))
                })
            continue
        
        # Extract evaluation complete metrics
        eval_match = re.search(
            r'Evaluation complete: cophenetic=([\d.-]+) \((\d+) pairs\), '
            r'spearman=([\d.-]+), norm_cv=([\d.]+), dist_cv=([\d.]+), collapse=(\w+)',
            line
        )
        if eval_match and current_epoch is not None:
            # Find the metric entry for this epoch
            for m in metrics:
                if m.get('epoch') == current_epoch:
                    m.update({
                        'cophenetic': float(eval_match.group(1)),
                        'n_pairs': int(eval_match.group(2)),
                        'spearman': float(eval_match.group(3)),
                        'norm_cv': float(eval_match.group(4)),
                        'dist_cv': float(eval_match.group(5)),
                        'collapse': eval_match.group(6) == 'True'
                    })
                    break
    
    # Sort by epoch
    metrics.sort(key=lambda x: x.get('epoch', 0))
    return metrics


def create_visualizations(metrics: List[Dict], output_dir: Path, stage: str):
    """Create visualization plots for the metrics."""
    
    if not metrics:
        print("No metrics found to visualize!")
        return
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    epochs = [m['epoch'] for m in metrics]
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Hyperbolic Radius
    ax1 = plt.subplot(3, 2, 1)
    radius_means = [m.get('radius_mean', 0) for m in metrics]
    radius_stds = [m.get('radius_std', 0) for m in metrics]
    ax1.plot(epochs, radius_means, 'b-o', label='Mean', linewidth=2, markersize=6)
    ax1.fill_between(epochs, 
                     [r - s for r, s in zip(radius_means, radius_stds)],
                     [r + s for r, s in zip(radius_means, radius_stds)],
                     alpha=0.2, label='±1 std')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Hyperbolic Radius', fontsize=12)
    ax1.set_title('Hyperbolic Radius Over Time', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 2. Hierarchy Preservation Metrics
    ax2 = plt.subplot(3, 2, 2)
    cophenetic = [m.get('cophenetic', 0) for m in metrics if 'cophenetic' in m]
    spearman = [m.get('spearman', 0) for m in metrics if 'spearman' in m]
    epochs_corr = [m['epoch'] for m in metrics if 'cophenetic' in m]
    
    if epochs_corr:
        ax2.plot(epochs_corr, cophenetic, 'g-o', label='Cophenetic', linewidth=2, markersize=6)
        ax2.plot(epochs_corr, spearman, 'r-s', label='Spearman', linewidth=2, markersize=6)
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax2.axhline(y=0.7, color='g', linestyle='--', alpha=0.5, label='Target (0.7)')
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Correlation', fontsize=12)
        ax2.set_title('Hierarchy Preservation Correlations', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        ax2.set_ylim([-0.5, 1.0])
    
    # 3. Coefficient of Variation
    ax3 = plt.subplot(3, 2, 3)
    norm_cv = [m.get('norm_cv', 0) for m in metrics if 'norm_cv' in m]
    dist_cv = [m.get('dist_cv', 0) for m in metrics if 'dist_cv' in m]
    epochs_cv = [m['epoch'] for m in metrics if 'norm_cv' in m]
    
    if epochs_cv:
        ax3.plot(epochs_cv, norm_cv, 'm-o', label='Norm CV', linewidth=2, markersize=6)
        ax3.plot(epochs_cv, dist_cv, 'c-s', label='Distance CV', linewidth=2, markersize=6)
        ax3.set_xlabel('Epoch', fontsize=12)
        ax3.set_ylabel('Coefficient of Variation', fontsize=12)
        ax3.set_title('Embedding Diversity Metrics', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
    
    # 4. Hyperbolic Radius vs Hierarchy Correlation
    ax4 = plt.subplot(3, 2, 4)
    if epochs_corr and cophenetic:
        scatter = ax4.scatter(radius_means[:len(cophenetic)], cophenetic, 
                             c=epochs_corr, cmap='viridis', s=100, alpha=0.6)
        ax4.set_xlabel('Hyperbolic Radius (mean)', fontsize=12)
        ax4.set_ylabel('Cophenetic Correlation', fontsize=12)
        ax4.set_title('Radius vs Hierarchy Preservation', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax4, label='Epoch')
    
    # 5. Radius Standard Deviation
    ax5 = plt.subplot(3, 2, 5)
    ax5.plot(epochs, radius_stds, 'orange', marker='o', linewidth=2, markersize=6)
    ax5.set_xlabel('Epoch', fontsize=12)
    ax5.set_ylabel('Radius Std Dev', fontsize=12)
    ax5.set_title('Hyperbolic Radius Spread', fontsize=14, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    
    # 6. Summary Statistics Table
    ax6 = plt.subplot(3, 2, 6)
    ax6.axis('off')
    
    if metrics:
        latest = metrics[-1]
        summary_text = f"""
        LATEST METRICS (Epoch {latest.get('epoch', 'N/A')})
        
        Hyperbolic Radius:
          Mean: {latest.get('radius_mean', 0):.4f}
          Std:  {latest.get('radius_std', 0):.4f}
        
        Hierarchy Preservation:
          Cophenetic: {latest.get('cophenetic', 0):.4f}
          Spearman:   {latest.get('spearman', 0):.4f}
        
        Diversity:
          Norm CV:     {latest.get('norm_cv', 0):.4f}
          Distance CV: {latest.get('dist_cv', 0):.4f}
        
        Status:
          Collapse: {'Yes' if latest.get('collapse', False) else 'No'}
        """
        
        if len(metrics) > 1:
            first = metrics[0]
            trends = f"""
        TRENDS (Epoch {first.get('epoch', 'N/A')} → {latest.get('epoch', 'N/A')})
        
        Radius:      {first.get('radius_mean', 0):.4f} → {latest.get('radius_mean', 0):.4f}
        Cophenetic:  {first.get('cophenetic', 0):.4f} → {latest.get('cophenetic', 0):.4f}
        Spearman:    {first.get('spearman', 0):.4f} → {latest.get('spearman', 0):.4f}
        Distance CV: {first.get('dist_cv', 0):.4f} → {latest.get('dist_cv', 0):.4f}
        """
            summary_text += trends
        
        ax6.text(0.1, 0.5, summary_text, fontsize=10, 
                family='monospace', verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle(f'{stage.upper()} Training Metrics', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    output_file = output_dir / f'{stage}_metrics.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved visualization to: {output_file}")
    
    plt.close()


def print_analysis(metrics: List[Dict], stage: str):
    """Print detailed analysis of the metrics."""
    
    if not metrics:
        print("No metrics to analyze!")
        return
    
    print("\n" + "=" * 90)
    print(f"ANALYSIS: {stage.upper()}")
    print("=" * 90)
    
    # Hyperbolic Radius Analysis
    radius_means = [m.get('radius_mean', 0) for m in metrics]
    if radius_means:
        print(f"\n📊 HYPERBOLIC RADIUS:")
        print(f"   Initial: {radius_means[0]:.4f}")
        print(f"   Latest:  {radius_means[-1]:.4f}")
        print(f"   Change:  {radius_means[-1] - radius_means[0]:+.4f} ({((radius_means[-1]/radius_means[0] - 1) * 100):+.1f}%)")
        
        if radius_means[-1] > 20:
            print(f"   ⚠️  WARNING: Radius is getting large (>20). Monitor for stability.")
        elif radius_means[-1] > 10:
            print(f"   ℹ️  INFO: Radius is moderate (10-20). This is reasonable.")
        else:
            print(f"   ✓ Radius is in normal range (<10).")
    
    # Hierarchy Preservation Analysis
    cophenetic = [m.get('cophenetic', 0) for m in metrics if 'cophenetic' in m]
    spearman = [m.get('spearman', 0) for m in metrics if 'spearman' in m]
    
    if cophenetic:
        print(f"\n📈 HIERARCHY PRESERVATION:")
        print(f"   Cophenetic: {cophenetic[0]:.4f} → {cophenetic[-1]:.4f} ({cophenetic[-1] - cophenetic[0]:+.4f})")
        print(f"   Spearman:   {spearman[0]:.4f} → {spearman[-1]:.4f} ({spearman[-1] - spearman[0]:+.4f})")
        
        if cophenetic[-1] > 0.7:
            print(f"   ✓ Excellent hierarchy preservation!")
        elif cophenetic[-1] > 0.5:
            print(f"   ℹ️  Good hierarchy preservation, but could improve.")
        elif cophenetic[-1] > 0.3:
            print(f"   ⚠️  Moderate hierarchy preservation. Model may need more training.")
        else:
            print(f"   ⚠️  WARNING: Low hierarchy preservation. Consider:")
            print(f"      - Checking if ground truth distances are correct")
            print(f"      - Verifying training data quality")
            print(f"      - Adjusting learning rate or loss function")
    
    # Collapse Detection
    collapse_flags = [m.get('collapse', False) for m in metrics if 'collapse' in m]
    if collapse_flags:
        if any(collapse_flags):
            print(f"\n⚠️  COLLAPSE DETECTED:")
            collapsed_epochs = [m['epoch'] for m in metrics if m.get('collapse', False)]
            print(f"   Collapse occurred at epochs: {collapsed_epochs}")
        else:
            print(f"\n✓ NO COLLAPSE DETECTED")
            print(f"   All embeddings show good diversity.")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    
    if cophenetic and cophenetic[-1] < 0.5:
        print(f"   1. Hierarchy correlations are low. This could be because:")
        print(f"      - Model is still learning (only {metrics[-1].get('epoch', 0)} epochs completed)")
        print(f"      - Hyperbolic space may need more time to organize hierarchy")
        print(f"      - Consider checking if evaluation sample size is sufficient")
    
    if radius_means and radius_means[-1] > 15:
        print(f"   2. Hyperbolic radius is growing rapidly. Monitor for:")
        print(f"      - Numerical stability issues")
        print(f"      - Whether this growth correlates with better metrics")
    
    if cophenetic and len(cophenetic) > 3:
        recent_trend = cophenetic[-3:]
        if all(recent_trend[i] <= recent_trend[i+1] for i in range(len(recent_trend)-1)):
            print(f"   3. Cophenetic correlation is improving! Continue training.")
        elif all(recent_trend[i] >= recent_trend[i+1] for i in range(len(recent_trend)-1)):
            print(f"   3. ⚠️  Cophenetic correlation is declining. Consider:")
            print(f"      - Early stopping if this continues")
            print(f"      - Learning rate reduction")
    
    print()


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize training metrics')
    parser.add_argument('--stage', type=str, default='02_text',
                       help='Stage name to filter (default: 02_text)')
    parser.add_argument('--log-file', type=Path, 
                       default=project_root / 'logs' / 'train_sequential.log',
                       help='Path to log file')
    parser.add_argument('--output-dir', type=Path,
                       default=project_root / 'outputs' / 'visualizations',
                       help='Output directory for plots')
    
    args = parser.parse_args()
    
    if not args.log_file.exists():
        print(f"Error: Log file not found: {args.log_file}")
        sys.exit(1)
    
    print(f"Parsing metrics from: {args.log_file}")
    metrics = parse_log_file(args.log_file, stage=args.stage)
    
    if not metrics:
        print(f"No metrics found for stage '{args.stage}' in log file!")
        sys.exit(1)
    
    print(f"Found {len(metrics)} evaluation epochs")
    
    # Create visualizations
    create_visualizations(metrics, args.output_dir, args.stage)
    
    # Print analysis
    print_analysis(metrics, args.stage)
    
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


if __name__ == '__main__':
    main()

