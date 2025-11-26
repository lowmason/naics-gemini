'''
Visualize training metrics from log files.
'''

# -------------------------------------------------------------------------------------------------
# Imports
# -------------------------------------------------------------------------------------------------

import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Define project root
project_root = Path(__file__).parent.parent.parent.parent

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None


# -------------------------------------------------------------------------------------------------
# Parse log file
# -------------------------------------------------------------------------------------------------

def parse_log_file(log_file: Path, stage: Optional[str] = None) -> List[Dict]:
    '''Parse training log file and extract evaluation metrics.'''
    
    with open(log_file, 'r') as f:
        content = f.read()
    
    metrics = []
    lines = content.split('\n')
    
    in_target_stage = False
    current_epoch = None
    current_timestamp = None
    
    for i, line in enumerate(lines):
        # Check if we're entering the target stage
        if stage and stage in line:
            # Look for "Using curriculum" to detect stage start
            if 'Using curriculum' in line and stage in line:
                in_target_stage = True
                continue
            # Also check for checkpoint paths that contain the stage name
            elif 'checkpoint' in line.lower() and stage in line:
                # This indicates we're in the stage
                if not in_target_stage:
                    in_target_stage = True
                continue
        
        # Check if we're leaving the target stage (new stage starts)
        if in_target_stage and stage:
            # Look for a different stage starting
            for other_stage in ['01_text', '02_text', '03_text', '04_text', '05_text']:
                if other_stage != stage and 'Using curriculum' in line and other_stage in line:
                    in_target_stage = False
                    break
        
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
        
        # Extract cophenetic correlation
        cophenetic_match = re.search(
            r'Hierarchy preservation: cophenetic=([\d.-]+) \((\d+) pairs\)',
            line
        )
        if cophenetic_match and current_epoch is not None:
            for m in metrics:
                if m.get('epoch') == current_epoch:
                    m.update({
                        'cophenetic': float(cophenetic_match.group(1)),
                        'n_pairs': int(cophenetic_match.group(2))
                    })
                    break
        
        # Extract Norm CV
        norm_cv_match = re.search(r'Norm CV:\s+([\d.]+)', line)
        if norm_cv_match and current_epoch is not None:
            for m in metrics:
                if m.get('epoch') == current_epoch:
                    m['norm_cv'] = float(norm_cv_match.group(1))
                    break
        
        # Extract Distance CV
        dist_cv_match = re.search(r'Distance CV:\s+([\d.]+)', line)
        if dist_cv_match and current_epoch is not None:
            for m in metrics:
                if m.get('epoch') == current_epoch:
                    m['dist_cv'] = float(dist_cv_match.group(1))
                    break
        
        # Extract Collapse
        collapse_match = re.search(r'Collapse:\s+(\w+)', line)
        if collapse_match and current_epoch is not None:
            for m in metrics:
                if m.get('epoch') == current_epoch:
                    m['collapse'] = collapse_match.group(1) == 'True'
                    break
    
    # Sort by epoch
    metrics.sort(key=lambda x: x.get('epoch', 0))
    return metrics


# -------------------------------------------------------------------------------------------------
# Create visualizations
# -------------------------------------------------------------------------------------------------

def create_visualizations(metrics: List[Dict], output_dir: Path, stage: str):
    
    '''Create visualization plots for the metrics.'''
    
    if not HAS_MATPLOTLIB:
        print('Matplotlib is not available. Cannot create visualizations.')
        return
    
    assert plt is not None  # Type guard: plt is guaranteed to be available here
    
    if not metrics:
        print('No metrics found to visualize!')
        return
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    epochs = [m['epoch'] for m in metrics]
    
    # Create figure with subplots
    _ = plt.figure(figsize=(16, 12))
    
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
    
    # 2. Training and Validation Loss
    ax2 = plt.subplot(3, 2, 2)
    train_loss = [m.get('train_loss', None) for m in metrics]
    val_loss = [m.get('val_loss', None) for m in metrics]
    epochs_loss = [m['epoch'] for m in metrics if 'train_loss' in m or 'val_loss' in m]
    
    if epochs_loss:
        if any(loss is not None for loss in train_loss):
            train_loss_clean = [loss for loss in train_loss if loss is not None]
            epochs_train = [
                e for e, loss in zip(epochs_loss, train_loss) if loss is not None
            ]
            ax2.plot(
                epochs_train, train_loss_clean, 'b-o',
                label='Train Loss', linewidth=2, markersize=6
            )
        if any(loss is not None for loss in val_loss):
            val_loss_clean = [loss for loss in val_loss if loss is not None]
            epochs_val = [
                e for e, loss in zip(epochs_loss, val_loss) if loss is not None
            ]
            ax2.plot(epochs_val, val_loss_clean, 'r-s', label='Val Loss', linewidth=2, markersize=6)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Loss', fontsize=12)
        ax2.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        ax2.set_yscale('log')
    
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
    
    # 4. Hierarchy Preservation (Cophenetic)
    ax4 = plt.subplot(3, 2, 4)
    cophenetic = [m.get('cophenetic', 0) for m in metrics if 'cophenetic' in m]
    epochs_corr = [m['epoch'] for m in metrics if 'cophenetic' in m]
    
    if epochs_corr and cophenetic:
        ax4.plot(epochs_corr, cophenetic, 'g-o', label='Cophenetic', linewidth=2, markersize=6)
        ax4.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax4.axhline(y=0.7, color='g', linestyle='--', alpha=0.5, label='Target (0.7)')
        ax4.set_xlabel('Epoch', fontsize=12)
        ax4.set_ylabel('Cophenetic Correlation', fontsize=12)
        ax4.set_title('Hierarchy Preservation', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        ax4.set_ylim((-0.5, 1.0))
    
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
        
        Loss:
          Train: {latest.get('train_loss', 'N/A')}
          Val:   {latest.get('val_loss', 'N/A')}
        
        Hierarchy Preservation:
          Cophenetic: {(
              f"{latest.get('cophenetic', 0):.4f}"
              if 'cophenetic' in latest and latest.get('cophenetic') is not None
              else 'N/A'
          )}
        
        Diversity:
          Norm CV:     {(
              f"{latest.get('norm_cv', 0):.4f}"
              if 'norm_cv' in latest and latest.get('norm_cv') is not None
              else 'N/A'
          )}
          Distance CV: {(
              f"{latest.get('dist_cv', 0):.4f}"
              if 'dist_cv' in latest and latest.get('dist_cv') is not None
              else 'N/A'
          )}
        
        Status:
          Collapse: {'Yes' if latest.get('collapse', False) else 'No'}
        """
        
        if len(metrics) > 1:
            first = metrics[0]
            trends = f"""
        TRENDS (Epoch {first.get('epoch', 'N/A')} → {latest.get('epoch', 'N/A')})
        
        Radius:      {first.get('radius_mean', 0):.4f} → {latest.get('radius_mean', 0):.4f}
        Cophenetic:  {(
            f"{first.get('cophenetic', 0):.4f}"
            if 'cophenetic' in first and first.get('cophenetic') is not None
            else 'N/A'
        )} → {(
            f"{latest.get('cophenetic', 0):.4f}"
            if 'cophenetic' in latest and latest.get('cophenetic') is not None
            else 'N/A'
        )}
        Train Loss:  {first.get('train_loss', 'N/A')} → {latest.get('train_loss', 'N/A')}
        Val Loss:    {first.get('val_loss', 'N/A')} → {latest.get('val_loss', 'N/A')}
        Distance CV: {(
            f"{first.get('dist_cv', 0):.4f}"
            if 'dist_cv' in first and first.get('dist_cv') is not None
            else 'N/A'
        )} → {(
            f"{latest.get('dist_cv', 0):.4f}"
            if 'dist_cv' in latest and latest.get('dist_cv') is not None
            else 'N/A'
        )}
        """
            summary_text += trends
        
        ax6.text(0.1, 0.5, summary_text, fontsize=10, 
                family='monospace', verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle(f'{stage.upper()} Training Metrics', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=(0, 0, 1, 0.99))  
    
    output_file = output_dir / f'{stage}_metrics.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f'✓ Saved visualization to: {output_file}')
    
    plt.close()


# -------------------------------------------------------------------------------------------------
# Print analysis
# -------------------------------------------------------------------------------------------------

def print_analysis(metrics: List[Dict], stage: str):
    
    '''Print detailed analysis of the metrics.'''
    
    if not metrics:
        print('No metrics to analyze!')
        return
    
    print('\n' + '=' * 90)
    print(f'ANALYSIS: {stage.upper()}')
    print('=' * 90)
    
    # Hyperbolic Radius Analysis
    radius_means = [m.get('radius_mean', 0) for m in metrics]
    if radius_means:
        print('\n📊 HYPERBOLIC RADIUS:')
        print(f'   Initial: {radius_means[0]:.4f}')
        print(f'   Latest:  {radius_means[-1]:.4f}')
        change_pct = ((radius_means[-1]/radius_means[0] - 1) * 100)
        print(f'   Change:  {radius_means[-1] - radius_means[0]:+.4f} ({change_pct:+.1f}%)')
        
        if radius_means[-1] > 20:
            print('   ⚠️  WARNING: Radius is getting large (>20). Monitor for stability.')
        elif radius_means[-1] > 10:
            print('   ℹ️  INFO: Radius is moderate (10-20). This is reasonable.')
        else:
            print('   ✓ Radius is in normal range (<10).')
    
    # Loss Analysis
    train_losses = [
        m.get('train_loss')
        for m in metrics
        if 'train_loss' in m and m.get('train_loss') is not None
    ]
    val_losses = [
        m.get('val_loss')
        for m in metrics
        if 'val_loss' in m and m.get('val_loss') is not None
    ]
    
    if train_losses:
        print('\n📉 TRAINING LOSS:')
        initial_loss = train_losses[0]
        latest_loss = train_losses[-1]
        assert initial_loss is not None and latest_loss is not None
        print(f'   Initial: {initial_loss:.6f}')
        print(f'   Latest:  {latest_loss:.6f}')
        change_pct = ((latest_loss / initial_loss - 1) * 100)
        print(f'   Change:  {latest_loss - initial_loss:+.6f} ({change_pct:+.1f}%)')
    
    if val_losses:
        print('\n📉 VALIDATION LOSS:')
        initial_loss = val_losses[0]
        latest_loss = val_losses[-1]
        assert initial_loss is not None and latest_loss is not None
        print(f'   Initial: {initial_loss:.6f}')
        print(f'   Latest:  {latest_loss:.6f}')
        change_pct = ((latest_loss / initial_loss - 1) * 100)
        print(f'   Change:  {latest_loss - initial_loss:+.6f} ({change_pct:+.1f}%)')
        if latest_loss < initial_loss:
            print('   ✓ Validation loss is decreasing - model is learning!')
        else:
            print('   ⚠️  Validation loss is increasing - may be overfitting')
    
    # Hierarchy Preservation Analysis
    cophenetic = [m.get('cophenetic', 0) for m in metrics if 'cophenetic' in m]
    
    if cophenetic:
        print('\n📈 HIERARCHY PRESERVATION:')
        cophenetic_change = cophenetic[-1] - cophenetic[0]
        print(
            f'   Cophenetic: {cophenetic[0]:.4f} → {cophenetic[-1]:.4f} '
            f'({cophenetic_change:+.4f})'
        )
        
        if cophenetic[-1] > 0.7:
            print('   ✓ Excellent hierarchy preservation!')
        elif cophenetic[-1] > 0.5:
            print('   ℹ️  Good hierarchy preservation, but could improve.')
        elif cophenetic[-1] > 0.3:
            print('   ⚠️  Moderate hierarchy preservation. Model may need more training.')
        else:
            print('   ⚠️  WARNING: Low hierarchy preservation. Consider:')
            print('      - Checking if ground truth distances are correct')
            print('      - Verifying training data quality')
            print('      - Adjusting learning rate or loss function')
    
    # Collapse Detection
    collapse_flags = [m.get('collapse', False) for m in metrics if 'collapse' in m]
    if collapse_flags:
        if any(collapse_flags):
            print('\n⚠️  COLLAPSE DETECTED:')
            collapsed_epochs = [m['epoch'] for m in metrics if m.get('collapse', False)]
            print(f'   Collapse occurred at epochs: {collapsed_epochs}')
        else:
            print('\n✓ NO COLLAPSE DETECTED')
            print('   All embeddings show good diversity.')
    
    # Recommendations
    print('\n💡 RECOMMENDATIONS:')
    
    if cophenetic and cophenetic[-1] < 0.5:
        print('   1. Hierarchy correlations are low. This could be because:')
        epoch_count = metrics[-1].get('epoch', 0)
        print(f'      - Model is still learning (only {epoch_count} epochs completed)')
        print('      - Hyperbolic space may need more time to organize hierarchy')
        print('      - Consider checking if evaluation sample size is sufficient')
    
    if radius_means and radius_means[-1] > 15:
        print('   2. Hyperbolic radius is growing rapidly. Monitor for:')
        print('      - Numerical stability issues')
        print('      - Whether this growth correlates with better metrics')
    
    if cophenetic and len(cophenetic) > 3:
        recent_trend = cophenetic[-3:]
        if all(recent_trend[i] <= recent_trend[i+1] for i in range(len(recent_trend)-1)):
            print('   3. Cophenetic correlation is improving! Continue training.')
        elif all(recent_trend[i] >= recent_trend[i+1] for i in range(len(recent_trend)-1)):
            print('   3. ⚠️  Cophenetic correlation is declining. Consider:')
            print('      - Early stopping if this continues')
            print('      - Learning rate reduction')
    
    print()


# -------------------------------------------------------------------------------------------------
# Main entry point
# -------------------------------------------------------------------------------------------------

def main():

    '''Main entry point.'''
    
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
        print(f'Error: Log file not found: {args.log_file}')
        sys.exit(1)
    
    print(f'Parsing metrics from: {args.log_file}')
    metrics = parse_log_file(args.log_file, stage=args.stage)
    
    if not metrics:
        print(f"No metrics found for stage '{args.stage}' in log file!")
        sys.exit(1)
    
    print(f'Found {len(metrics)} evaluation epochs')
    
    # Create visualizations
    create_visualizations(metrics, args.output_dir, args.stage)
    
    # Print analysis
    print_analysis(metrics, args.stage)
    
    # Print summary table
    print('\n' + '=' * 90)
    print('METRICS SUMMARY TABLE')
    print('=' * 90)
    header = (
        f"{'Epoch':<8} {'Radius':<15} {'Train Loss':<12} "
        f"{'Val Loss':<12} {'Cophenetic':<12} {'Dist CV':<10} {'Collapse':<10}"
    )
    print(header)
    print('-' * 90)
    for m in metrics:
        epoch = m.get('epoch', 'N/A')
        radius = f"{m.get('radius_mean', 0):.2f}±{m.get('radius_std', 0):.2f}"
        train_loss = (
            f"{m.get('train_loss', 0):.6f}"
            if 'train_loss' in m and m.get('train_loss') is not None
            else 'N/A'
        )
        val_loss = (
            f"{m.get('val_loss', 0):.6f}"
            if 'val_loss' in m and m.get('val_loss') is not None
            else 'N/A'
        )
        cophenetic = f"{m.get('cophenetic', 0):.4f}" if 'cophenetic' in m else 'N/A'
        dist_cv = f"{m.get('dist_cv', 0):.4f}" if 'dist_cv' in m else 'N/A'
        collapse = 'Yes' if m.get('collapse', False) else 'No'
        print(
            f'{epoch:<8} {radius:<15} {train_loss:<12} {val_loss:<12} '
            f'{cophenetic:<12} {dist_cv:<10} {collapse:<10}'
        )
    print()


if __name__ == '__main__':
    main()

