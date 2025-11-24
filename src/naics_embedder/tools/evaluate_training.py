'''
Comprehensive evaluation script for training stages.
Extracts metrics from logs and TensorBoard, generates visualizations, and creates reports.
'''

import sys
from pathlib import Path
from typing import Dict, List, Optional
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

from naics_embedder.tools._visualize_metrics import (
    parse_log_file,
    create_visualizations,
    print_analysis,
    HAS_MATPLOTLIB
)

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False
    print("Warning: tensorboard not available. Will only extract from log files.")


def extract_tensorboard_metrics(tb_dir: Path, stage: str) -> Dict[int, Dict[str, float]]:
    '''
    Extract metrics from TensorBoard event files.
    
    Returns:
        Dictionary mapping epoch -> {metric_name: value}
    '''
    if not HAS_TENSORBOARD:
        return {}
    
    metrics = {}
    
    # Find the latest version directory for this stage
    stage_dir = tb_dir / stage
    if not stage_dir.exists():
        print(f"Warning: TensorBoard directory not found: {stage_dir}")
        return metrics
    
    # Find all version directories
    version_dirs = sorted([d for d in stage_dir.iterdir() if d.is_dir() and d.name.startswith('version_')])
    if not version_dirs:
        print(f"Warning: No version directories found in {stage_dir}")
        return metrics
    
    # Use the latest version
    latest_version = version_dirs[-1]
    event_files = list(latest_version.glob('events.out.tfevents.*'))
    
    if not event_files:
        print(f"Warning: No event files found in {latest_version}")
        return metrics
    
    # Use the first event file (usually there's only one)
    event_file = event_files[0]
    
    try:
        ea = EventAccumulator(str(event_file.parent))
        ea.Reload()
        
        # Get scalar tags
        scalar_tags = ea.Tags()['scalars']
        
        # Extract epoch-end values for train and val loss
        train_loss_tag = None
        val_loss_tag = None
        
        for tag in scalar_tags:
            if 'train' in tag.lower() and 'loss' in tag.lower() and 'total' in tag.lower():
                train_loss_tag = tag
            elif 'val' in tag.lower() and 'loss' in tag.lower() and 'contrastive' in tag.lower():
                val_loss_tag = tag
        
        # Extract values - need to map steps to epochs
        # PyTorch Lightning logs validation at end of each epoch
        # We'll extract the last value for each epoch
        if train_loss_tag:
            train_loss_scalars = ea.Scalars(train_loss_tag)
            # Group by epoch (assuming steps are logged per epoch)
            # For now, we'll take the last value per "epoch-like" step
            for scalar in train_loss_scalars:
                step = scalar.step
                # Try to infer epoch from step (this may need adjustment based on logging frequency)
                # If validation happens once per epoch, we can use step directly
                epoch = int(step)
                if epoch not in metrics:
                    metrics[epoch] = {}
                # Keep the latest value for this epoch
                if 'train_loss' not in metrics[epoch] or scalar.wall_time > metrics[epoch].get('_train_time', 0):
                    metrics[epoch]['train_loss'] = scalar.value
                    metrics[epoch]['_train_time'] = scalar.wall_time
        
        if val_loss_tag:
            val_loss_scalars = ea.Scalars(val_loss_tag)
            for scalar in val_loss_scalars:
                step = scalar.step
                epoch = int(step)
                if epoch not in metrics:
                    metrics[epoch] = {}
                # Keep the latest value for this epoch
                if 'val_loss' not in metrics[epoch] or scalar.wall_time > metrics[epoch].get('_val_time', 0):
                    metrics[epoch]['val_loss'] = scalar.value
                    metrics[epoch]['_val_time'] = scalar.wall_time
        
        # Clean up temporary time fields
        for epoch in metrics:
            metrics[epoch].pop('_train_time', None)
            metrics[epoch].pop('_val_time', None)
                
    except Exception as e:
        print(f"Error reading TensorBoard file {event_file}: {e}")
    
    return metrics


def load_json_metrics(metrics_file: Path) -> List[Dict]:
    '''Load evaluation metrics from JSON file.'''
    if not metrics_file.exists():
        return []
    
    try:
        with open(metrics_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading JSON metrics from {metrics_file}: {e}")
        return []


def merge_tensorboard_metrics(log_metrics: List[Dict], tb_metrics: Dict[int, Dict[str, float]]) -> List[Dict]:
    '''Merge TensorBoard metrics into log metrics.'''
    for metric in log_metrics:
        epoch = metric.get('epoch')
        if epoch is not None and epoch in tb_metrics:
            metric.update(tb_metrics[epoch])
    return log_metrics


def evaluate_stage(
    stage: str,
    log_file: Path,
    tb_dir: Path,
    output_dir: Path
) -> Dict:
    '''Evaluate a single training stage.'''
    print(f"\n{'='*90}")
    print(f"Evaluating stage: {stage}")
    print(f"{'='*90}\n")
    
    # Try to load from JSON first (preferred method)
    stage_dir = tb_dir / stage
    json_metrics = []
    if stage_dir.exists():
        # Find the latest version directory
        version_dirs = sorted([d for d in stage_dir.iterdir() if d.is_dir() and d.name.startswith('version_')])
        if version_dirs:
            latest_version = version_dirs[-1]
            json_file = latest_version / 'evaluation_metrics.json'
            json_metrics = load_json_metrics(json_file)
            if json_metrics:
                print(f"✓ Loaded {len(json_metrics)} evaluation epochs from JSON file: {json_file}")
    
    # Fall back to log parsing if JSON not available
    if not json_metrics:
        print(f"JSON metrics not found, parsing log file: {log_file}")
        log_metrics = parse_log_file(log_file, stage=stage)
        
        if not log_metrics:
            print(f"Warning: No metrics found for stage {stage}!")
            return {'stage': stage, 'metrics': [], 'error': 'No metrics found'}
        
        print(f"Found {len(log_metrics)} evaluation epochs in log file")
        json_metrics = log_metrics
    
    # Extract metrics from TensorBoard (for training/validation loss if not in JSON)
    # Only merge if JSON doesn't already have loss values
    has_loss_in_json = any('train_loss' in m or 'val_loss' in m for m in json_metrics)
    
    if HAS_TENSORBOARD and not has_loss_in_json:
        print(f"Extracting TensorBoard metrics from: {tb_dir / stage}")
        tb_metrics = extract_tensorboard_metrics(tb_dir, stage)
        if tb_metrics:
            print(f"Found TensorBoard metrics for {len(tb_metrics)} epochs")
            json_metrics = merge_tensorboard_metrics(json_metrics, tb_metrics)
        else:
            print("No TensorBoard metrics found")
    elif has_loss_in_json:
        print("Loss metrics already present in JSON, skipping TensorBoard extraction")
    else:
        print("TensorBoard not available, skipping TensorBoard extraction")
    
    # Generate visualizations
    print(f"\nGenerating visualizations...")
    try:
        create_visualizations(json_metrics, output_dir, stage)
    except Exception as e:
        print(f"Error creating visualizations: {e}")
    
    # Print analysis
    print_analysis(json_metrics, stage)
    
    return {
        'stage': stage,
        'metrics': json_metrics,
        'num_epochs': len(json_metrics)
    }


def create_comparative_analysis(all_stage_results: List[Dict], output_dir: Path):
    '''Create comparative visualizations across all stages.'''
    if not HAS_MATPLOTLIB:
        print("Matplotlib not available, skipping comparative analysis")
        return
    
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib not available, skipping comparative analysis")
        return
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Training Loss Comparison
    ax1 = axes[0, 0]
    for result in all_stage_results:
        stage = result['stage']
        metrics = result['metrics']
        epochs = [m['epoch'] for m in metrics if 'train_loss' in m and m.get('train_loss') is not None]
        train_losses = [m['train_loss'] for m in metrics if 'train_loss' in m and m.get('train_loss') is not None]
        if epochs and train_losses:
            ax1.plot(epochs, train_losses, 'o-', label=stage, linewidth=2, markersize=6)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Training Loss', fontsize=12)
    ax1.set_title('Training Loss Comparison', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_yscale('log')
    
    # 2. Validation Loss Comparison
    ax2 = axes[0, 1]
    for result in all_stage_results:
        stage = result['stage']
        metrics = result['metrics']
        epochs = [m['epoch'] for m in metrics if 'val_loss' in m and m.get('val_loss') is not None]
        val_losses = [m['val_loss'] for m in metrics if 'val_loss' in m and m.get('val_loss') is not None]
        if epochs and val_losses:
            ax2.plot(epochs, val_losses, 's-', label=stage, linewidth=2, markersize=6)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Validation Loss', fontsize=12)
    ax2.set_title('Validation Loss Comparison', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_yscale('log')
    
    # 3. Cophenetic Correlation Comparison
    ax3 = axes[1, 0]
    for result in all_stage_results:
        stage = result['stage']
        metrics = result['metrics']
        epochs = [m['epoch'] for m in metrics if 'cophenetic' in m]
        cophenetic = [m['cophenetic'] for m in metrics if 'cophenetic' in m]
        if epochs and cophenetic:
            ax3.plot(epochs, cophenetic, 'o-', label=stage, linewidth=2, markersize=6)
    ax3.axhline(y=0.7, color='g', linestyle='--', alpha=0.5, label='Target (0.7)')
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('Cophenetic Correlation', fontsize=12)
    ax3.set_title('Hierarchy Preservation Comparison', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    ax3.set_ylim([-0.5, 1.0])
    
    # 4. Hyperbolic Radius Comparison
    ax4 = axes[1, 1]
    for result in all_stage_results:
        stage = result['stage']
        metrics = result['metrics']
        epochs = [m['epoch'] for m in metrics]
        radius_means = [m.get('radius_mean', 0) for m in metrics]
        if epochs and radius_means:
            ax4.plot(epochs, radius_means, 'o-', label=stage, linewidth=2, markersize=6)
    ax4.set_xlabel('Epoch', fontsize=12)
    ax4.set_ylabel('Hyperbolic Radius (mean)', fontsize=12)
    ax4.set_title('Hyperbolic Radius Comparison', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    plt.suptitle('Training Stages Comparative Analysis', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    output_file = output_dir / 'comparative_analysis.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved comparative visualization to: {output_file}")
    plt.close()


def generate_report(all_stage_results: List[Dict], output_dir: Path, configs: Dict[str, Dict]):
    '''Generate comprehensive evaluation report.'''
    output_dir.mkdir(parents=True, exist_ok=True)
    report_file = output_dir / 'evaluation_report.md'
    
    with open(report_file, 'w') as f:
        f.write("# Training Stages Evaluation Report\n\n")
        f.write("## Executive Summary\n\n")
        f.write("This report provides a comprehensive evaluation of the first three training stages: ")
        f.write("01_text, 02_text, and 03_text.\n\n")
        
        # Per-stage analysis
        f.write("## Per-Stage Analysis\n\n")
        for result in all_stage_results:
            stage = result['stage']
            metrics = result['metrics']
            
            if not metrics:
                f.write(f"### {stage}\n\n")
                f.write("No metrics available for this stage.\n\n")
                continue
            
            f.write(f"### {stage}\n\n")
            
            # Configuration
            if stage in configs:
                config = configs[stage]
                f.write("**Configuration:**\n")
                f.write(f"- Positive relations: {config.get('positive_relation', 'N/A')}\n")
                f.write(f"- Positive levels: {config.get('positive_level', 'N/A')}\n")
                f.write(f"- N positives: {config.get('n_positives', 'N/A')}\n")
                f.write(f"- N negatives: {config.get('n_negatives', 'N/A')}\n\n")
            
            # Metrics summary
            latest = metrics[-1]
            first = metrics[0]
            
            f.write("**Metrics Summary:**\n")
            f.write(f"- Epochs completed: {len(metrics)}\n")
            f.write(f"- Final epoch: {latest.get('epoch', 'N/A')}\n\n")
            
            # Loss metrics
            if 'train_loss' in latest and latest['train_loss'] is not None:
                f.write("**Loss Metrics:**\n")
                f.write(f"- Training loss: {first.get('train_loss', 'N/A')} → {latest.get('train_loss', 'N/A')}\n")
            if 'val_loss' in latest and latest['val_loss'] is not None:
                f.write(f"- Validation loss: {first.get('val_loss', 'N/A')} → {latest.get('val_loss', 'N/A')}\n")
            f.write("\n")
            
            # Hierarchy preservation
            if 'cophenetic' in latest:
                f.write("**Hierarchy Preservation:**\n")
                f.write(f"- Cophenetic correlation: {first.get('cophenetic', 0):.4f} → {latest.get('cophenetic', 0):.4f}\n")
                f.write("\n")
            
            # Hyperbolic radius
            f.write("**Hyperbolic Radius:**\n")
            f.write(f"- Mean: {first.get('radius_mean', 0):.4f} → {latest.get('radius_mean', 0):.4f}\n")
            f.write(f"- Std: {first.get('radius_std', 0):.4f} → {latest.get('radius_std', 0):.4f}\n")
            f.write("\n")
            
            # Collapse detection
            collapse_count = sum(1 for m in metrics if m.get('collapse', False))
            f.write(f"**Collapse Detection:** {collapse_count} epochs with collapse detected\n\n")
        
        # Comparative analysis
        f.write("## Comparative Analysis\n\n")
        f.write("### Loss Progression\n\n")
        f.write("Comparing training and validation loss across stages:\n\n")
        
        for result in all_stage_results:
            stage = result['stage']
            metrics = result['metrics']
            if metrics:
                latest = metrics[-1]
                first = metrics[0]
                f.write(f"- **{stage}**: ")
                if 'train_loss' in latest and latest['train_loss'] is not None:
                    train_change = latest['train_loss'] - first.get('train_loss', latest['train_loss'])
                    f.write(f"Train loss change: {train_change:+.6f}, ")
                if 'val_loss' in latest and latest['val_loss'] is not None:
                    val_change = latest['val_loss'] - first.get('val_loss', latest['val_loss'])
                    f.write(f"Val loss change: {val_change:+.6f}")
                f.write("\n")
        
        f.write("\n### Hierarchy Preservation\n\n")
        for result in all_stage_results:
            stage = result['stage']
            metrics = result['metrics']
            if metrics and 'cophenetic' in metrics[-1]:
                latest_coph = metrics[-1]['cophenetic']
                first_coph = metrics[0].get('cophenetic', latest_coph)
                f.write(f"- **{stage}**: {first_coph:.4f} → {latest_coph:.4f} "
                       f"({latest_coph - first_coph:+.4f})\n")
        
        f.write("\n## Key Findings\n\n")
        f.write("1. Training progression across stages\n")
        f.write("2. Loss trends and convergence\n")
        f.write("3. Hierarchy preservation improvements\n")
        f.write("4. Hyperbolic radius evolution\n\n")
        
        f.write("## Recommendations\n\n")
        f.write("Based on the analysis:\n")
        f.write("- Continue monitoring loss trends\n")
        f.write("- Track hierarchy preservation metrics\n")
        f.write("- Consider adjustments based on validation performance\n\n")
    
    print(f"\n✓ Saved evaluation report to: {report_file}")


def main():
    '''Main entry point.'''
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate training stages')
    parser.add_argument('--stages', nargs='+', default=['01_text', '02_text', '03_text'],
                       help='Stages to evaluate')
    parser.add_argument('--log-file', type=Path,
                       default=project_root / 'logs' / 'logs' / 'train.log',
                       help='Path to log file')
    parser.add_argument('--tb-dir', type=Path,
                       default=project_root / 'outputs',
                       help='TensorBoard output directory')
    parser.add_argument('--output-dir', type=Path,
                       default=project_root / 'outputs' / 'visualizations',
                       help='Output directory for plots and reports')
    
    args = parser.parse_args()
    
    if not args.log_file.exists():
        print(f"Error: Log file not found: {args.log_file}")
        sys.exit(1)
    
    # Load configs
    configs = {}
    for stage in args.stages:
        config_file = project_root / 'conf' / 'text_curriculum' / f'{stage}.yaml'
        if config_file.exists():
            import yaml
            with open(config_file, 'r') as f:
                configs[stage] = yaml.safe_load(f)
    
    # Evaluate each stage
    all_results = []
    for stage in args.stages:
        result = evaluate_stage(stage, args.log_file, args.tb_dir, args.output_dir)
        all_results.append(result)
    
    # Create comparative analysis
    print(f"\n{'='*90}")
    print("Creating comparative analysis...")
    print(f"{'='*90}\n")
    create_comparative_analysis(all_results, args.output_dir)
    
    # Generate report
    print(f"\n{'='*90}")
    print("Generating evaluation report...")
    print(f"{'='*90}\n")
    generate_report(all_results, args.output_dir, configs)
    
    print(f"\n{'='*90}")
    print("Evaluation complete!")
    print(f"{'='*90}\n")


if __name__ == '__main__':
    main()

