# -------------------------------------------------------------------------------------------------
# Post-Training Monitoring and Visualization (#58)
# -------------------------------------------------------------------------------------------------
'''
Post-training analysis and visualization for curriculum learning.

This module provides tools for analyzing completed training runs:
  - Phase transition visualization
  - Generalization gap tracking
  - Plateau detection events
  - Rollback analysis
  - Per-phase metric summaries

Not a real-time dashboard - designed for post-hoc analysis.
'''

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Optional imports for visualization
try:
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None

# -------------------------------------------------------------------------------------------------
# Training History Analysis
# -------------------------------------------------------------------------------------------------

@dataclass
class PhaseMetrics:
    '''Aggregated metrics for a single curriculum phase.'''

    phase_name: str
    start_epoch: int
    end_epoch: int
    duration_epochs: int

    # Loss metrics
    initial_loss: float
    final_loss: float
    best_loss: float
    loss_improvement: float

    # Gap metrics
    avg_generalization_gap: float
    max_generalization_gap: float

    # Performance metrics
    final_mrr: float
    final_hits_at_10: float

    # Events
    num_plateaus: int
    num_rollbacks: int

class CurriculumAnalyzer:
    '''
    Analyzes completed curriculum training runs.

    Loads training history and provides summary statistics,
    phase breakdowns, and visualization utilities.
    '''

    def __init__(self, history_path: Optional[str] = None) -> None:
        self._history: List[Dict[str, Any]] = []
        self._events: List[Dict[str, Any]] = []
        self._phase_metrics: Dict[str, PhaseMetrics] = {}

        if history_path:
            self.load_history(history_path)

    def load_history(self, path: str) -> None:
        '''Load training history from JSON file.'''
        with open(path, 'r') as f:
            data = json.load(f)

        if isinstance(data, list):
            self._history = data
        elif isinstance(data, dict):
            self._history = data.get('history', [])
            self._events = data.get('events', [])

        logger.info(f'Loaded {len(self._history)} history entries from {path}')
        self._compute_phase_metrics()

    def load_from_controller_history(self, history: List[Dict[str, Any]]) -> None:
        '''Load history directly from controller.'''
        self._history = history
        self._compute_phase_metrics()

    def _compute_phase_metrics(self) -> None:
        '''Compute aggregated metrics for each phase.'''
        if not self._history:
            return

        # Group by phase
        phase_entries: Dict[str, List[Dict[str, Any]]] = {}
        for entry in self._history:
            phase = entry.get('phase', 'PHASE_1_ANCHORING')
            if phase not in phase_entries:
                phase_entries[phase] = []
            phase_entries[phase].append(entry)

        # Compute metrics for each phase
        for phase_name, entries in phase_entries.items():
            if not entries:
                continue

            losses = [e.get('val_loss', e.get('loss', 0)) for e in entries]
            gaps = [e.get('generalization_gap', 0) for e in entries]
            epochs = [e.get('epoch', 0) for e in entries]

            self._phase_metrics[phase_name] = PhaseMetrics(
                phase_name=phase_name,
                start_epoch=min(epochs) if epochs else 0,
                end_epoch=max(epochs) if epochs else 0,
                duration_epochs=len(entries),
                initial_loss=losses[0] if losses else 0.0,
                final_loss=losses[-1] if losses else 0.0,
                best_loss=min(losses) if losses else 0.0,
                loss_improvement=(
                    (losses[0] - losses[-1]) / losses[0] if losses and losses[0] > 0 else 0.0
                ),
                avg_generalization_gap=float(np.mean(gaps)) if gaps else 0.0,
                max_generalization_gap=float(max(gaps)) if gaps else 0.0,
                final_mrr=entries[-1].get('mrr', 0.0) if entries else 0.0,
                final_hits_at_10=entries[-1].get('hits_at_10', 0.0) if entries else 0.0,
                num_plateaus=sum(1 for e in entries if e.get('plateau_detected', False)),
                num_rollbacks=sum(1 for e in entries if e.get('rollback', False)),
            )

    def get_phase_summary(self) -> Dict[str, PhaseMetrics]:
        '''Get summary metrics for each phase.'''
        return self._phase_metrics

    def get_phase_transitions(self) -> List[Dict[str, Any]]:
        '''Get list of phase transition events.'''
        transitions = []
        prev_phase = None

        for entry in self._history:
            phase = entry.get('phase', 'PHASE_1_ANCHORING')
            if phase != prev_phase:
                transitions.append(
                    {
                        'epoch': entry.get('epoch', 0),
                        'from_phase': prev_phase,
                        'to_phase': phase,
                        'loss': entry.get('val_loss', entry.get('loss', 0)),
                        'mrr': entry.get('mrr', 0),
                    }
                )
                prev_phase = phase

        return transitions

    def get_rollback_events(self) -> List[Dict[str, Any]]:
        '''Get list of rollback events.'''
        rollbacks = []

        for i, entry in enumerate(self._history):
            if entry.get('rollback', False):
                rollbacks.append(
                    {
                        'epoch': entry.get('epoch', 0),
                        'phase': entry.get('phase', ''),
                        'generalization_gap': entry.get('generalization_gap', 0),
                        'loss': entry.get('val_loss', entry.get('loss', 0)),
                    }
                )

        return rollbacks

    def get_plateau_events(self) -> List[Dict[str, Any]]:
        '''Get list of plateau detection events.'''
        plateaus = []

        for entry in self._history:
            if entry.get('plateau_detected', False):
                plateaus.append(
                    {
                        'epoch': entry.get('epoch', 0),
                        'phase': entry.get('phase', ''),
                        'loss': entry.get('val_loss', entry.get('loss', 0)),
                        'loss_velocity': entry.get('loss_velocity', 0),
                    }
                )

        return plateaus

    def get_metrics_timeseries(self) -> Dict[str, List[float]]:
        '''Get time series of key metrics.'''
        return {
            'epoch': [e.get('epoch', i) for i, e in enumerate(self._history)],
            'train_loss': [e.get('train_loss', 0) for e in self._history],
            'val_loss': [e.get('val_loss', e.get('loss', 0)) for e in self._history],
            'generalization_gap': [e.get('generalization_gap', 0) for e in self._history],
            'mrr': [e.get('mrr', 0) for e in self._history],
            'loss_velocity': [e.get('loss_velocity', 0) for e in self._history],
            'curvature': [e.get('curvature', 1.0) for e in self._history],
        }

    def print_summary(self) -> None:
        '''Print a text summary of the training run.'''
        print('\n' + '=' * 70)
        print('CURRICULUM TRAINING SUMMARY')
        print('=' * 70)

        if not self._history:
            print('No training history loaded.')
            return

        # Overall stats
        total_epochs = len(self._history)
        initial_loss = self._history[0].get('val_loss', self._history[0].get('loss', 0))
        final_loss = self._history[-1].get('val_loss', self._history[-1].get('loss', 0))

        print(f'\nTotal epochs: {total_epochs}')
        print(f'Initial loss: {initial_loss:.4f}')
        print(f'Final loss: {final_loss:.4f}')
        print(f'Improvement: {(initial_loss - final_loss) / initial_loss * 100:.1f}%')

        # Phase breakdown
        print('\n' + '-' * 70)
        print('PHASE BREAKDOWN')
        print('-' * 70)

        for phase_name, metrics in self._phase_metrics.items():
            print(f'\n{phase_name}:')
            print(
                f'  Epochs: {metrics.start_epoch} - {metrics.end_epoch} '
                f'({metrics.duration_epochs} total)'
            )
            print(
                f'  Loss: {metrics.initial_loss:.4f} -> {metrics.final_loss:.4f} '
                f'({metrics.loss_improvement * 100:.1f}% improvement)'
            )
            print(f'  Avg gen gap: {metrics.avg_generalization_gap:.4f}')
            print(f'  Plateaus: {metrics.num_plateaus}, Rollbacks: {metrics.num_rollbacks}')

        # Events
        transitions = self.get_phase_transitions()
        rollbacks = self.get_rollback_events()

        print('\n' + '-' * 70)
        print('EVENTS')
        print('-' * 70)

        print(f'\nPhase transitions: {len(transitions)}')
        for t in transitions:
            if t['from_phase']:
                print(f"  Epoch {t['epoch']}: {t['from_phase']} -> {t['to_phase']}")

        print(f'\nRollbacks: {len(rollbacks)}')
        for r in rollbacks:
            print(f"  Epoch {r['epoch']}: gap={r['generalization_gap']:.4f}")

        print('\n' + '=' * 70)

# -------------------------------------------------------------------------------------------------
# Visualization
# -------------------------------------------------------------------------------------------------

def plot_curriculum_training(
    analyzer: CurriculumAnalyzer,
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 10),
) -> Optional[Any]:
    '''
    Create visualization of curriculum training run.

    Args:
        analyzer: CurriculumAnalyzer with loaded history
        output_path: Optional path to save figure
        figsize: Figure size

    Returns:
        matplotlib figure if available, None otherwise
    '''
    if not HAS_MATPLOTLIB or plt is None:
        logger.warning('matplotlib not available, skipping visualization')
        return None

    metrics = analyzer.get_metrics_timeseries()
    transitions = analyzer.get_phase_transitions()
    rollbacks = analyzer.get_rollback_events()

    fig, axes = plt.subplots(3, 2, figsize=figsize)  # type: ignore[union-attr]
    fig.suptitle('Curriculum Training Analysis', fontsize=14, fontweight='bold')

    epochs = metrics['epoch']

    # Plot 1: Loss curves
    ax1 = axes[0, 0]
    ax1.plot(epochs, metrics['train_loss'], label='Train Loss', alpha=0.8)
    ax1.plot(epochs, metrics['val_loss'], label='Val Loss', alpha=0.8)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Add phase transition markers
    phase_colors = {
        'PHASE_1_ANCHORING': '#2ecc71',
        'PHASE_2_EXPANSION': '#3498db',
        'PHASE_3_DISCRIMINATION': '#e74c3c',
        'PHASE_4_STABILIZATION': '#9b59b6',
    }

    for t in transitions[1:]:  # Skip first (start)
        color = phase_colors.get(t['to_phase'], 'gray')
        ax1.axvline(x=t['epoch'], color=color, linestyle='--', alpha=0.5)

    # Plot 2: Generalization gap
    ax2 = axes[0, 1]
    ax2.plot(epochs, metrics['generalization_gap'], color='orange', alpha=0.8)
    ax2.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Gap threshold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Gap (val - train)')
    ax2.set_title('Generalization Gap')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Mark rollbacks
    for r in rollbacks:
        ax2.axvline(x=r['epoch'], color='red', linestyle='-', alpha=0.7)
        ax2.annotate(
            'Rollback',
            (r['epoch'], r['generalization_gap']),
            xytext=(5, 10),
            textcoords='offset points',
            fontsize=8,
        )

    # Plot 3: Loss velocity
    ax3 = axes[1, 0]
    ax3.plot(epochs, metrics['loss_velocity'], color='purple', alpha=0.8)
    ax3.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('d(loss)/dt')
    ax3.set_title('Loss Velocity (Plateau Detection)')
    ax3.grid(True, alpha=0.3)

    # Plot 4: Curvature
    ax4 = axes[1, 1]
    ax4.plot(epochs, metrics['curvature'], color='teal', alpha=0.8)
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Curvature (c)')
    ax4.set_title('Hyperbolic Curvature')
    ax4.grid(True, alpha=0.3)

    # Plot 5: MRR (if available)
    ax5 = axes[2, 0]
    if any(m > 0 for m in metrics['mrr']):
        ax5.plot(epochs, metrics['mrr'], color='green', alpha=0.8)
        ax5.set_xlabel('Epoch')
        ax5.set_ylabel('MRR')
        ax5.set_title('Mean Reciprocal Rank')
        ax5.grid(True, alpha=0.3)
    else:
        ax5.text(
            0.5,
            0.5,
            'MRR not available',
            ha='center',
            va='center',
            transform=ax5.transAxes,
            fontsize=12,
            color='gray',
        )
        ax5.set_title('Mean Reciprocal Rank')

    # Plot 6: Phase timeline
    ax6 = axes[2, 1]
    phase_summary = analyzer.get_phase_summary()

    y_pos = 0
    patches = []
    for phase_name, metrics_obj in phase_summary.items():
        color = phase_colors.get(phase_name, 'gray')
        width = metrics_obj.duration_epochs
        rect = mpatches.Rectangle(
            (metrics_obj.start_epoch, y_pos - 0.3),
            width,
            0.6,
            facecolor=color,
            alpha=0.7,
            edgecolor='black',
        )
        ax6.add_patch(rect)
        ax6.text(
            metrics_obj.start_epoch + width / 2,
            y_pos,
            phase_name.replace('PHASE_', 'P').replace('_', '\n'),
            ha='center',
            va='center',
            fontsize=8,
            fontweight='bold',
        )
        patches.append((phase_name, color))
        y_pos += 1

    ax6.set_xlim(0, max(epochs) + 1)
    ax6.set_ylim(-0.5, y_pos)
    ax6.set_xlabel('Epoch')
    ax6.set_title('Phase Timeline')
    ax6.set_yticks([])

    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f'Saved curriculum plot to {output_path}')

    return fig

def generate_report(
    analyzer: CurriculumAnalyzer,
    output_dir: str,
    include_plots: bool = True,
) -> None:
    '''
    Generate a complete analysis report.

    Args:
        analyzer: CurriculumAnalyzer with loaded history
        output_dir: Directory to save report files
        include_plots: Whether to generate plots
    '''
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save summary JSON
    summary = {
        'phase_metrics': {
            name: {
                'phase_name': m.phase_name,
                'start_epoch': m.start_epoch,
                'end_epoch': m.end_epoch,
                'duration_epochs': m.duration_epochs,
                'initial_loss': m.initial_loss,
                'final_loss': m.final_loss,
                'best_loss': m.best_loss,
                'loss_improvement': m.loss_improvement,
                'avg_generalization_gap': m.avg_generalization_gap,
                'max_generalization_gap': m.max_generalization_gap,
                'final_mrr': m.final_mrr,
                'num_plateaus': m.num_plateaus,
                'num_rollbacks': m.num_rollbacks,
            }
            for name, m in analyzer.get_phase_summary().items()
        },
        'transitions': analyzer.get_phase_transitions(),
        'rollbacks': analyzer.get_rollback_events(),
        'plateaus': analyzer.get_plateau_events(),
    }

    with open(output_path / 'curriculum_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # Save metrics CSV
    metrics = analyzer.get_metrics_timeseries()
    csv_lines = ['epoch,train_loss,val_loss,generalization_gap,mrr,loss_velocity,curvature']
    for i in range(len(metrics['epoch'])):
        csv_lines.append(
            f"{metrics['epoch'][i]},"
            f"{metrics['train_loss'][i]:.6f},"
            f"{metrics['val_loss'][i]:.6f},"
            f"{metrics['generalization_gap'][i]:.6f},"
            f"{metrics['mrr'][i]:.6f},"
            f"{metrics['loss_velocity'][i]:.6f},"
            f"{metrics['curvature'][i]:.4f}"
        )

    with open(output_path / 'curriculum_metrics.csv', 'w') as f:
        f.write('\n'.join(csv_lines))

    # Generate plots
    if include_plots and HAS_MATPLOTLIB:
        plot_curriculum_training(
            analyzer,
            output_path=str(output_path / 'curriculum_analysis.png'),
        )

    # Print summary
    analyzer.print_summary()

    logger.info(f'Report generated in {output_dir}')

# -------------------------------------------------------------------------------------------------
# CLI Entry Point
# -------------------------------------------------------------------------------------------------

def main() -> None:
    '''Command-line entry point for analysis.'''
    import argparse

    parser = argparse.ArgumentParser(description='Analyze curriculum training history')
    parser.add_argument(
        'history_file',
        help='Path to training history JSON file',
    )
    parser.add_argument(
        '--output-dir',
        default='./outputs/curriculum_analysis',
        help='Output directory for report',
    )
    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Skip plot generation',
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(message)s')

    analyzer = CurriculumAnalyzer(args.history_file)
    generate_report(
        analyzer,
        output_dir=args.output_dir,
        include_plots=not args.no_plots,
    )

if __name__ == '__main__':
    main()
