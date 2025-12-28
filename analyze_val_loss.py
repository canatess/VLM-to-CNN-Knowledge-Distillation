#!/usr/bin/env python3
"""
Analyze validation loss curves for ResNet18 models.
"""

import json
from pathlib import Path
import numpy as np

def load_val_loss(file_path):
    """Load validation loss from history.json."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data.get('val_loss', [])

def analyze_val_loss():
    """Analyze validation loss for different methods."""
    outputs_dir = Path('/home/emre/cmp722_distillation/outputs')

    methods = {
        'Transfer': 'resnet18_transfer',
        'Logit KD': 'resnet18_logit_kd',
        'Attention KD': 'resnet18_attention_kd',
        'Combined KD': 'resnet18_combined_kd'
    }

    analysis = {}

    for method_name, dir_name in methods.items():
        history_file = outputs_dir / dir_name / 'history.json'
        if history_file.exists():
            val_loss = load_val_loss(history_file)
            if val_loss:
                analysis[method_name] = {
                    'initial_loss': val_loss[0],
                    'final_loss': val_loss[-1],
                    'min_loss': min(val_loss),
                    'mean_loss': np.mean(val_loss),
                    'std_loss': np.std(val_loss),
                    'epochs': len(val_loss),
                    'convergence_epoch': next((i for i, loss in enumerate(val_loss) if loss == min(val_loss)), len(val_loss))
                }

    return analysis

def print_analysis(analysis):
    """Print detailed analysis."""
    print("=== Validation Loss Analysis for ResNet18 Models ===\n")

    for method, stats in analysis.items():
        print(f"**{method}:**")
        print(".4f")
        print(".4f")
        print(".4f")
        print(".4f")
        print(".4f")
        print(f"  - Convergence: Achieved minimum loss at epoch {stats['convergence_epoch'] + 1}")
        print(f"  - Stability: Standard deviation of {stats['std_loss']:.4f}")
        print()

    # Comparative analysis
    print("=== Comparative Analysis ===")
    final_losses = {method: stats['final_loss'] for method, stats in analysis.items()}
    best_method = min(final_losses, key=final_losses.get)
    worst_method = max(final_losses, key=final_losses.get)

    print(f"Best performing method: {best_method} (final loss: {final_losses[best_method]:.4f})")
    print(f"Worst performing method: {worst_method} (final loss: {final_losses[worst_method]:.4f})")

    # Improvement over transfer learning
    transfer_loss = final_losses['Transfer']
    for method, loss in final_losses.items():
        if method != 'Transfer':
            improvement = ((transfer_loss - loss) / transfer_loss) * 100
            print(".1f")

    # Convergence analysis
    print("\n=== Convergence Analysis ===")
    convergence_epochs = {method: stats['convergence_epoch'] + 1 for method, stats in analysis.items()}
    fastest_convergence = min(convergence_epochs, key=convergence_epochs.get)
    slowest_convergence = max(convergence_epochs, key=convergence_epochs.get)

    print(f"Fastest convergence: {fastest_convergence} (epoch {convergence_epochs[fastest_convergence]})")
    print(f"Slowest convergence: {slowest_convergence} (epoch {convergence_epochs[slowest_convergence]})")

    # Stability analysis
    print("\n=== Stability Analysis ===")
    stabilities = {method: stats['std_loss'] for method, stats in analysis.items()}
    most_stable = min(stabilities, key=stabilities.get)
    least_stable = max(stabilities, key=stabilities.get)

    print(f"Most stable training: {most_stable} (std: {stabilities[most_stable]:.4f})")
    print(f"Least stable training: {least_stable} (std: {stabilities[least_stable]:.4f})")

if __name__ == "__main__":
    analysis = analyze_val_loss()
    print_analysis(analysis)