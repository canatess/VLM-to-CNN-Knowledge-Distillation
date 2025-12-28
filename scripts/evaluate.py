"""
Evaluation script for trained models.

Usage:
    # Evaluate a trained model
    python scripts/evaluate.py --model_path outputs/experiment/best_model.pth --config outputs/experiment/config.yaml

    # Evaluate and save predictions
    python scripts/evaluate.py --model_path best_model.pth --config config.yaml --save_predictions
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import json

from src.utils import Config, set_seed, get_device
from src.data import build_dataloaders
from src.models import StudentCNN
from src.training import evaluate_model, compute_top_k_accuracy, measure_inference_time, count_parameters, model_size_mb


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate trained model")
    
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to trained model checkpoint")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to configuration file")
    parser.add_argument("--data_root", type=str, default=None,
                        help="Override data root path")
    parser.add_argument("--save_predictions", action="store_true",
                        help="Save predictions to file")
    parser.add_argument("--measure_speed", action="store_true",
                        help="Measure inference speed")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (cuda or cpu)")
    
    return parser.parse_args()


def main():
    """Main evaluation function."""
    args = parse_args()
    
    # Load configuration
    config = Config.load(args.config)
    if args.data_root:
        config.data_root = args.data_root
    
    # Set seed for reproducibility
    set_seed(config.seed)
    
    # Get device
    device = get_device(args.device or config.device)
    
    print("\n" + "="*80)
    print("Model Evaluation")
    print("="*80)
    print(f"\nModel: {args.model_path}")
    print(f"Architecture: {config.student_architecture}")
    print(f"Device: {device}\n")
    
    # Load test data
    print("Loading test dataset...")
    _, test_loader = build_dataloaders(
        root=config.data_root,
        image_size=config.image_size,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        val_ratio=0.0
    )
    print(f"  Test samples: {len(test_loader.dataset)}\n")
    
    # Build model
    print("Loading model...")
    model = StudentCNN(
        architecture=config.student_architecture,
        num_classes=config.num_classes,
        pretrained=False,  # We're loading trained weights
        extract_features=False
    )
    
    # Load checkpoint
    checkpoint = torch.load(args.model_path, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    # Model info
    param_count = count_parameters(model)
    size_mb = model_size_mb(model)
    print(f"  Parameters: {param_count['total']:,}")
    print(f"  Model size: {size_mb:.2f} MB\n")
    
    # Evaluate
    print("Evaluating on test set...")
    results = evaluate_model(
        model,
        test_loader,
        device,
        return_predictions=args.save_predictions,
        compute_all_metrics=True
    )
    
    print(f"\nTest Results:")
    print(f"  Accuracy:      {results['accuracy']:.2f}%")
    print(f"  Top-5 Acc:     {results['top5_accuracy']:.2f}%")
    print(f"  F1 (Macro):    {results['f1_macro']:.2f}%")
    print(f"  F1 (Weighted): {results['f1_weighted']:.2f}%")
    print(f"  F1 (Micro):    {results['f1_micro']:.2f}%")
    print(f"  Loss:          {results['loss']:.4f}")
    print(f"  Correct:       {results['correct']}/{results['total']}")
    
    # Save predictions if requested
    if args.save_predictions:
        output_path = Path(args.model_path).parent / "predictions.npz"
        np.savez(
            output_path,
            predictions=results['predictions'],
            targets=results['targets']
        )
        print(f"\nPredictions saved to: {output_path}")
    
    # Measure inference speed if requested
    if args.measure_speed:
        print("\nMeasuring inference speed...")
        timing = measure_inference_time(
            model,
            input_size=(1, 3, config.image_size, config.image_size),
            device=device,
            warmup_runs=10,
            num_runs=100
        )
        print(f"  Mean: {timing['mean_ms']:.2f} ms")
        print(f"  Std:  {timing['std_ms']:.2f} ms")
        print(f"  Min:  {timing['min_ms']:.2f} ms")
        print(f"  Max:  {timing['max_ms']:.2f} ms")
    
    # Save summary
    summary = {
        "model_path": str(args.model_path),
        "architecture": config.student_architecture,
        "parameters": param_count['total'],
        "size_mb": size_mb,
        "accuracy": float(results['accuracy']),
        "top5_accuracy": float(results['top5_accuracy']),
        "f1_macro": float(results['f1_macro']),
        "f1_weighted": float(results['f1_weighted']),
        "f1_micro": float(results['f1_micro']),
        "loss": float(results['loss']),
    }
    
    if args.measure_speed:
        summary["inference_time_ms"] = {
            "mean": float(timing['mean_ms']),
            "std": float(timing['std_ms'])
        }
    
    output_path = Path(args.model_path).parent / "evaluation_results.json"
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nEvaluation summary saved to: {output_path}")
    print("\nEvaluation completed!\n")


if __name__ == "__main__":
    main()
