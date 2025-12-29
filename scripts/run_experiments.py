import argparse
import sys
from pathlib import Path
import json
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import Config, ensure_dir


def parse_args():
    parser = argparse.ArgumentParser(description="Run comprehensive KD experiments")
    
    parser.add_argument("--architectures", nargs="+", 
                        default=["resnet18", "mobilenetv3_small", "vgg16"],
                        help="Student architectures to test")
    parser.add_argument("--distillation_types", nargs="+",
                        default=["none", "logit", "attention", "combined"],
                        help="Distillation methods to test")
    parser.add_argument("--num_epochs", type=int, default=30,
                        help="Number of epochs per experiment")
    parser.add_argument("--output_dir", type=str, default="./experiments",
                        help="Base output directory")
    parser.add_argument("--data_root", type=str, default="CUB_200_2011",
                        help="Path to CUB dataset")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use")
    
    return parser.parse_args()


def run_single_experiment(arch: str, distill_type: str, config: Config, output_dir: Path):
    import subprocess
    
    # Update config
    config.student_architecture = arch
    config.distillation_type = distill_type
    config.experiment_name = f"{arch}_{distill_type}"
    
    # Create experiment directory
    exp_dir = output_dir / config.experiment_name
    ensure_dir(exp_dir)
    
    # Save config
    config_path = exp_dir / "config.yaml"
    config.save(str(config_path))
    
    print(f"\n{'='*80}")
    print(f"Running: {config.experiment_name}")
    print(f"{'='*80}")
    
    # Run training script
    cmd = [
        sys.executable,
        "scripts/train.py",
        "--config", str(config_path),
        "--output_dir", str(output_dir)
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running experiment: {e}")
        return False


def collect_results(output_dir: Path, architectures: list, distill_types: list):
    results = []
    
    for arch in architectures:
        for distill_type in distill_types:
            exp_name = f"{arch}_{distill_type}"
            exp_dir = output_dir / exp_name
            
            # Try to load evaluation results
            eval_path = exp_dir / "evaluation_results.json"
            history_path = exp_dir / "history.json"
            
            if eval_path.exists():
                with open(eval_path, "r") as f:
                    eval_data = json.load(f)
                
                results.append({
                    "architecture": arch,
                    "distillation": distill_type,
                    "test_accuracy": eval_data.get("accuracy", 0),
                    "top5_accuracy": eval_data.get("top5_accuracy", 0),
                    "parameters": eval_data.get("parameters", 0),
                    "size_mb": eval_data.get("size_mb", 0),
                })
            elif history_path.exists():
                with open(history_path, "r") as f:
                    history = json.load(f)
                
                # Get best validation accuracy or last test accuracy
                test_acc = history.get("test_acc", [0])[-1] if history.get("test_acc") else 0
                val_acc = max(history.get("val_acc", [0])) if history.get("val_acc") else 0
                
                results.append({
                    "architecture": arch,
                    "distillation": distill_type,
                    "test_accuracy": test_acc,
                    "val_accuracy": val_acc,
                    "parameters": 0,
                    "size_mb": 0,
                })
    
    return results


def main():
    args = parse_args()
    
    print("\n" + "="*80)
    print("Comprehensive Knowledge Distillation Experiments")
    print("="*80)
    print(f"\nArchitectures: {', '.join(args.architectures)}")
    print(f"Distillation methods: {', '.join(args.distillation_types)}")
    print(f"Epochs per experiment: {args.num_epochs}")
    print(f"Output directory: {args.output_dir}\n")
    
    # Create output directory
    output_dir = ensure_dir(args.output_dir)
    
    # Base configuration
    base_config = Config()
    base_config.num_epochs = args.num_epochs
    base_config.data_root = args.data_root
    base_config.device = args.device
    base_config.output_dir = str(output_dir)
    
    # Run all experiments
    total_experiments = len(args.architectures) * len(args.distillation_types)
    current_exp = 0
    
    for arch in args.architectures:
        for distill_type in args.distillation_types:
            current_exp += 1
            print(f"\nExperiment {current_exp}/{total_experiments}")
            
            success = run_single_experiment(arch, distill_type, base_config, output_dir)
            
            if not success:
                print(f"Failed: {arch}_{distill_type}")
    
    print("\n" + "="*80)
    print("Collecting results...")
    print("="*80)
    
    # Collect and save results
    results = collect_results(output_dir, args.architectures, args.distillation_types)
    
    if results:
        # Create DataFrame
        df = pd.DataFrame(results)
        
        # Save to CSV
        csv_path = output_dir / "results_summary.csv"
        df.to_csv(csv_path, index=False)
        print(f"\nResults saved to: {csv_path}")
        
        # Print summary table
        print("\nResults Summary:")
        print(df.to_string(index=False))
        
        # Print best results
        if "test_accuracy" in df.columns:
            best_result = df.loc[df["test_accuracy"].idxmax()]
            print(f"\nBest Result:")
            print(f"  Architecture: {best_result['architecture']}")
            print(f"  Distillation: {best_result['distillation']}")
            print(f"  Test Accuracy: {best_result['test_accuracy']:.2f}%")
    
    print("\n" + "="*80)
    print("All experiments completed!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
