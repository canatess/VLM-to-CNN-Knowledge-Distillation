#!/bin/bash
#SBATCH --job-name=all-experiments
#SBATCH --account=hpcusers
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=7-00:00:00
#SBATCH --chdir=/home/emre/cmp722_distillation
#SBATCH --output=slurm-%x-%A.out
#SBATCH --error=slurm-%x-%A.err
#SBATCH --gpus=1

# 🔧 IMPORTANT: disable Xet backend, avoid CAS permission issues
export HF_HUB_ENABLE_XET=0

echo "=== Loading Conda ==="
module load conda/latest
eval "$(conda shell.bash hook)"

ENV_NAME="cmp722_distillation_env"

# Create env if it does not exist
if ! conda env list | grep -q "$ENV_NAME"; then
  echo "=== Creating Conda env: $ENV_NAME ==="
  conda create -y -n "$ENV_NAME" python=3.10
fi

echo "=== Activating env: $ENV_NAME ==="
conda activate "$ENV_NAME"

echo "=== Installing Python dependencies ==="
pip install --upgrade pip
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

echo "=== Running All Experiments ==="

# Run all architecture experiments
./experiment_scripts/run_resnet18_experiments.sh
./experiment_scripts/run_resnet34_experiments.sh
./experiment_scripts/run_resnet50_experiments.sh
./experiment_scripts/run_vgg16_experiments.sh
./experiment_scripts/run_vgg19_experiments.sh
./experiment_scripts/run_mobilenetv3_small_experiments.sh
./experiment_scripts/run_mobilenetv3_large_experiments.sh
./experiment_scripts/run_efficientnet_b0_experiments.sh
./experiment_scripts/run_efficientnet_b1_experiments.sh
./experiment_scripts/run_densenet_experiments.sh

echo "All experiments completed!"