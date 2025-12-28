# Quick Start Guide

## 🚀 Getting Started in 5 Minutes

### Step 1: Install Dependencies

```bash
cd cub_kd
pip install -r requirements.txt
```

### Step 2: Verify Dataset

Make sure your CUB-200-2011 dataset is in the project root:
```
Project/
├── cub_kd/           # This framework
└── CUB_200_2011/     # Dataset
```

Or update the data path in configs:
```yaml
data_root: "../CUB_200_2011"  # or your path
```

### Step 3: Run Your First Experiment

**Option A: Standard Training (Baseline)**
```bash
python scripts/train.py --distillation_type none --experiment_name baseline
```

**Option B: Logit Distillation**
```bash
python scripts/train.py --config configs/logit_kd.yaml
```

**Option C: Combined Distillation** (Recommended)
```bash
python scripts/train.py --config configs/combined_kd.yaml
```

### Step 4: Monitor Training

Training output shows:
```
Epoch 1/50
  Train Loss: 3.2154
  Train Acc:  45.23%
  Val Loss:   2.8932
  Val Acc:    52.67%
  Saved best model to outputs/experiment/best_model.pth
```

### Step 5: Evaluate Results

```bash
python scripts/evaluate.py \
    --model_path outputs/your_experiment/best_model.pth \
    --config outputs/your_experiment/config.yaml \
    --measure_speed
```

## 📊 Quick Comparison Experiments

Run all methods on a single architecture:

```bash
# Baseline (no KD)
python scripts/train.py --student_architecture resnet18 --distillation_type none --num_epochs 30

# Logit KD
python scripts/train.py --student_architecture resnet18 --config configs/logit_kd.yaml --num_epochs 30

# Attention KD
python scripts/train.py --student_architecture resnet18 --config configs/attention_kd.yaml --num_epochs 30

# Combined KD
python scripts/train.py --student_architecture resnet18 --config configs/combined_kd.yaml --num_epochs 30
```

## 🎯 Expected Results

Typical accuracy improvements with knowledge distillation:

| Method | ResNet-18 | MobileNetV3-Small |
|--------|-----------|-------------------|
| Baseline (ImageNet pretrained) | ~72% | ~68% |
| + Logit KD | ~75% | ~71% |
| + Attention KD | ~74% | ~70% |
| + Combined KD | ~76% | ~72% |

## ⚙️ Common Configurations

### For Quick Testing (Fast)
```yaml
num_epochs: 10
batch_size: 64
learning_rate: 0.001
```

### For Best Results (Slow)
```yaml
num_epochs: 100
batch_size: 32
learning_rate: 0.0003
```

### For Limited GPU Memory
```yaml
batch_size: 16
use_amp: true  # Mixed precision
```

## 🔧 Troubleshooting

### CUDA Out of Memory
```bash
python scripts/train.py --batch_size 16 --use_amp true
```

### Slow Data Loading (Windows)
```bash
python scripts/train.py --num_workers 0
```

### Dataset Not Found
Update config or use argument:
```bash
python scripts/train.py --data_root "path/to/CUB_200_2011"
```

## 📈 Visualize Results

After running experiments, check:
- `outputs/experiment/history.json` - Training curves
- `outputs/experiment/evaluation_results.json` - Final metrics

## 🎓 Next Steps

1. **Try Different Architectures**: Use `--student_architecture mobilenetv3_small` or `vgg16`
2. **Tune Hyperparameters**: Adjust `alpha_kd`, `alpha_attention`, `temperature`
3. **Run Full Comparison**: Use `scripts/run_experiments.py`
4. **Analyze Attention Maps**: Extract and visualize what the model learns

Happy experimenting! 🚀
