# CUB-200-2011 Knowledge Distillation Framework

A unified framework for **VLM (Vision-Language Model) to CNN knowledge distillation** for fine-grained bird classification using the CUB-200-2011 dataset.

This framework combines multiple knowledge distillation techniques in a single, well-structured codebase:
- **Logit-based distillation**: Transfer soft probability distributions from teacher to student
- **Attention-based distillation**: Transfer visual attention maps from CLIP to CNN
- **Combined distillation**: Utilize both logit and attention transfer simultaneously

## 🎯 Features

- **Unified Architecture**: Single codebase supporting multiple KD techniques
- **Flexible Configuration**: YAML-based configuration system for easy experimentation
- **Multiple Student Architectures**: ResNet, VGG, MobileNet, EfficientNet, and more
- **CLIP Teacher**: Uses pre-trained CLIP vision-language model as teacher
- **Comprehensive Evaluation**: Top-1, Top-5 accuracy, inference speed measurements
- **Experiment Management**: Run and compare multiple experiments systematically

## 📁 Project Structure

```
VLM-to-CNN-Knowledge-Distillation/
├── configs/                    # Configuration files
│   ├── base.yaml              # Base configuration
│   ├── logit_kd.yaml          # Logit distillation config
│   ├── attention_kd.yaml      # Attention distillation config
│   └── combined_kd.yaml       # Combined distillation config
├── src/                       # Source code
│   ├── data/                  # Dataset and data loading
│   │   ├── dataset.py         # CUB-200-2011 dataset implementation
│   │   └── transforms.py      # Image transformations
│   ├── models/                # Model implementations
│   │   ├── teacher.py         # CLIP teacher model
│   │   └── student.py         # CNN student models
│   ├── distillation/          # Knowledge distillation
│   │   ├── losses.py          # Distillation loss functions
│   │   ├── attention.py       # Attention processing utilities
│   │   └── distiller.py       # High-level KD orchestrator
│   ├── training/              # Training and evaluation
│   │   ├── trainer.py         # Training loop
│   │   └── evaluator.py       # Evaluation utilities
│   └── utils/                 # Utilities
│       ├── config.py          # Configuration management
│       ├── helpers.py         # Helper functions
│       └── metrics.py         # Metrics tracking
├── scripts/                   # Execution scripts
│   ├── train.py              # Main training script
│   ├── evaluate.py           # Evaluation script
│   └── run_experiments.py    # Run multiple experiments
└── requirements.txt          # Python dependencies
```

## 🚀 Installation

### 1. Clone the repository

```bash
cd VLM-to-CNN-Knowledge-Distillation
```

### 2. Create virtual environment (recommended)

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download CUB-200-2011 dataset

The dataset should be placed in the project root with the following structure:
```
CUB_200_2011/
├── images/
├── images.txt
├── image_class_labels.txt
├── train_test_split.txt
└── classes.txt
```

## 💻 Usage

### Basic Training

Train a student model with default configuration:

```bash
python scripts/train.py
```

### Train with Specific Configuration

Use predefined configurations for different KD methods:

```bash
# Logit-based distillation
python scripts/train.py --config configs/logit_kd.yaml

# Attention-based distillation
python scripts/train.py --config configs/attention_kd.yaml

# Combined distillation (logit + attention)
python scripts/train.py --config configs/combined_kd.yaml
```

### Custom Training

Override specific parameters:

```bash
python scripts/train.py \
    --config configs/base.yaml \
    --student_architecture mobilenetv3_small \
    --num_epochs 30 \
    --batch_size 64 \
    --learning_rate 0.001
```

### Train from Scratch (No Distillation)

```bash
python scripts/train.py \
    --pretrained false \
    --distillation_type none \
    --num_epochs 100
```

### Evaluate Trained Model

```bash
python scripts/evaluate.py \
    --model_path outputs/experiment/best_model.pth \
    --config outputs/experiment/config.yaml \
    --save_predictions \
    --measure_speed
```

### Run Comprehensive Experiments

Compare multiple architectures and distillation methods:

```bash
python scripts/run_experiments.py \
    --architectures resnet18 mobilenetv3_small vgg16 \
    --distillation_types none logit attention combined \
    --num_epochs 30 \
    --output_dir ./experiments
```

## 📊 Supported Architectures

### Student Models (CNN)
- ResNet: `resnet18`, `resnet34`, `resnet50`
- VGG: `vgg16`, `vgg19`
- MobileNet: `mobilenetv3_small`, `mobilenetv3_large`
- EfficientNet: `efficientnet_b0`, `efficientnet_b1`
- DenseNet: `densenet121`

### Teacher Model (VLM)
- CLIP: `openai/clip-vit-base-patch32` (default)

## ⚙️ Configuration

Key configuration parameters in `configs/base.yaml`:

```yaml
# Data
data_root: "CUB_200_2011"
batch_size: 32
image_size: 224

# Models
student_architecture: "resnet18"
teacher_model: "openai/clip-vit-base-patch32"

# Training
num_epochs: 50
learning_rate: 0.0003
optimizer: "adamw"

# Distillation
distillation_type: "combined"  # none, logit, attention, combined
alpha_ce: 1.0          # Weight for cross-entropy loss
alpha_kd: 1.0          # Weight for logit distillation
alpha_attention: 0.1   # Weight for attention distillation
temperature: 4.0       # Temperature for softmax
```

## 📈 Distillation Methods

### 1. Logit-Based Distillation
Transfers soft probability distributions from teacher to student using KL divergence.

```yaml
distillation_type: "logit"
alpha_kd: 1.0
temperature: 4.0
```

### 2. Attention-Based Distillation
Transfers visual attention maps from CLIP vision encoder to CNN feature maps.

```yaml
distillation_type: "attention"
alpha_attention: 0.5
attention_loss_type: "mse"  # mse, l1, or kl
```

### 3. Combined Distillation
Uses both logit and attention transfer simultaneously for maximum knowledge transfer.

```yaml
distillation_type: "combined"
alpha_ce: 1.0
alpha_kd: 1.0
alpha_attention: 0.1
```

## 📝 Output Structure

Each experiment creates a directory with:

```
outputs/experiment_name/
├── config.yaml              # Saved configuration
├── best_model.pth          # Best model checkpoint
├── history.json            # Training history
├── evaluation_results.json # Evaluation metrics
└── predictions.npz         # Model predictions (optional)
```

## 🔬 Example Workflow

```bash
# 1. Train a ResNet-18 with combined distillation
python scripts/train.py \
    --config configs/combined_kd.yaml \
    --student_architecture resnet18 \
    --experiment_name resnet18_combined

# 2. Evaluate the trained model
python scripts/evaluate.py \
    --model_path outputs/resnet18_combined/best_model.pth \
    --config outputs/resnet18_combined/config.yaml \
    --measure_speed

# 3. Compare with baseline (no distillation)
python scripts/train.py \
    --student_architecture resnet18 \
    --distillation_type none \
    --experiment_name resnet18_baseline
```

## 🎓 Citation

If you use this framework in your research, please cite:

```bibtex
@misc{VLM-to-CNN-Knowledge-Distillation,
  title={Unified Knowledge Distillation Framework for Fine-Grained Bird Classification},
  author={Can Ali Ateş, Abdullah Enes Ergün, Emre Çoban},
  year={2025}
}
```

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- CUB-200-2011 dataset: [Caltech-UCSD Birds-200-2011](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html)
- CLIP model: [OpenAI CLIP](https://github.com/openai/CLIP)
- timm library: [PyTorch Image Models](https://github.com/huggingface/pytorch-image-models)

## 📧 Contact

For questions or issues, please open an issue on GitHub or contact [your-email@example.com].
