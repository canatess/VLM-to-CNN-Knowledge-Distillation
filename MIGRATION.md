# Migration Guide: Old Structure → New Unified Framework

## 📦 What Changed?

This document explains how the **new unified framework** improves upon and consolidates the two previous separate codebases.

## 🔄 Structural Comparison

### Before: Two Separate Codebases

```
Project/
├── attention_distillation/     # Attention-based KD
│   ├── scripts/
│   │   └── 10_train_attn_kd.py
│   └── src/
│       ├── data/cub.py
│       ├── kd/losses.py
│       ├── models/
│       └── train/trainer.py
│
└── logit_distillation/         # Logit-based KD
    ├── datasets_cub.py
    ├── distill.py
    ├── models.py
    └── run_experiments.py
```

**Problems:**
- ❌ Duplicated code (dataset, models, training loop)
- ❌ Inconsistent APIs and conventions
- ❌ Cannot easily combine both KD methods
- ❌ Hard to maintain and extend
- ❌ Difficult to compare results fairly

### After: Unified Framework

```
cub_kd/
├── configs/                    # Centralized configuration
├── src/
│   ├── data/                  # Single dataset implementation
│   ├── models/                # Unified teacher & student models
│   ├── distillation/          # ALL KD methods in one place
│   ├── training/              # Single training & eval system
│   └── utils/                 # Shared utilities
└── scripts/                   # Clean entry points
```

**Benefits:**
- ✅ No code duplication
- ✅ Consistent, clean API
- ✅ Easy to switch between KD methods
- ✅ Combine multiple methods (logit + attention)
- ✅ Easy to extend with new methods
- ✅ Fair comparison with identical setup

## 🎯 Feature Comparison

| Feature | Old (Separate) | New (Unified) |
|---------|----------------|---------------|
| **Logit KD** | ✓ (in logit_distillation/) | ✅ Integrated |
| **Attention KD** | ✓ (in attention_distillation/) | ✅ Integrated |
| **Combined KD** | ✗ Not possible | ✅ **New!** |
| **Multiple architectures** | Partially | ✅ Full support |
| **Configuration system** | Hardcoded/partial | ✅ YAML configs |
| **Experiment management** | Manual | ✅ Automated |
| **Code reuse** | ~40% duplication | ✅ 100% shared |
| **Extensibility** | Difficult | ✅ Easy |

## 🔄 Code Migration Examples

### Example 1: Dataset Loading

**Old (attention_distillation):**
```python
from src.data.cub import make_loaders
train_loader, test_loader = make_loaders(
    root="CUB_200_2011",
    image_size=224,
    batch_size=32,
    num_workers=0
)
```

**Old (logit_distillation):**
```python
from datasets_cub import CUB200, build_transforms
train_ds = CUB200(root="CUB_200_2011", train=True, 
                  transform=build_transforms(224, train=True))
```

**New (Unified):**
```python
from src.data import build_dataloaders
train_loader, val_loader, test_loader = build_dataloaders(
    root="CUB_200_2011",
    image_size=224,
    batch_size=32,
    num_workers=4,
    val_ratio=0.1  # NEW: automatic validation split
)
```

### Example 2: Model Creation

**Old (attention_distillation):**
```python
from src.models.teacher_clip import CLIPTeacher
from src.models.student_timm import Student

teacher = CLIPTeacher(model_id="openai/clip-vit-base-patch32", 
                     class_names=class_names)
student = Student(arch="resnet18", num_classes=200)
```

**Old (logit_distillation):**
```python
from models import ClipTeacher, build_student

teacher = ClipTeacher(model_name="openai/clip-vit-base-patch32",
                     class_names=class_names)
student = build_student("resnet18", num_classes=200)
```

**New (Unified):**
```python
from src.models import CLIPTeacher, StudentCNN

teacher = CLIPTeacher(model_name="openai/clip-vit-base-patch32",
                     class_names=class_names,
                     extract_attention=True)  # NEW: optional attention
student = StudentCNN(architecture="resnet18",
                    num_classes=200,
                    extract_features=True)  # NEW: for attention KD
```

### Example 3: Training

**Old (attention_distillation):**
```python
from src.train.trainer import train_one_epoch

for epoch in range(num_epochs):
    metrics = train_one_epoch(
        student, teacher, train_loader, optimizer, device,
        alpha_kd=1.0, T=4.0, alpha_attn=0.1
    )
    # Manual evaluation and saving...
```

**Old (logit_distillation):**
```python
# Custom training loop with manual loss computation
for images, labels in train_loader:
    optimizer.zero_grad()
    student_logits = student(images)
    teacher_logits = teacher.forward(images, device)
    loss = kd_kl_loss(student_logits, teacher_logits, T=4.0)
    # ... more manual code
```

**New (Unified):**
```python
from src.distillation import KnowledgeDistiller
from src.training import Trainer

# High-level distillation interface
distiller = KnowledgeDistiller(
    teacher=teacher,
    student=student,
    distillation_type="combined",  # logit, attention, or combined
    alpha_kd=1.0,
    alpha_attention=0.1,
    temperature=4.0
)

# Automatic training with all features
trainer = Trainer(
    model=student,
    train_loader=train_loader,
    val_loader=val_loader,
    test_loader=test_loader,
    optimizer=optimizer,
    distiller=distiller,
    device=device
)

history = trainer.train(num_epochs=50)
```

### Example 4: Running Experiments

**Old:** Manual script execution for each configuration
```bash
# Manually edit scripts for different configs
python attention_distillation/scripts/10_train_attn_kd.py
python logit_distillation/run_experiments.py
# Results in different formats, hard to compare
```

**New:** Configuration-driven experiments
```bash
# Single command with different configs
python scripts/train.py --config configs/logit_kd.yaml
python scripts/train.py --config configs/attention_kd.yaml
python scripts/train.py --config configs/combined_kd.yaml

# Or run comprehensive comparison
python scripts/run_experiments.py \
    --architectures resnet18 mobilenetv3_small \
    --distillation_types none logit attention combined
```

## 🆕 New Features Not in Old Codebases

1. **Combined Distillation**: Use both logit and attention transfer simultaneously
2. **Configuration System**: YAML-based configs for reproducibility
3. **Validation Split**: Automatic stratified train/val split
4. **Mixed Precision**: Optional AMP for faster training
5. **Comprehensive Evaluation**: Top-5 accuracy, inference time, model size
6. **Experiment Management**: Automated result collection and comparison
7. **Better Logging**: Progress bars, structured output, history tracking
8. **Extensibility**: Easy to add new KD methods, architectures, or losses

## 🎓 How to Replicate Old Experiments

### Replicate: attention_distillation experiments

**Old:**
```bash
cd attention_distillation
python scripts/10_train_attn_kd.py
```

**New:**
```bash
cd cub_kd
python scripts/train.py \
    --config configs/attention_kd.yaml \
    --student_architecture resnet18
```

### Replicate: logit_distillation experiments

**Old:**
```bash
cd logit_distillation
python run_experiments.py
```

**New:**
```bash
cd cub_kd
python scripts/train.py \
    --config configs/logit_kd.yaml \
    --student_architecture resnet18
```

### NEW: Combined approach (best of both worlds)

```bash
python scripts/train.py \
    --config configs/combined_kd.yaml \
    --student_architecture resnet18
```

## 📊 Advantages Summary

### Code Quality
- **DRY Principle**: No duplicated code
- **Modular Design**: Clear separation of concerns
- **Type Hints**: Better IDE support and documentation
- **Docstrings**: Comprehensive documentation

### Maintainability
- **Single Source of Truth**: One implementation per component
- **Easy Updates**: Change once, apply everywhere
- **Testing**: Easier to write and maintain tests

### Research Productivity
- **Fast Iteration**: Change configs, not code
- **Fair Comparison**: Identical setup for all methods
- **Reproducibility**: Save full configuration with results
- **Extensibility**: Add new methods with minimal code

### Performance
- **Optimized**: Best practices from both codebases
- **Efficient**: Optional mixed precision, data loading optimization
- **Scalable**: Easy to parallelize across GPUs

## 🚀 Next Steps

1. **Install** the new framework: `pip install -r requirements.txt`
2. **Run** baseline experiments to verify setup
3. **Compare** with your old results
4. **Extend** with your own improvements
5. **Contribute** back improvements to the codebase

## 💡 Tips for Transition

- Start with the Quick Start guide
- Use provided configs as templates
- Compare outputs with old codebase initially
- Gradually adopt new features (combined KD, validation split, etc.)
- Refer to this guide when migrating custom code

## 📞 Support

If you have questions about migrating from the old structure:
1. Check the README.md for API documentation
2. Review code examples in scripts/
3. Open an issue with "[Migration]" prefix

Welcome to the unified framework! 🎉
