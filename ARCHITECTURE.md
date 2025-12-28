# Architecture Overview

## 🏗️ System Architecture

This document provides a detailed overview of the unified knowledge distillation framework architecture.

## 📐 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     User Interface                          │
│  (Config Files, Command Line, Python API)                   │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│                  Main Scripts Layer                         │
│  • train.py         • evaluate.py    • run_experiments.py  │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│                  Core Framework                             │
│  ┌──────────────┬──────────────┬──────────────┐           │
│  │   Data       │   Models     │ Distillation │           │
│  │              │              │              │           │
│  │  • Dataset   │  • Teacher   │  • Losses    │           │
│  │  • Loaders   │  • Student   │  • Attention │           │
│  │  • Transform │  • Factory   │  • Distiller │           │
│  └──────┬───────┴──────┬───────┴──────┬───────┘           │
│         │              │              │                     │
│  ┌──────▼──────────────▼──────────────▼───────┐           │
│  │           Training & Evaluation             │           │
│  │  • Trainer  • Evaluator  • Metrics          │           │
│  └─────────────────────┬────────────────────────┘           │
│                        │                                     │
│  ┌─────────────────────▼────────────────────────┐           │
│  │              Utilities                       │           │
│  │  • Config  • Helpers  • Metrics  • Logging  │           │
│  └──────────────────────────────────────────────┘           │
└─────────────────────────────────────────────────────────────┘
```

## 🔍 Component Details

### 1. Data Module (`src/data/`)

**Purpose**: Handle all data loading and preprocessing

**Components**:
- `dataset.py`: CUB-200-2011 dataset implementation
- `transforms.py`: Image augmentation and preprocessing

**Key Features**:
- Lazy loading for memory efficiency
- Stratified train/val split
- Support for both training and evaluation transforms
- Configurable batch size and workers

**Example Flow**:
```
Raw Images → Transform → Dataset → DataLoader → Batches
```

### 2. Models Module (`src/models/`)

**Purpose**: Define teacher and student architectures

**Components**:
- `teacher.py`: CLIP vision-language model (frozen)
- `student.py`: CNN architectures (trainable)

**Teacher Model (CLIPTeacher)**:
```
Input Image → CLIP Vision Encoder → {
    • Logits (for classification)
    • Attention Maps (optional)
    • Image Embeddings
}
```

**Student Model (StudentCNN)**:
```
Input Image → CNN Backbone → {
    • Logits (classification)
    • Feature Maps (for attention)
}
```

**Supported Architectures**:
- ResNet family (18, 34, 50)
- VGG family (16, 19)
- MobileNet (v3 small/large)
- EfficientNet (B0, B1)
- DenseNet (121)

### 3. Distillation Module (`src/distillation/`)

**Purpose**: Implement knowledge transfer mechanisms

**Components**:
- `losses.py`: All distillation loss functions
- `attention.py`: Attention map processing utilities
- `distiller.py`: High-level KD orchestrator

**Loss Functions**:

1. **Cross-Entropy Loss**:
   ```python
   L_CE = CrossEntropy(student_logits, ground_truth)
   ```

2. **Logit Distillation Loss** (KL Divergence):
   ```python
   L_KD = KL(softmax(student/T), softmax(teacher/T)) × T²
   ```

3. **Attention Distillation Loss**:
   ```python
   L_Attn = MSE(student_attention, teacher_attention)
   # or L1, KL variants
   ```

4. **Combined Loss**:
   ```python
   L_total = α_ce × L_CE + α_kd × L_KD + α_attn × L_Attn
   ```

**Attention Processing Pipeline**:
```
CLIP Attention Weights → Rollout → Normalize → Resize
                                                    ↓
CNN Feature Maps → Spatial Attention → Normalize → Match Resolution
                                                    ↓
                                    Compute Loss ← ←
```

### 4. Training Module (`src/training/`)

**Purpose**: Orchestrate training and evaluation

**Components**:
- `trainer.py`: Main training loop with KD support
- `evaluator.py`: Evaluation metrics and utilities

**Training Flow**:
```
┌─────────────────────────────────────────┐
│  1. Load Batch                          │
│     ↓                                   │
│  2. Teacher Forward (frozen)            │
│     • Get teacher logits               │
│     • Get teacher attention (optional) │
│     ↓                                   │
│  3. Student Forward (trainable)         │
│     • Get student logits               │
│     • Get student attention (optional) │
│     ↓                                   │
│  4. Compute Losses                      │
│     • Cross-entropy                    │
│     • Distillation (optional)          │
│     • Attention (optional)             │
│     ↓                                   │
│  5. Backward & Update                   │
│     • Compute gradients                │
│     • Update student weights           │
│     ↓                                   │
│  6. Log & Evaluate                      │
└─────────────────────────────────────────┘
```

### 5. Utils Module (`src/utils/`)

**Purpose**: Provide common utilities

**Components**:
- `config.py`: Configuration management (YAML/JSON)
- `helpers.py`: Helper functions (seed, device, paths)
- `metrics.py`: Metric tracking and statistics

## 🔄 Data Flow Diagrams

### Training with Combined Distillation

```
Input Batch (images, labels)
        │
        ├──────────────────┬──────────────────┐
        ↓                  ↓                  ↓
    Teacher            Student           Ground Truth
  (CLIP frozen)      (CNN train)
        │                  │                  │
        ├─ Logits ─────────┤                  │
        │                  │                  │
        ├─ Attention ──────┤                  │
        │                  │                  │
        ↓                  ↓                  ↓
    ┌────────────────────────────────────────────┐
    │         Loss Computation                   │
    │  • CE Loss (student vs labels)            │
    │  • KD Loss (student vs teacher logits)    │
    │  • Attn Loss (student vs teacher attn)    │
    └────────────────┬───────────────────────────┘
                     ↓
              Weighted Sum → Total Loss
                     ↓
              Backward Pass
                     ↓
         Update Student Weights
```

### Evaluation Flow

```
Test Batch
    ↓
Student Model (eval mode)
    ↓
Predictions
    ↓
┌───────────────────┐
│ Compute Metrics   │
├───────────────────┤
│ • Accuracy        │
│ • Top-5 Acc       │
│ • Loss            │
│ • Per-class Acc   │
│ • Confusion Mat   │
└───────────────────┘
    ↓
Save Results
```

## 🎯 Design Patterns

### 1. Factory Pattern
Used for model creation:
```python
def build_student(architecture: str, ...) -> StudentCNN:
    return StudentCNN(architecture=architecture, ...)
```

### 2. Strategy Pattern
Different distillation strategies:
```python
distiller = KnowledgeDistiller(
    distillation_type="logit"  # or "attention", "combined"
)
```

### 3. Builder Pattern
Configuration building:
```python
config = Config()
config.update(yaml_config)
config.update(cli_args)
```

### 4. Observer Pattern
Metrics tracking:
```python
tracker = MetricsTracker()
tracker.update({"loss": loss, "acc": acc})
```

## 📊 Configuration System

### Configuration Hierarchy

```
base.yaml (defaults)
    ↓
specific_config.yaml (overrides)
    ↓
command line args (final overrides)
    ↓
Runtime Config Object
```

### Config Loading Flow

```python
# 1. Load base config
config = Config()  # Default values

# 2. Load from file (optional)
if config_file:
    config = Config.load(config_file)

# 3. Override with CLI args
for arg, value in cli_args:
    setattr(config, arg, value)

# 4. Validate and use
config.validate()
```

## 🔧 Extension Points

### Adding New Student Architecture

1. Check if supported by timm:
   ```python
   timm.list_models("*your_arch*")
   ```

2. Add to ARCH_MAPPING in `student.py` (if needed)

3. Use in config:
   ```yaml
   student_architecture: "your_new_arch"
   ```

### Adding New Loss Function

1. Implement in `distillation/losses.py`:
   ```python
   def my_custom_loss(student_output, teacher_output) -> Tensor:
       # Your implementation
       return loss
   ```

2. Register in `combined_loss()` function

3. Add weight parameter to Config

### Adding New KD Method

1. Create new file in `distillation/`
2. Implement loss computation
3. Update `KnowledgeDistiller` class
4. Add configuration option

## 📈 Performance Considerations

### Memory Optimization
- Teacher model frozen (no gradients stored)
- Optional gradient checkpointing for large models
- Efficient data loading with pinned memory
- Optional mixed precision (AMP)

### Speed Optimization
- Multi-worker data loading
- Non-blocking GPU transfers
- Compiled models (torch.compile in future)
- Batch processing

### Scalability
- Modular design allows distributed training
- Configuration system supports hyperparameter search
- Efficient evaluation with minimal overhead

## 🧪 Testing Strategy

### Unit Tests (Planned)
- Data loading correctness
- Model output shapes
- Loss computation
- Metric calculation

### Integration Tests (Planned)
- End-to-end training pipeline
- Configuration loading
- Model saving/loading
- Evaluation accuracy

### Smoke Tests
- Quick training run (1 epoch)
- All architectures loadable
- All configs valid

## 📚 Dependencies

### Core Dependencies
```
torch >= 2.0.0          # PyTorch framework
torchvision >= 0.15.0   # Vision utilities
transformers >= 4.30.0  # CLIP model
timm >= 0.9.0           # CNN architectures
```

### Why These Versions?
- **PyTorch 2.0+**: Native AMP, better performance
- **Transformers 4.30+**: Stable CLIP implementation
- **timm 0.9+**: Wide architecture support, pretrained weights

## 🎓 Best Practices

### Code Organization
✅ Clear separation of concerns
✅ Type hints for better IDE support
✅ Comprehensive docstrings
✅ Modular, reusable components

### Configuration Management
✅ YAML for human readability
✅ Defaults in code, overrides in config
✅ Validation before training
✅ Save config with results

### Experiment Tracking
✅ Unique experiment names
✅ Timestamped output directories
✅ Save full configuration
✅ Log all metrics

### Performance
✅ Profile before optimizing
✅ Use appropriate batch sizes
✅ Enable mixed precision when possible
✅ Monitor GPU utilization

This architecture provides a solid foundation for research in knowledge distillation while remaining flexible and extensible for future improvements!
