# Experiment Scripts

This folder contains PowerShell scripts to run the full 5-experiment suite for different CNN architectures.

## Experiments Run by Each Script

Each script runs 5 experiments sequentially:

1. **From Scratch**: Train without pretrained weights or distillation
2. **Transfer Learning**: Train with ImageNet pretrained weights (no distillation)
3. **Logit-based KD**: Knowledge distillation using soft labels from CLIP teacher
4. **Attention-based KD**: Knowledge distillation using spatial attention maps from CLIP teacher
5. **Combined KD**: Combined logit and attention distillation

## Usage

```powershell
# Execute a script

.\experiment_scripts\run_resnet34_experiments.ps1
.\experiment_scripts\run_mobilenetv3_large_experiments.ps1
```

## Configuration

Edit the variables at the top of each script to modify:

- `$EPOCHS` - Number of training epochs (default: 20)
- `$BATCH_SIZE` - Batch size (default: 32)
- `$ARCH` - Model architecture name

## Output

Results are saved to the `outputs/` directory with the following structure:
```
outputs/
├── resnet18_scratch/
├── resnet18_transfer/
├── resnet18_logit_kd/
├── resnet18_attention_kd/
└── resnet18_combined_kd/
```

Each experiment folder contains:
- `config.yaml` - Complete configuration used
- `best_model.pth` - Best model checkpoint
- Training logs and metrics

## Error Handling

Scripts will stop if any experiment fails and display an error message. Check the terminal output to see which experiment encountered an issue.
