python scripts/train.py --student_architecture resnet18 --pretrained false --distillation_type none --experiment_name resnet18_scratch --num_epochs 30 --batch_size 32

python scripts/train.py --student_architecture resnet18 --pretrained true --distillation_type none --experiment_name resnet18_transfer --num_epochs 30 --batch_size 32

python scripts/train.py --config configs/logit_kd.yaml --student_architecture resnet18 --experiment_name resnet18_logit_kd --num_epochs 30

python scripts/train.py --config configs/attention_kd.yaml --student_architecture resnet18 --experiment_name resnet18_attention_kd --num_epochs 30

python scripts/train.py --config configs/combined_kd.yaml --student_architecture resnet18 --experiment_name resnet18_combined_kd --num_epochs 30
