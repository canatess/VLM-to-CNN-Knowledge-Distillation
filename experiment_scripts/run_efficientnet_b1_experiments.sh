# EfficientNet-B1 experiments

python scripts/train.py --student_architecture efficientnet_b1 --pretrained false --distillation_type none --experiment_name efficientnet_b1_scratch --num_epochs 30 --batch_size 32

python scripts/train.py --student_architecture efficientnet_b1 --pretrained true --distillation_type none --experiment_name efficientnet_b1_transfer --num_epochs 30 --batch_size 32

python scripts/train.py --config configs/logit_kd.yaml --student_architecture efficientnet_b1 --experiment_name efficientnet_b1_logit_kd --num_epochs 30

python scripts/train.py --config configs/attention_kd.yaml --student_architecture efficientnet_b1 --experiment_name efficientnet_b1_attention_kd --num_epochs 30

python scripts/train.py --config configs/combined_kd.yaml --student_architecture efficientnet_b1 --experiment_name efficientnet_b1_combined_kd --num_epochs 30
