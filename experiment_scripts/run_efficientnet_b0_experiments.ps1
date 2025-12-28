# EfficientNet-B0 experiments

.venv\Scripts\python.exe scripts/train.py --student_architecture efficientnet_b0 --pretrained false --distillation_type none --experiment_name efficientnet_b0_scratch --num_epochs 20 --batch_size 32

.venv\Scripts\python.exe scripts/train.py --student_architecture efficientnet_b0 --pretrained true --distillation_type none --experiment_name efficientnet_b0_transfer --num_epochs 20 --batch_size 32

.venv\Scripts\python.exe scripts/train.py --config configs/logit_kd.yaml --student_architecture efficientnet_b0 --experiment_name efficientnet_b0_logit_kd --num_epochs 20

.venv\Scripts\python.exe scripts/train.py --config configs/attention_kd.yaml --student_architecture efficientnet_b0 --experiment_name efficientnet_b0_attention_kd --num_epochs 20

.venv\Scripts\python.exe scripts/train.py --config configs/combined_kd.yaml --student_architecture efficientnet_b0 --experiment_name efficientnet_b0_combined_kd --num_epochs 20
