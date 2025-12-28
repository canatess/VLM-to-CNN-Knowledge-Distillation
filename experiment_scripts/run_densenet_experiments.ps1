# DenseNet-121 experiments

.venv\Scripts\python.exe scripts/train.py --student_architecture densenet121 --pretrained false --distillation_type none --experiment_name densenet121_scratch --num_epochs 20 --batch_size 32

.venv\Scripts\python.exe scripts/train.py --student_architecture densenet121 --pretrained true --distillation_type none --experiment_name densenet121_transfer --num_epochs 20 --batch_size 32

.venv\Scripts\python.exe scripts/train.py --config configs/logit_kd.yaml --student_architecture densenet121 --experiment_name densenet121_logit_kd --num_epochs 20

.venv\Scripts\python.exe scripts/train.py --config configs/attention_kd.yaml --student_architecture densenet121 --experiment_name densenet121_attention_kd --num_epochs 20

.venv\Scripts\python.exe scripts/train.py --config configs/combined_kd.yaml --student_architecture densenet121 --experiment_name densenet121_combined_kd --num_epochs 20
