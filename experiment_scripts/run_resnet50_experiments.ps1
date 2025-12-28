# ResNet-50 experiments

.venv\Scripts\python.exe scripts/train.py --student_architecture resnet50 --pretrained false --distillation_type none --experiment_name resnet50_scratch --num_epochs 20 --batch_size 32

.venv\Scripts\python.exe scripts/train.py --student_architecture resnet50 --pretrained true --distillation_type none --experiment_name resnet50_transfer --num_epochs 20 --batch_size 32

.venv\Scripts\python.exe scripts/train.py --config configs/logit_kd.yaml --student_architecture resnet50 --experiment_name resnet50_logit_kd --num_epochs 20

.venv\Scripts\python.exe scripts/train.py --config configs/attention_kd.yaml --student_architecture resnet50 --experiment_name resnet50_attention_kd --num_epochs 20

.venv\Scripts\python.exe scripts/train.py --config configs/combined_kd.yaml --student_architecture resnet50 --experiment_name resnet50_combined_kd --num_epochs 20
