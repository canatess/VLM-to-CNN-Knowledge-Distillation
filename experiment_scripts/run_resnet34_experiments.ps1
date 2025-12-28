# ResNet-34 experiments

.venv\Scripts\python.exe scripts/train.py --student_architecture resnet34 --pretrained false --distillation_type none --experiment_name resnet34_scratch --num_epochs 20 --batch_size 32

.venv\Scripts\python.exe scripts/train.py --student_architecture resnet34 --pretrained true --distillation_type none --experiment_name resnet34_transfer --num_epochs 20 --batch_size 32

.venv\Scripts\python.exe scripts/train.py --config configs/logit_kd.yaml --student_architecture resnet34 --experiment_name resnet34_logit_kd --num_epochs 20

.venv\Scripts\python.exe scripts/train.py --config configs/attention_kd.yaml --student_architecture resnet34 --experiment_name resnet34_attention_kd --num_epochs 20

.venv\Scripts\python.exe scripts/train.py --config configs/combined_kd.yaml --student_architecture resnet34 --experiment_name resnet34_combined_kd --num_epochs 20
