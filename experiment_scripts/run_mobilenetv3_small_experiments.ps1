# MobileNetV3-Small experiments

.venv\Scripts\python.exe scripts/train.py --student_architecture mobilenetv3_small --pretrained false --distillation_type none --experiment_name mobilenetv3_small_scratch --num_epochs 20 --batch_size 32

.venv\Scripts\python.exe scripts/train.py --student_architecture mobilenetv3_small --pretrained true --distillation_type none --experiment_name mobilenetv3_small_transfer --num_epochs 20 --batch_size 32

.venv\Scripts\python.exe scripts/train.py --config configs/logit_kd.yaml --student_architecture mobilenetv3_small --experiment_name mobilenetv3_small_logit_kd --num_epochs 20

.venv\Scripts\python.exe scripts/train.py --config configs/attention_kd.yaml --student_architecture mobilenetv3_small --experiment_name mobilenetv3_small_attention_kd --num_epochs 20

.venv\Scripts\python.exe scripts/train.py --config configs/combined_kd.yaml --student_architecture mobilenetv3_small --experiment_name mobilenetv3_small_combined_kd --num_epochs 20
