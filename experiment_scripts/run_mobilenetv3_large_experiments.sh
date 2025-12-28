# MobileNetV3-Large experiments

python scripts/train.py --student_architecture mobilenetv3_large --pretrained false --distillation_type none --experiment_name mobilenetv3_large_scratch --num_epochs 30 --batch_size 32

python scripts/train.py --student_architecture mobilenetv3_large --pretrained true --distillation_type none --experiment_name mobilenetv3_large_transfer --num_epochs 30 --batch_size 32

python scripts/train.py --config configs/logit_kd.yaml --student_architecture mobilenetv3_large --experiment_name mobilenetv3_large_logit_kd --num_epochs 30

python scripts/train.py --config configs/attention_kd.yaml --student_architecture mobilenetv3_large --experiment_name mobilenetv3_large_attention_kd --num_epochs 30

python scripts/train.py --config configs/combined_kd.yaml --student_architecture mobilenetv3_large --experiment_name mobilenetv3_large_combined_kd --num_epochs 30
