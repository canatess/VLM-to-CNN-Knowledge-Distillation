# VGG-16 experiments

python scripts/train.py --student_architecture vgg16 --pretrained false --distillation_type none --experiment_name vgg16_scratch --num_epochs 30 --batch_size 32

python scripts/train.py --student_architecture vgg16 --pretrained true --distillation_type none --experiment_name vgg16_transfer --num_epochs 30 --batch_size 32

python scripts/train.py --config configs/logit_kd.yaml --student_architecture vgg16 --experiment_name vgg16_logit_kd --num_epochs 30

python scripts/train.py --config configs/attention_kd.yaml --student_architecture vgg16 --experiment_name vgg16_attention_kd --num_epochs 30

python scripts/train.py --config configs/combined_kd.yaml --student_architecture vgg16 --experiment_name vgg16_combined_kd --num_epochs 30
