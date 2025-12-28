# VGG-19 experiments

python scripts/train.py --student_architecture vgg19 --pretrained false --distillation_type none --experiment_name vgg19_scratch --num_epochs 30 --batch_size 32

python scripts/train.py --student_architecture vgg19 --pretrained true --distillation_type none --experiment_name vgg19_transfer --num_epochs 30 --batch_size 32

python scripts/train.py --config configs/logit_kd.yaml --student_architecture vgg19 --experiment_name vgg19_logit_kd --num_epochs 30

python scripts/train.py --config configs/attention_kd.yaml --student_architecture vgg19 --experiment_name vgg19_attention_kd --num_epochs 30

python scripts/train.py --config configs/combined_kd.yaml --student_architecture vgg19 --experiment_name vgg19_combined_kd --num_epochs 30
