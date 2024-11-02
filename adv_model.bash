#!/bin/bash

python adv.py --num_epochs 10 --noise_strength 0.2 --pretrain_epochs 20 -lr 1e-4 --feature resnet50_mimic --debug --model resnet50 &
python adv.py --num_epochs 10 --noise_strength 0.2 --pretrain_epochs 20 -lr 1e-4 --feature resnet50_chexpert --dataset chexpert --debug --model resnet50
python adv.py --num_epochs 10 --noise_strength 0.2 --pretrain_epochs 20 -lr 1e-4 --feature resnet50_tcga --dataset tcga --debug --model resnet50 &

python adv.py --num_epochs 20 --noise_strength 0.2 --pretrain_epochs 150 -lr 1e-4 --feature vit_mimic --debug --model vit
python adv.py --num_epochs 20 --noise_strength 0.2 --pretrain_epochs 150 -lr 1e-4 --feature vit_chexpert --dataset chexpert --debug --model vit &
python adv.py --num_epochs 20 --noise_strength 0.2 --pretrain_epochs 150 -lr 1e-4 --feature vit_tcga --dataset tcga --debug --model vit
