#!/bin/bash

python adv.py --num_epochs 10 --noise_strength 0.1 --pretrain_epochs 10 --model vit -lr 1e-4 --feature vit_total_n01 &
python adv.py --num_epochs 10 --noise_strength 0.2 --pretrain_epochs 10 --model vit -lr 1e-4 --feature vit_total_n02 &
python adv.py --num_epochs 10 --noise_strength 0.3 --pretrain_epochs 10 --model vit -lr 1e-4 --feature vit_total_n03 &
python adv.py --num_epochs 10 --noise_strength 0.4 --pretrain_epochs 10 --model vit -lr 1e-4 --feature vit_total_n04 &
python adv.py --num_epochs 10 --noise_strength 0.5 --pretrain_epochs 10 --model vit -lr 1e-4 --feature vit_total_n05

# python adv.py --num_epochs 10 --noise_strength 0.1 --pretrain_epochs 10 --model vit -lr 1e-4 --feature vit_total_ch_n01 --dataset chexpert &
# python adv.py --num_epochs 10 --noise_strength 0.2 --pretrain_epochs 10 --model vit -lr 1e-4 --feature vit_total_ch_n02 --dataset chexpert &
# python adv.py --num_epochs 10 --noise_strength 0.3 --pretrain_epochs 10 --model vit -lr 1e-4 --feature vit_total_ch_n03 --dataset chexpert &
# python adv.py --num_epochs 10 --noise_strength 0.4 --pretrain_epochs 10 --model vit -lr 1e-4 --feature vit_total_ch_n04 --dataset chexpert &
# python adv.py --num_epochs 10 --noise_strength 0.5 --pretrain_epochs 10 --model vit -lr 1e-4 --feature vit_total_ch_n05 --dataset chexpert

# python adv.py --num_epochs 10 --noise_strength 0.1 --pretrain_epochs 10 --model vit -lr 1e-4 --feature vit_total_tcga_n01 --dataset tcga &
# python adv.py --num_epochs 10 --noise_strength 0.2 --pretrain_epochs 10 --model vit -lr 1e-4 --feature vit_total_tcga_n02 --dataset tcga &
# python adv.py --num_epochs 10 --noise_strength 0.3 --pretrain_epochs 10 --model vit -lr 1e-4 --feature vit_total_tcga_n03 --dataset tcga &
# python adv.py --num_epochs 10 --noise_strength 0.4 --pretrain_epochs 10 --model vit -lr 1e-4 --feature vit_total_tcga_n04 --dataset tcga &
# python adv.py --num_epochs 10 --noise_strength 0.5 --pretrain_epochs 10 --model vit -lr 1e-4 --feature vit_total_tcga_n05 --dataset tcga

# python adv.py --num_epochs 10 --noise_strength 0.1 --pretrain_epochs 10 -lr 1e-4 --feature total_n01 &
# python adv.py --num_epochs 10 --noise_strength 0.2 --pretrain_epochs 10 -lr 1e-4 --feature total_n02 &
# python adv.py --num_epochs 10 --noise_strength 0.3 --pretrain_epochs 10 -lr 1e-4 --feature total_n03 &
# python adv.py --num_epochs 10 --noise_strength 0.4 --pretrain_epochs 10 -lr 1e-4 --feature total_n04 &
# python adv.py --num_epochs 10 --noise_strength 0.5 --pretrain_epochs 10 -lr 1e-4 --feature total_n05

# python adv.py --num_epochs 10 --noise_strength 0.1 --pretrain_epochs 10 -lr 1e-4 --feature total_ch_n01 --dataset chexpert &
# python adv.py --num_epochs 10 --noise_strength 0.2 --pretrain_epochs 10 -lr 1e-4 --feature total_ch_n02 --dataset chexpert &
# python adv.py --num_epochs 10 --noise_strength 0.3 --pretrain_epochs 10 -lr 1e-4 --feature total_ch_n03 --dataset chexpert &
# python adv.py --num_epochs 10 --noise_strength 0.4 --pretrain_epochs 10 -lr 1e-4 --feature total_ch_n04 --dataset chexpert &
# python adv.py --num_epochs 10 --noise_strength 0.5 --pretrain_epochs 10 -lr 1e-4 --feature total_ch_n05 --dataset chexpert

# python adv.py --num_epochs 10 --noise_strength 0.1 --pretrain_epochs 10 -lr 1e-4 --feature total_tcga_n01 --dataset tcga &
# python adv.py --num_epochs 10 --noise_strength 0.2 --pretrain_epochs 10 -lr 1e-4 --feature total_tcga_n02 --dataset tcga &
# python adv.py --num_epochs 10 --noise_strength 0.3 --pretrain_epochs 10 -lr 1e-4 --feature total_tcga_n03 --dataset tcga &
# python adv.py --num_epochs 10 --noise_strength 0.4 --pretrain_epochs 10 -lr 1e-4 --feature total_tcga_n04 --dataset tcga &
# python adv.py --num_epochs 10 --noise_strength 0.5 --pretrain_epochs 10 -lr 1e-4 --feature total_tcga_n05 --dataset tcga