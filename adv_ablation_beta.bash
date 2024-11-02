#!/bin/bash

python adv.py --num_epochs 10 --noise_strength 0.2 --pretrain_epochs 10 -lr 1e-4 --feature mimic_beta_00 --dataset mimic-cxr --debug --beta 0 &
python adv.py --num_epochs 10 --noise_strength 0.2 --pretrain_epochs 10 -lr 1e-4 --feature mimic_beta_01 --dataset mimic-cxr --debug --beta 1
python adv.py --num_epochs 10 --noise_strength 0.2 --pretrain_epochs 10 -lr 1e-4 --feature mimic_beta_02 --dataset mimic-cxr --debug --beta 2 &
python adv.py --num_epochs 10 --noise_strength 0.2 --pretrain_epochs 10 -lr 1e-4 --feature mimic_beta_03 --dataset mimic-cxr --debug --beta 3
python adv.py --num_epochs 10 --noise_strength 0.2 --pretrain_epochs 10 -lr 1e-4 --feature mimic_beta_04 --dataset mimic-cxr --debug --beta 4 &
python adv.py --num_epochs 10 --noise_strength 0.2 --pretrain_epochs 10 -lr 1e-4 --feature mimic_beta_05 --dataset mimic-cxr --debug --beta 5

python adv.py --num_epochs 10 --noise_strength 0.2 --pretrain_epochs 10 -lr 1e-4 --feature chexpert_beta_00 --dataset chexpert --beta 0 &
python adv.py --num_epochs 10 --noise_strength 0.2 --pretrain_epochs 10 -lr 1e-4 --feature chexpert_beta_01 --dataset chexpert --beta 1
python adv.py --num_epochs 10 --noise_strength 0.2 --pretrain_epochs 10 -lr 1e-4 --feature chexpert_beta_02 --dataset chexpert --beta 2 &
python adv.py --num_epochs 10 --noise_strength 0.2 --pretrain_epochs 10 -lr 1e-4 --feature chexpert_beta_03 --dataset chexpert --beta 3
python adv.py --num_epochs 10 --noise_strength 0.2 --pretrain_epochs 10 -lr 1e-4 --feature chexpert_beta_04 --dataset chexpert --beta 4 &
python adv.py --num_epochs 10 --noise_strength 0.2 --pretrain_epochs 10 -lr 1e-4 --feature chexpert_beta_05 --dataset chexpert --beta 5

python adv.py --num_epochs 10 --noise_strength 0.2 --pretrain_epochs 10 -lr 1e-4 --feature tcga_beta_00 --dataset tcga --debug --beta 0 &
python adv.py --num_epochs 10 --noise_strength 0.2 --pretrain_epochs 10 -lr 1e-4 --feature tcga_beta_01 --dataset tcga --debug --beta 1
python adv.py --num_epochs 10 --noise_strength 0.2 --pretrain_epochs 10 -lr 1e-4 --feature tcga_beta_02 --dataset tcga --debug --beta 2 &
python adv.py --num_epochs 10 --noise_strength 0.2 --pretrain_epochs 10 -lr 1e-4 --feature tcga_beta_03 --dataset tcga --debug --beta 3
python adv.py --num_epochs 10 --noise_strength 0.2 --pretrain_epochs 10 -lr 1e-4 --feature tcga_beta_04 --dataset tcga --debug --beta 4 &
python adv.py --num_epochs 10 --noise_strength 0.2 --pretrain_epochs 10 -lr 1e-4 --feature tcga_beta_05 --dataset tcga --debug --beta 5