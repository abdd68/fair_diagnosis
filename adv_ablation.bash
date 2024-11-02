#!/bin/bash

python adv.py --num_epochs 10 --noise_strength 0.2 --pretrain_epochs 10 -lr 1e-4 --feature mimic_alpha_00 --dataset mimic-cxr --debug --alpha 0 &
python adv.py --num_epochs 10 --noise_strength 0.2 --pretrain_epochs 10 -lr 1e-4 --feature mimic_alpha_01 --dataset mimic-cxr --debug --alpha 1
python adv.py --num_epochs 10 --noise_strength 0.2 --pretrain_epochs 10 -lr 1e-4 --feature mimic_alpha_02 --dataset mimic-cxr --debug --alpha 2 &
python adv.py --num_epochs 10 --noise_strength 0.2 --pretrain_epochs 10 -lr 1e-4 --feature mimic_alpha_03 --dataset mimic-cxr --debug --alpha 3
python adv.py --num_epochs 10 --noise_strength 0.2 --pretrain_epochs 10 -lr 1e-4 --feature mimic_alpha_04 --dataset mimic-cxr --debug --alpha 4 &
python adv.py --num_epochs 10 --noise_strength 0.2 --pretrain_epochs 10 -lr 1e-4 --feature mimic_alpha_05 --dataset mimic-cxr --debug --alpha 5

python adv.py --num_epochs 10 --noise_strength 0.2 --pretrain_epochs 1 -lr 1e-4 --feature chexpert_alpha_00 --dataset chexpert --alpha 0 &
python adv.py --num_epochs 10 --noise_strength 0.2 --pretrain_epochs 1 -lr 1e-4 --feature chexpert_alpha_01 --dataset chexpert --alpha 1
python adv.py --num_epochs 10 --noise_strength 0.2 --pretrain_epochs 1 -lr 1e-4 --feature chexpert_alpha_02 --dataset chexpert --alpha 2 &
python adv.py --num_epochs 10 --noise_strength 0.2 --pretrain_epochs 1 -lr 1e-4 --feature chexpert_alpha_03 --dataset chexpert --alpha 3
python adv.py --num_epochs 10 --noise_strength 0.2 --pretrain_epochs 1 -lr 1e-4 --feature chexpert_alpha_04 --dataset chexpert --alpha 4 &
python adv.py --num_epochs 10 --noise_strength 0.2 --pretrain_epochs 1 -lr 1e-4 --feature chexpert_alpha_05 --dataset chexpert --alpha 5

python adv.py --num_epochs 10 --noise_strength 0.2 --pretrain_epochs 10 -lr 1e-4 --feature tcga_alpha_00 --dataset tcga --debug --alpha 0 &
python adv.py --num_epochs 10 --noise_strength 0.2 --pretrain_epochs 10 -lr 1e-4 --feature tcga_alpha_01 --dataset tcga --debug --alpha 1
python adv.py --num_epochs 10 --noise_strength 0.2 --pretrain_epochs 10 -lr 1e-4 --feature tcga_alpha_02 --dataset tcga --debug --alpha 2 &
python adv.py --num_epochs 10 --noise_strength 0.2 --pretrain_epochs 10 -lr 1e-4 --feature tcga_alpha_03 --dataset tcga --debug --alpha 3
python adv.py --num_epochs 10 --noise_strength 0.2 --pretrain_epochs 10 -lr 1e-4 --feature tcga_alpha_04 --dataset tcga --debug --alpha 4 &
python adv.py --num_epochs 10 --noise_strength 0.2 --pretrain_epochs 10 -lr 1e-4 --feature tcga_alpha_05 --dataset tcga --debug --alpha 5



python adv.py --num_epochs 10 --noise_strength 0.2 --pretrain_epochs 10 -lr 1e-4 --feature mimic_beta_00 --dataset mimic-cxr --debug --beta 0 &
python adv.py --num_epochs 10 --noise_strength 0.2 --pretrain_epochs 10 -lr 1e-4 --feature mimic_beta_01 --dataset mimic-cxr --debug --beta 1
python adv.py --num_epochs 10 --noise_strength 0.2 --pretrain_epochs 10 -lr 1e-4 --feature mimic_beta_02 --dataset mimic-cxr --debug --beta 2 &
python adv.py --num_epochs 10 --noise_strength 0.2 --pretrain_epochs 10 -lr 1e-4 --feature mimic_beta_03 --dataset mimic-cxr --debug --beta 3
python adv.py --num_epochs 10 --noise_strength 0.2 --pretrain_epochs 10 -lr 1e-4 --feature mimic_beta_04 --dataset mimic-cxr --debug --beta 4 &
python adv.py --num_epochs 10 --noise_strength 0.2 --pretrain_epochs 10 -lr 1e-4 --feature mimic_beta_05 --dataset mimic-cxr --debug --beta 5

python adv.py --num_epochs 10 --noise_strength 0.2 --pretrain_epochs 1 -lr 1e-4 --feature chexpert_beta_00 --dataset chexpert --beta 0 &
python adv.py --num_epochs 10 --noise_strength 0.2 --pretrain_epochs 1 -lr 1e-4 --feature chexpert_beta_01 --dataset chexpert --beta 1
python adv.py --num_epochs 10 --noise_strength 0.2 --pretrain_epochs 1 -lr 1e-4 --feature chexpert_beta_02 --dataset chexpert --beta 2 &
python adv.py --num_epochs 10 --noise_strength 0.2 --pretrain_epochs 1 -lr 1e-4 --feature chexpert_beta_03 --dataset chexpert --beta 3
python adv.py --num_epochs 10 --noise_strength 0.2 --pretrain_epochs 1 -lr 1e-4 --feature chexpert_beta_04 --dataset chexpert --beta 4 &
python adv.py --num_epochs 10 --noise_strength 0.2 --pretrain_epochs 1 -lr 1e-4 --feature chexpert_beta_05 --dataset chexpert --beta 5

python adv.py --num_epochs 10 --noise_strength 0.2 --pretrain_epochs 10 -lr 1e-4 --feature tcga_beta_00 --dataset tcga --debug --beta 0 &
python adv.py --num_epochs 10 --noise_strength 0.2 --pretrain_epochs 10 -lr 1e-4 --feature tcga_beta_01 --dataset tcga --debug --beta 1
python adv.py --num_epochs 10 --noise_strength 0.2 --pretrain_epochs 10 -lr 1e-4 --feature tcga_beta_02 --dataset tcga --debug --beta 2 &
python adv.py --num_epochs 10 --noise_strength 0.2 --pretrain_epochs 10 -lr 1e-4 --feature tcga_beta_03 --dataset tcga --debug --beta 3
python adv.py --num_epochs 10 --noise_strength 0.2 --pretrain_epochs 10 -lr 1e-4 --feature tcga_beta_04 --dataset tcga --debug --beta 4 &
python adv.py --num_epochs 10 --noise_strength 0.2 --pretrain_epochs 10 -lr 1e-4 --feature tcga_beta_05 --dataset tcga --debug --beta 5



python adv.py --num_epochs 10 --noise_strength 0.2 --pretrain_epochs 10 -lr 1e-4 --feature mimic_threshold_00 --dataset mimic-cxr --debug --threshold 0.0 &
python adv.py --num_epochs 10 --noise_strength 0.2 --pretrain_epochs 10 -lr 1e-4 --feature mimic_threshold_02 --dataset mimic-cxr --debug --threshold 0.2
python adv.py --num_epochs 10 --noise_strength 0.2 --pretrain_epochs 10 -lr 1e-4 --feature mimic_threshold_04 --dataset mimic-cxr --debug --threshold 0.4 &
python adv.py --num_epochs 10 --noise_strength 0.2 --pretrain_epochs 10 -lr 1e-4 --feature mimic_threshold_06 --dataset mimic-cxr --debug --threshold 0.6
python adv.py --num_epochs 10 --noise_strength 0.2 --pretrain_epochs 10 -lr 1e-4 --feature mimic_threshold_08 --dataset mimic-cxr --debug --threshold 0.8 &
python adv.py --num_epochs 10 --noise_strength 0.2 --pretrain_epochs 10 -lr 1e-4 --feature mimic_threshold_10 --dataset mimic-cxr --debug --threshold 0.9999

python adv.py --num_epochs 10 --noise_strength 0.2 --pretrain_epochs 1 -lr 1e-4 --feature chexpert_threshold_00 --dataset chexpert --threshold 0.0 &
python adv.py --num_epochs 10 --noise_strength 0.2 --pretrain_epochs 1 -lr 1e-4 --feature chexpert_threshold_02 --dataset chexpert --threshold 0.2
python adv.py --num_epochs 10 --noise_strength 0.2 --pretrain_epochs 1 -lr 1e-4 --feature chexpert_threshold_04 --dataset chexpert --threshold 0.4 &
python adv.py --num_epochs 10 --noise_strength 0.2 --pretrain_epochs 1 -lr 1e-4 --feature chexpert_threshold_06 --dataset chexpert --threshold 0.6
python adv.py --num_epochs 10 --noise_strength 0.2 --pretrain_epochs 1 -lr 1e-4 --feature chexpert_threshold_08 --dataset chexpert --threshold 0.8 &
python adv.py --num_epochs 10 --noise_strength 0.2 --pretrain_epochs 1 -lr 1e-4 --feature chexpert_threshold_10 --dataset chexpert --threshold 0.9999

python adv.py --num_epochs 10 --noise_strength 0.2 --pretrain_epochs 10 -lr 1e-4 --feature tcga_threshold_00 --dataset tcga --debug --threshold 0.0 &
python adv.py --num_epochs 10 --noise_strength 0.2 --pretrain_epochs 10 -lr 1e-4 --feature tcga_threshold_02 --dataset tcga --debug --threshold 0.2
python adv.py --num_epochs 10 --noise_strength 0.2 --pretrain_epochs 10 -lr 1e-4 --feature tcga_threshold_04 --dataset tcga --debug --threshold 0.4 &
python adv.py --num_epochs 10 --noise_strength 0.2 --pretrain_epochs 10 -lr 1e-4 --feature tcga_threshold_06 --dataset tcga --debug --threshold 0.6
python adv.py --num_epochs 10 --noise_strength 0.2 --pretrain_epochs 10 -lr 1e-4 --feature tcga_threshold_08 --dataset tcga --debug --threshold 0.8 &
python adv.py --num_epochs 10 --noise_strength 0.2 --pretrain_epochs 10 -lr 1e-4 --feature tcga_threshold_10 --dataset tcga --debug --threshold 0.9999