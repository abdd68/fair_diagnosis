#!/bin/bash

# python adv.py --num_epochs 10 --noise_strength 0 --pretrain_epochs 20 -lr 1e-4 --feature mimic_eta_00 --debug &
# python adv.py --num_epochs 10 --noise_strength 0.1 --pretrain_epochs 20 -lr 1e-4 --feature mimic_eta_01 --debug 
# python adv.py --num_epochs 10 --noise_strength 0.2 --pretrain_epochs 20 -lr 1e-4 --feature mimic_eta_02 --debug &
# python adv.py --num_epochs 10 --noise_strength 0.3 --pretrain_epochs 20 -lr 1e-4 --feature mimic_eta_03 --debug 
# python adv.py --num_epochs 10 --noise_strength 0.4 --pretrain_epochs 20 -lr 1e-4 --feature mimic_eta_04 --debug &
# python adv.py --num_epochs 10 --noise_strength 0.5 --pretrain_epochs 20 -lr 1e-4 --feature mimic_eta_05 --debug 

# python adv.py --num_epochs 10 --noise_strength 0 --pretrain_epochs 20 -lr 1e-4 --feature chexpert_eta_00 --dataset chexpert --debug &
# python adv.py --num_epochs 10 --noise_strength 0.1 --pretrain_epochs 20 -lr 1e-4 --feature chexpert_eta_01 --dataset chexpert --debug 
# python adv.py --num_epochs 10 --noise_strength 0.2 --pretrain_epochs 20 -lr 1e-4 --feature chexpert_eta_02 --dataset chexpert --debug &
# python adv.py --num_epochs 10 --noise_strength 0.3 --pretrain_epochs 20 -lr 1e-4 --feature chexpert_eta_03 --dataset chexpert --debug 
# python adv.py --num_epochs 10 --noise_strength 0.4 --pretrain_epochs 20 -lr 1e-4 --feature chexpert_eta_04 --dataset chexpert --debug &
# python adv.py --num_epochs 10 --noise_strength 0.5 --pretrain_epochs 20 -lr 1e-4 --feature chexpert_eta_05 --dataset chexpert --debug 

# python adv.py --num_epochs 10 --noise_strength 0 --pretrain_epochs 20 -lr 1e-4 --feature tcga_eta_00 --dataset tcga --debug &
# python adv.py --num_epochs 10 --noise_strength 0.1 --pretrain_epochs 20 -lr 1e-4 --feature tcga_eta_01 --dataset tcga --debug 
# python adv.py --num_epochs 10 --noise_strength 0.2 --pretrain_epochs 20 -lr 1e-4 --feature tcga_eta_02 --dataset tcga --debug &
# python adv.py --num_epochs 10 --noise_strength 0.3 --pretrain_epochs 20 -lr 1e-4 --feature tcga_eta_03 --dataset tcga --debug 
# python adv.py --num_epochs 10 --noise_strength 0.4 --pretrain_epochs 20 -lr 1e-4 --feature tcga_eta_04 --dataset tcga --debug &
# python adv.py --num_epochs 10 --noise_strength 0.5 --pretrain_epochs 20 -lr 1e-4 --feature tcga_eta_05 --dataset tcga --debug 

python adv.py --num_epochs 10 --noise_strength 0 --pretrain_epochs 10 -lr 1e-4 --feature chexpert_eta_00 --dataset chexpert &
python adv.py --num_epochs 10 --noise_strength 0.1 --pretrain_epochs 10 -lr 1e-4 --feature chexpert_eta_01 --dataset chexpert  
python adv.py --num_epochs 10 --noise_strength 0.2 --pretrain_epochs 10 -lr 1e-4 --feature chexpert_eta_02 --dataset chexpert &
python adv.py --num_epochs 10 --noise_strength 0.3 --pretrain_epochs 10 -lr 1e-4 --feature chexpert_eta_03 --dataset chexpert 
python adv.py --num_epochs 10 --noise_strength 0.4 --pretrain_epochs 10 -lr 1e-4 --feature chexpert_eta_04 --dataset chexpert &
python adv.py --num_epochs 10 --noise_strength 0.5 --pretrain_epochs 10 -lr 1e-4 --feature chexpert_eta_05 --dataset chexpert 