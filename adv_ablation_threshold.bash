#!/bin/bash

python adv.py --num_epochs 10 --noise_strength 0.2 --pretrain_epochs 10 -lr 1e-4 --feature mimic_threshold_00 --dataset mimic-cxr --debug --threshold 0.0 &
python adv.py --num_epochs 10 --noise_strength 0.2 --pretrain_epochs 10 -lr 1e-4 --feature mimic_threshold_02 --dataset mimic-cxr --debug --threshold 0.2
python adv.py --num_epochs 10 --noise_strength 0.2 --pretrain_epochs 10 -lr 1e-4 --feature mimic_threshold_04 --dataset mimic-cxr --debug --threshold 0.4 &
python adv.py --num_epochs 10 --noise_strength 0.2 --pretrain_epochs 10 -lr 1e-4 --feature mimic_threshold_06 --dataset mimic-cxr --debug --threshold 0.6
python adv.py --num_epochs 10 --noise_strength 0.2 --pretrain_epochs 10 -lr 1e-4 --feature mimic_threshold_08 --dataset mimic-cxr --debug --threshold 0.8 &
python adv.py --num_epochs 10 --noise_strength 0.2 --pretrain_epochs 10 -lr 1e-4 --feature mimic_threshold_10 --dataset mimic-cxr --debug --threshold 0.9999

python adv.py --num_epochs 10 --noise_strength 0.2 --pretrain_epochs 10 -lr 1e-4 --feature chexpert_threshold_00 --dataset chexpert --threshold 0.0 &
python adv.py --num_epochs 10 --noise_strength 0.2 --pretrain_epochs 10 -lr 1e-4 --feature chexpert_threshold_02 --dataset chexpert --threshold 0.2
python adv.py --num_epochs 10 --noise_strength 0.2 --pretrain_epochs 10 -lr 1e-4 --feature chexpert_threshold_04 --dataset chexpert --threshold 0.4 &
python adv.py --num_epochs 10 --noise_strength 0.2 --pretrain_epochs 10 -lr 1e-4 --feature chexpert_threshold_06 --dataset chexpert --threshold 0.6
python adv.py --num_epochs 10 --noise_strength 0.2 --pretrain_epochs 10 -lr 1e-4 --feature chexpert_threshold_08 --dataset chexpert --threshold 0.8 &
python adv.py --num_epochs 10 --noise_strength 0.2 --pretrain_epochs 10 -lr 1e-4 --feature chexpert_threshold_10 --dataset chexpert --threshold 0.9999

python adv.py --num_epochs 10 --noise_strength 0.2 --pretrain_epochs 10 -lr 1e-4 --feature tcga_threshold_00 --dataset tcga --debug --threshold 0.0 &
python adv.py --num_epochs 10 --noise_strength 0.2 --pretrain_epochs 10 -lr 1e-4 --feature tcga_threshold_02 --dataset tcga --debug --threshold 0.2
python adv.py --num_epochs 10 --noise_strength 0.2 --pretrain_epochs 10 -lr 1e-4 --feature tcga_threshold_04 --dataset tcga --debug --threshold 0.4 &
python adv.py --num_epochs 10 --noise_strength 0.2 --pretrain_epochs 10 -lr 1e-4 --feature tcga_threshold_06 --dataset tcga --debug --threshold 0.6
python adv.py --num_epochs 10 --noise_strength 0.2 --pretrain_epochs 10 -lr 1e-4 --feature tcga_threshold_08 --dataset tcga --debug --threshold 0.8 &
python adv.py --num_epochs 10 --noise_strength 0.2 --pretrain_epochs 10 -lr 1e-4 --feature tcga_threshold_10 --dataset tcga --debug --threshold 0.9999