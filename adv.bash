#!/bin/bash

python adv.py --num_epochs 5 --noise_strength 0.1 --pretrain_epochs 10 -lr 1e-4 --no_adversarial
python adv.py --num_epochs 5 --noise_strength 0.1 --pretrain_epochs 5 -lr 1e-4