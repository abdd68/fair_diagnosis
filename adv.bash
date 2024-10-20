#!/bin/bash

python adv.py --num_epochs 3 --noise_strength 0.1 --debug  --pretrain_epochs 5 -lr 8e-5
python adv.py --num_epochs 3 --noise_strength 0.1 --debug  --pretrain_epochs 5 -lr 6e-5
python adv.py --num_epochs 3 --noise_strength 0.1 --debug  --pretrain_epochs 5 -lr 4e-5
python adv.py --num_epochs 3 --noise_strength 0.1 --debug  --pretrain_epochs 5 -lr 2e-5
python adv.py --num_epochs 3 --noise_strength 0.1 --debug  --pretrain_epochs 5 -lr 8e-6