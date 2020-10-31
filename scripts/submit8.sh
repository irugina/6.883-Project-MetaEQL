#!/bin/sh
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:volta:1
#SBATCH -o results/log8.out
#SBATCH --job-name=eql8

python -u eql_maml.py --m=2 --exp_number=7
