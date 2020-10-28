#!/bin/sh
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:volta:1
#SBATCH -o results/log4.out
#SBATCH --job-name=eql4

python -u eql_maml.py --m=1 --exp_number=7
