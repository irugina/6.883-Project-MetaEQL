#!/bin/sh
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:volta:1
#SBATCH -o results/log1.out
#SBATCH --job-name=eql1

python -u eql_maml.py --m=1 --exp_number=1
