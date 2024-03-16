#!/bin/bash
#
#SBATCH --job-name=TLGate
#SBATCH --partition=soundbendor
#SBATCH -A soundbendor
#SBATCH --gres=gpu:2
#SBATCH -o sbatch_logs/main.out
#SBATCH -e sbatch_logs/main.err

module load slurm
module load cuda/12.2
source env/bin/activate
python3 KD.py 1 1 512


# 30 epochs, 10 tests, 512 batch size
