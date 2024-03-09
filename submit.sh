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
python3 gate.py -1 30 10 512
