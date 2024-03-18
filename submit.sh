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
python3 KD.py 6 0.2 10 10 512
python3 KD.py 6 0.4 10 10 512
python3 KD.py 6 0.8 10 10 512
python3 KD.py 4 0.5 10 10 512
python3 KD.py 6 0.5 10 10 512
python3 KD.py 8 0.5 10 10 512
python3 KD.py 2 0.5 10 10 512
# 10 epochs, 10 tests, 512 batch size
