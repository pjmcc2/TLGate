#!/bin/bash
#
#SBATCH --job-name=TLGate
#SBATCH --partition=dgxs,dgx2,dgxh,soundbendor
#SBATCH -A soundbendor
#SBATCH --gres=gpu:2
#SBATCH -o sbatch_logs/main.out
#SBATCH -e sbatch_logs/main.err

module load slurm
module load cuda/12.2
source genv/bin/activate
python3 gate.py 1 1 30 10 512

#gatetype Temp. 10 epochs, 10 tests, 512 batch size
#python3 gate.py 0 30 10 512
#python3 gate.py 1 30 10 512
#python3 gate.py 2 30 10 512
#python3 gate.py all 30 10 512