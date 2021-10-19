#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --mem=60g
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --gres=gpu:4
#SBATCH --output=%j.out
#SBATCH --error=%j.out
#SBATCH --open-mode=append
#SBATCH --signal=B:SIGUSR1@120
#SBATCH --job-name='run_dvc_queue'

source venv/bin/activate

dvc exp run --run-all --jobs 4
