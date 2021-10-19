#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --mem=60g
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --gres=gpu:4
#SBATCH --output=./ruche_logs/%j.out
#SBATCH --open-mode=append
#SBATCH --signal=B:SIGUSR1@120
#SBATCH --job-name='queue_and_run'

source venv/bin/activate

python -m scripts.queue_experiments -f grid.yaml

dvc exp run --run-all --jobs 4
