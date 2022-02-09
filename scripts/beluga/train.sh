#!/bin/bash
#SBATCH --mem=32000 # Require full memory
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=10   # There are 40 CPU cores on Beluga GPU nodes
#SBATCH --time=02:30:00
#SBATCH --array=0-3
#SBATCH --account=rrg-ebrahimi

#SBATCH --mail-user=malik.boudiaf.1@etsmtl.net
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL



source ~/.bash_profile
module load python/3.8.2
source ~/ENV/bin/activate

# Extracting data to current node for fast I/O

DATASET=mini_imagenet
DATA_DIR=${SLURM_TMPDIR}/data
mkdir -p $DATA_DIR
tar xf ~/scratch/open-set/data/${DATASET}.tar.gz -C ${DATA_DIR}


make GPUS="0 1" DATADIR=${DATA_DIR} BACKBONE=resnet12 DATASETS=${DATASET} experimental_training

