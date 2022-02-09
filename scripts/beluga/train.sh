#!/bin/bash
#SBATCH --mem=20000 # Require full memory
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=12   # There are 40 CPU cores on Beluga GPU nodes
#SBATCH --time=02:30:00
#SBATCH --array=0-0
#SBATCH --account=rrg-ebrahimi

#SBATCH --mail-user=malik.boudiaf.1@etsmtl.net
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL

# ETA with 2 GPUs
# For mini : 2 h 30
# For tiered: 24 h

source ~/.bash_profile
module load python/3.8.2
source ~/ENV/bin/activate

# Extracting data to current node for fast I/O

DATASET=mini_imagenet
BACKBONE=resnet12
DATA_DIR=${SLURM_TMPDIR}/data
mkdir -p $DATA_DIR
tar xf ~/scratch/open-set/data/${DATASET}.tar.gz -C ${DATA_DIR}


make GPUS="0 1" \
     EXP=${BACKBONE} \
     DATADIR=${DATA_DIR} \
     BACKBONE=${BACKBONE} \
     DATASETS=${DATASET} \
     experimental_training

