#!/bin/bash
#SBATCH --mem=10000
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=6   # There are 40 CPU cores on Beluga GPU nodes
#SBATCH --time=03:00:00
#SBATCH --array=0-0
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



DATASET=mini_imagenet
BACKBONE=resnet12
DATA_DIR=${SLURM_TMPDIR}/data
mkdir -p $DATA_DIR
tar xf ~/scratch/open-set/data/${DATASET}.tar.gz -C ${DATA_DIR}
mkdir -p ${DATA_DIR}/features
cp -Rv data/features/${DATASET} ${DATA_DIR}/features/

make benchmark_pyod_detectors
