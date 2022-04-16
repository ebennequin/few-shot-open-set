#!/bin/bash
#SBATCH --mem=10000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6   # There are 40 CPU cores on Beluga GPU nodes
#SBATCH --time=02:00:00
#SBATCH --array=0-5
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
BACKBONE=resnet12
DATA_DIR=${SLURM_TMPDIR}/data
mkdir -p ${DATA_DIR}
tar xf ~/scratch/open-set/data/${DATASET}.tar.gz -C ${DATA_DIR}
mkdir -p ${DATA_DIR}/features
cp -Rv data/features/${DATASET} ${DATA_DIR}/features/

METHODS=(LaplacianShot ICI TIM_GD BDCSPN Finetune MAP)
TRANSFORMS=("Pool BaseCentering L2norm" "Pool" "Pool" "Pool" "Pool" "Pool Power QRreduction L2norm MeanCentering")

# for SLURM_ARRAY_TASK_ID in {1..5}; do
method=${METHODS[$((SLURM_ARRAY_TASK_ID))]}
transforms=${TRANSFORMS[$((SLURM_ARRAY_TASK_ID))]}
make EXP=tune_${method} \
     DATADIR=${DATA_DIR} \
     SPLIT=val \
     TUNE="classifier transforms" \
     N_TASKS=500 \
     CLS_TRANSFORMS="${transforms}" \
     SRC_DATASET=${DATASET} \
     TGT_DATASET=${DATASET} \
     BACKBONE=resnet12 \
     CLASSIFIER=${method} \
     run
# done


