#!/bin/bash
#SBATCH -J imagenet1k
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=chenxiang.zhang@uni.lu
#SBATCH --account=p200535
#SBATCH --qos=default
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --gpus=4
#SBATCH --partition=gpu
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm-%x-%j.out

echo -e "--------------------------------"
echo -e "Start:\t $(date)"
echo -e "JobID:\t ${SLURM_JOBID}"
echo -e "Node:\t ${SLURM_NODELIST}"
echo -e "--------------------------------\n"

eval "$(micromamba shell hook --shell bash)"
micromamba activate tests

export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=8
export NCCL_DEBUG=INFO
export HF_HOME="/project/home/p200535/.cache"

mkdir -p checkpoints
mkdir -p logs

# Training configuration
BATCH_SIZE=256  # Per GPU (total = 256 * 4 = 1024)
EPOCHS=90
LR=0.001
WEIGHT_DECAY=0.0001
WARMUP_EPOCHS=5
NUM_WORKERS=8

torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=4 \
    train.py \
    --epochs $EPOCHS \
    --bs $BATCH_SIZE \
    --lr $LR \
    --wd $WEIGHT_DECAY \
    --warmup_epochs $WARMUP_EPOCHS \
    --num_workers $NUM_WORKERS \
    --dir_output ./checkpoints
