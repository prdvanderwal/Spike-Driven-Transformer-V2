#!/bin/bash
#SBATCH -J pp
#SBATCH --output=slurm-%j_x.out
#SBATCH --array=0-3
#SBATCH -c 8
#SBATCH --time=00-04:00:00
#SBATCH --gres=gpu:2
#SBATCH --mem=32GB
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16

set -e
set -x

# Activate virtual environment
source $HOME/venvs/lisnn/bin/activate

# Set distributed training variables
BASE_PORT=12346
export MASTER_PORT=$((BASE_PORT + SLURM_ARRAY_TASK_ID))
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export OMP_NUM_THREADS=8

echo "MASTER_ADDR=$MASTER_ADDR"
echo "WORLD_SIZE=$WORLD_SIZE"

# Runtime variables
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SEED=32
SESSION_ID=$(openssl rand -hex 4)

DATASET_NAME="CIFAR10"

EXPERIMENT_NAME="${TIMESTAMP}_s${SEED}"


# Initialize job array
case ${SLURM_ARRAY_TASK_ID} in
0) torchrun --standalone --nproc_per_node=2 train.py --name ${EXPERIMENT_NAME} --wandb_tags s1-lidiff s2-pptoken s3-spiking --stage1 lidiff --stage2 pptoken --stage3 spiking;;
1) torchrun --standalone --nproc_per_node=2 train.py --name ${EXPERIMENT_NAME} --wandb_tags s1-pptoken s2-pptoken s3-spiking --stage1 pptoken --stage2 pptoken --stage3 spiking;;
2) torchrun --standalone --nproc_per_node=2 train.py --name ${EXPERIMENT_NAME} --wandb_tags s1-lidiff s2-lidiff s3-spiking --stage1 lidiff --stage2 lidiff --stage3 spiking;;
3) torchrun --standalone --nproc_per_node=2 train.py --name ${EXPERIMENT_NAME} --wandb_tags s1-token s2-token s3-spiking --stage1 token --stage2 token --stage3 spiking;;
esac

echo "script finished"

<<'END'


