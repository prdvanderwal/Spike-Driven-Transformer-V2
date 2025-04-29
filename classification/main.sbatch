#!/bin/bash
#SBATCH -J CIFAR10-LISNN-Pushpull
#SBATCH --output=slurm-%j_x.out
#SBATCH --array=0
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

COMMON_ARGS="--batch_size 32 --epochs 300 --model spiliformer_tiny --data_path ../data/cifar10/ \
             --model_mode ms --layer_decay 1.0 --lr 1e-3 --time_steps 4 --weight_decay 6e-2 \
             --dataset ${DATASET_NAME} --seed ${SEED} --color_jitter 0.2 --aa rand-m9 \
             --reprob 0.1 --remode pixel --recount 1"
EXPERIMENT_NAME="${TIMESTAMP}_s${SEED}"


# Initialize job array
case ${SLURM_ARRAY_TASK_ID} in
0) torchrun --standalone --nproc_per_node=2 main_finetune.py ${COMMON_ARGS} --push_pull --name ${EXPERIMENT_NAME} --wandb_tags ${DATASET_NAME} pp "s${SEED}" ;;
# 0) torchrun --standalone --nproc_per_node=2 main_finetune.py ${COMMON_ARGS} --output_dir ${EXPERIMENT_NAME} --log_dir ${EXPERIMENT_NAME};;
esac

echo "script finished"

<<'END'


