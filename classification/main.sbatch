#!/bin/bash
#SBATCH -J IN100-PP
#SBATCH --output=slurm-%j_x.out
#SBATCH --array=0
#SBATCH -c 8
#SBATCH --time=00-00:15:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32GB
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8

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

DATASET_NAME="ImageNet-100"
DATASET_DIR="$TMPDIR/dataset/"

mkdir -p "$TMPDIR/dataset"

# Extract dataset
SECONDS=0
echo "Extracting ${DATASET_NAME}.tar"
tar -xf "/scratch/hb-prismaset/datasets/${DATASET_NAME}.tar" -C "$TMPDIR/dataset"
echo "Time taken to extract dataset: ${SECONDS} seconds"

COMMON_ARGS="--stage1 token --stage2 token --stage3 spiking \
             --batch_size 8 --epochs 90 --model QKFormer_10_768 --data_path ${DATASET_DIR}/ImageNet-100 \
             --model_mode ms --dataset ${DATASET_NAME} --seed ${SEED} --time_steps 1"
EXPERIMENT_NAME="${TIMESTAMP}_s${SEED}"


# Initialize job array
case ${SLURM_ARRAY_TASK_ID} in
0) torchrun --standalone --nproc_per_node=1 main_finetune.py ${COMMON_ARGS} --name ${EXPERIMENT_NAME} --wandb_tags ${DATASET_NAME} QK "s${SEED}" ;;
# 0) torchrun --standalone --nproc_per_node=2 main_finetune.py ${COMMON_ARGS} --output_dir ${EXPERIMENT_NAME} --log_dir ${EXPERIMENT_NAME};;
esac

echo "script finished"

<<'END'


