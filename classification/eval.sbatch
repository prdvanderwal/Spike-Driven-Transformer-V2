#!/bin/bash
#SBATCH -J CIFAR10-LISNN-Pushpull
#SBATCH --output=slurm-%j_x.out
#SBATCH --array=0-1
#SBATCH -c 8
#SBATCH --time=00-01:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=24GB
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

DATASET_DIR="$TMPDIR/dataset/"


mkdir -p "$TMPDIR/dataset"

echo "MASTER_ADDR=$MASTER_ADDR"
echo "WORLD_SIZE=$WORLD_SIZE"

echo "Extracting CIFAR-10C.tar"
tar -xf "/scratch/p315895/datasets/CIFAR-10-C.tar" -C "$TMPDIR/dataset"

ls "$TMPDIR/dataset/"
ls "$TMPDIR/dataset/CIFAR-10-C"


# Runtime variables
DATASET_NAME="CIFAR10"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SEED=32
SESSION_ID=$(openssl rand -hex 4)

COMMON_ARGS="--batch_size 128 --model spiliformer_tiny --data_path ../data/cifar10/ \
             --model_mode ms --dataset ${DATASET_NAME} --seed ${SEED} --eval --c_eval \
             --time_steps 4 --wandb"

EXPERIMENT_NAME="outputs/${DATASET_NAME}/${TIMESTAMP}_s${SEED}"


# Initialize job array
case ${SLURM_ARRAY_TASK_ID} in
0) torchrun --standalone --nproc_per_node=1 main_finetune.py ${COMMON_ARGS} --push_pull --lateral_inhibition --wandb_tags CIFAR10 only-pp tiny --resume /home4/p315895/Spike-Driven-Transformer-V2/classification/outputs/CIFAR10/20250429_100501_s32-onlypp/best_model.pth;;
1) torchrun --standalone --nproc_per_node=1 main_finetune.py ${COMMON_ARGS} --wandb_tags CIFAR10 no-inhi tiny --resume /home4/p315895/Spike-Driven-Transformer-V2/classification/outputs/CIFAR10/20250428_224411_s32_noinhibition/best_model.pth
esac

echo "script finished"


<<'END'
  




#/home4/p315895/Spike-Driven-Transformer-V2/classification/outputs/CIFAR10/20250425_223838_pushpull/checkpoint-399.pth