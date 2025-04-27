#!/bin/bash
#SBATCH -J CIFAR10-LISNN-Pushpull
#SBATCH --output=slurm-%j_x.out
#SBATCH --array=0
#SBATCH -c 8
#SBATCH --time=00-02:00:00
#SBATCH --gres=gpu:2
#SBATCH --mem=24GB
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

DATASET_DIR="$TMPDIR/dataset/"


mkdir -p "$TMPDIR/dataset"

echo "MASTER_ADDR=$MASTER_ADDR"
echo "WORLD_SIZE=$WORLD_SIZE"

echo "Extracting CIFAR-10C.tar"
tar -xf "/scratch/p315895/datasets/CIFAR-10-C.tar" -C "$TMPDIR/dataset"

ls "$TMPDIR/dataset/"
ls "$TMPDIR/dataset/CIFAR-10-C"


# Runtime variables
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SEED=32
SESSION_ID=$(openssl rand -hex 4)


torchrun --standalone --nproc_per_node=2 \
  main_finetune.py \
  --batch_size 32 \
  --model spiliformer_tiny \
  --data_path ../data/cifar10/ \
  --output_dir "outputs/CIFAR10/${TIMESTAMP}_eval/" \
  --log_dir "outputs/CIFAR10/${TIMESTAMP}_eval/" \
  --model_mode ms \
  --dataset CIFAR10 \
  --seed ${SEED} \
  --eval \
  --resume /home4/p315895/Spike-Driven-Transformer-V2/classification/outputs/CIFAR10/20250424_235937/checkpoint-399.pth \
  --c_eval




#/home4/p315895/Spike-Driven-Transformer-V2/classification/outputs/CIFAR10/20250425_223838_pushpull/checkpoint-399.pth