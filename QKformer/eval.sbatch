#!/bin/bash
#SBATCH -J pp
#SBATCH --output=slurm-%j_x.out
#SBATCH --array=0
#SBATCH -c 8
#SBATCH --time=00-02:00:00
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

echo "MASTER_ADDR=$MASTER_ADDR"
echo "WORLD_SIZE=$WORLD_SIZE"

# Runtime variables
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SEED=32
SESSION_ID=$(openssl rand -hex 4)

DATASET_NAME="CIFAR10"

DATASET_DIR="$TMPDIR/dataset/"
mkdir -p "$TMPDIR/dataset"

echo "Extracting CIFAR-10C.tar"
tar -xf "/scratch/p315895/datasets/CIFAR-10-C.tar" -C "$TMPDIR/dataset"

EXPERIMENT_NAME="${TIMESTAMP}_s${SEED}"
COMMON_ARGS="--stage1 lidiff lidiff pptoken token --stage2 lidiff pptoken pptoken token --stage3 spiking spiking spiking spiking \
             --resume_paths /home4/p315895/Spike-Driven-Transformer-V2/QKformer/checkpoints/cifar10/li-li-spi-e300-s32/model_best.pth.tar \
             /home4/p315895/Spike-Driven-Transformer-V2/QKformer/checkpoints/cifar10/li-ppt-spi-e300-s32/model_best.pth.tar \
             /home4/p315895/Spike-Driven-Transformer-V2/QKformer/checkpoints/cifar10/ppt-ppt-spi-e300-s32/model_best.pth.tar \
             /home4/p315895/Spike-Driven-Transformer-V2/QKformer/checkpoints/cifar10/tok-tok-spi-e300-s32/model_best.pth.tar \
             --experiment_name li-li-spi li-ppt-spi ppt-ppt-spi tok-tok-spi \
             --spike_rate" 

# Initialize job array
case ${SLURM_ARRAY_TASK_ID} in
0) torchrun --standalone --nproc_per_node=1 multiple_evals.py --name ${EXPERIMENT_NAME} ${COMMON_ARGS} --wandb_tags CIFAR10 Eval 4v spike_rate;;

esac

echo "script finished"

<<'END'


