#!/usr/bin/env bash

## Name of your SLURM job
#SBATCH --job-name=DCDI

## Files for logs: here we redirect stoout and sterr to the same file
#SBATCH --output=/home/ethan/test_DCDI.out
#SBATCH --error=/home/ethan/test_DCDIerr.out
#SBATCH --open-mode=append

## Time limit for the job
#SBATCH --time=200:00:00

## How many CPUs to request. Maximum is 124.
#SBATCH --cpus-per-task=50

## How much memory to request in MB. Maximum is 460GB.
#SBATCH --mem=100000

## How many GPU to request. Maximum is 3.
#SBATCH --gres=gpu:1

## You can also request a percentage of one GPU.
## Example to get 20% of a GPU.
## This approach has severe limitation as it can only be used by
## a single user at a time. More tests should be performed whether this
## ok to use it. Please experiment and report findings.
##SBATCH --gres=mps:20

set -e

cd /home/ethan/IFT6168/project/dcdi

echo "SLURM_JOB_UID=$SLURM_JOB_UID"
echo "SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
echo "SLURM_ARRAY_TASK_COUNT=$SLURM_ARRAY_TASK_COUNT"

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"

# In a SLURM job, you CANNOT use `conda activate` and instead MUST use:
source activate ift6168_env

NODE_SIZES=(10 20 30 40 50)

for NODES in "${NODE_SIZES[@]}"; do
    EDGES=$((NODES * 3))
    DATA_PATH="/home/ethan/IFT6168/project/data/simulated/data_p${NODES}_e${EDGES}_n10000_scaling_nodes_p${NODES}"
    for DATASET_INDEX in {1..5}; do
        echo "Running DCDI for d=$NODES, i_dataset=$DATASET_INDEX"

        # Create unique experiment path
        EXP_PATH="/home/ethan/IFT6168/exp_scratch/data_p${NODES}_${DATASET_INDEX}"
        mkdir -p "$EXP_PATH"

        python main.py \
            --train \
            --data-path $DATA_PATH \
            --num-vars $NODES \
            --i-dataset $DATASET_INDEX \
            --exp-path $EXP_PATH \
            --model DCDI-DSF \
            --intervention \
            --intervention-type perfect \
            --intervention-knowledge known \
            --reg-coeff 0.1 \
            --gpu
    done
done

# Optional: allow a pause before job release
sleep 20s