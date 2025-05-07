#!/bin/bash

# Activate Conda environment
source /cim/ehoney/miniconda3/etc/profile.d/conda.sh  # Adjust path if Conda is installed elsewhere
conda activate dcdi_conda_env

# Base command
BASE_CMD="python /cim/ehoney/IFT6168_project/dcdi/main.py \
    --train \
    --data-path /cim/ehoney/IFT6168_project/dcdi/data/perfect/data_p10_e10_n10000_linear_struct \
    --num-vars 10 \
    --model DCDI-G \
    --intervention \
    --intervention-type perfect \
    --intervention-knowledge known \
    --reg-coeff 0.5 \
    --gpu \
    --plot-freq 1000"

# Default GPU_ID is 0
GPU_ID=0
export CUDA_VISIBLE_DEVICES=$GPU_ID

echo "Running with GPU $GPU_ID"

# Iterate over i_dataset from 1 to 10
for i_dataset in {1..10}; do
    EXP_PATH="/cim/ehoney/IFT6168_project/experiments/proper/dataset_${i_dataset}_regimes-ignored_all"
    mkdir -p "$EXP_PATH"
    regimes=$(seq -s " " 1 10)
    echo "Running i-dataset $i_dataset with regimes ignored: $regimes"
    $BASE_CMD --i-dataset $i_dataset --exp-path $EXP_PATH --regimes-to-ignore $regimes
done
