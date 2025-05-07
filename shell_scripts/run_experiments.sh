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

# Allow GPU_ID and i_dataset to be set via command-line arguments, with defaults
GPU_ID=${1:-0}          # Default GPU_ID is 0
i_dataset=${2:-1}       # Default i_dataset is 1

export CUDA_VISIBLE_DEVICES=$GPU_ID

echo "Running with GPU $GPU_ID --i-dataset $i_dataset"
# Iterate over different values for --regimes-to-ignore
for i in {0..9}; do
    EXP_PATH="/cim/ehoney/IFT6168_project/experiments/proper/dataset_${i_dataset}_regimes-ignored_${i}"
    mkdir -p "$EXP_PATH"
    if [ "$i" -eq 0 ]; then
        echo "Regimes ignored: NONE"
        $BASE_CMD --i-dataset $i_dataset --exp-path $EXP_PATH
    else
        regimes=$(seq -s " " 1 $i)
        echo "Regimes ignored: $regimes"
        $BASE_CMD --i-dataset $i_dataset --exp-path $EXP_PATH --regimes-to-ignore $regimes
    fi
done