#!/usr/bin/env bash
#SBATCH --job-name=NOTEARSLin
#SBATCH --output=/home/ethan/notears.out
#SBATCH --error=/home/ethan/notears_err.out
#SBATCH --time=300:00:00
#SBATCH --cpus-per-task=50
#SBATCH --mem=100000
#SBATCH --gres=gpu:0    # NOTEARS is CPU-only

set -e

source activate ift6168_env

NODE_SIZES=(10 20 30 40 50 60)
OUT_CSV="/home/ethan/IFT6168/exp_scratch/notears_results.csv"
mkdir -p "$(dirname "$OUT_CSV")"

for NODES in "${NODE_SIZES[@]}"; do
  EDGES=$(( NODES * 3 ))
  BASE_DIR="/home/ethan/IFT6168/project/data/simulated/data_p${NODES}_e${EDGES}_n10000_scaling_nodes_p${NODES}"
  
  echo "=== Node size: $NODES  (edges=$EDGES) ==="
  # iterate over every subdirectory in $BASE_DIR (each replicate)
  for REPL in 1 2 3 4 5; do
    echo "  • Running NOTEARS on replicate #$REPL in $BASE_DIR"
    python /home/ethan/IFT6168/project/notears/run_notears_linear.py \
      --data-dir   "$BASE_DIR" \
      --i-dataset  "$REPL" \
      --output-csv "$OUT_CSV"
  done
done

