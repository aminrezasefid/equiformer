#!/bin/bash

# Default values
CUDA_DEVICE=0

STRUCTURES="precise3d,rdkit3d,optimized3d,rdkit2d"
#STRUCTURES="optimized3d"  # Default structure
DATASETS="TOXCAST"         # Default dataset

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --cuda-device)
            CUDA_DEVICE="$2"
            shift 2
            ;;
        --structures)
            STRUCTURES="$2"
            shift 2
            ;;
        --datasets)
            DATASETS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: ./tox21.sh --cuda-device <device> --structures <struct1,struct2,...> --datasets <dataset1,dataset2,...>"
            exit 1
            ;;
    esac
done

# Loading the required module
source /etc/profile
module load anaconda/2021a

export PYTHONNOUSERSITE=True    # prevent using packages from base
source activate equiformer

# Convert comma-separated strings to arrays
IFS=',' read -ra structures <<< "$STRUCTURES"
IFS=',' read -ra datasets <<< "$DATASETS"

# Iterate over each structure
for structure in "${structures[@]}"; do
    for dataset in "${datasets[@]}"; do
        dataset_root="../data/${dataset,,}"  # Convert dataset name to lowercase
        
        echo "Running with structure: $structure, dataset: $dataset, CUDA device: $CUDA_DEVICE"
        
        python train.py \
            --output-dir "../output/" \
            --model-name 'graph_attention_transformer_nonlinear_l2' \
            --input-irreps '5x0e' \
            --dataset "$dataset" \
            --epochs 100 \
            --task-type "class" \
            --dataset-root "$dataset_root" \
            --feature-type 'one_hot' \
            --no-standardize \
            --batch-size 32 \
            --radius 5.0 \
            --train-size 0.8 \
            --num-workers 16 \
            --val-size 0.1 \
            --test-size 0.1 \
            --output-channels 617 \
            --num-basis 128 \
            --drop-path 0.0 \
            --weight-decay 5e-3 \
            --lr 5e-4 \
            --min-lr 1e-6 \
            --no-model-ema \
            --no-amp \
            --structure "$structure" \
            --cuda-device "$CUDA_DEVICE"
    done
done
