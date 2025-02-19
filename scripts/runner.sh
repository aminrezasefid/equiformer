#!/bin/bash

# Default values
CUDA_DEVICE=0
DATASET_ARGS=none
STRUCTURES="precise3d,rdkit3d,optimized3d,rdkit2d"
OUTPUT_CHANNELS=1
BATCH_SIZE=64
EPOCHS=100  # Add default epochs value
#STRUCTURES="optimized3d"  # Default structure
#DATASETS="Clintox"         # Default dataset

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
        --dataset-args)
            DATASET_ARGS="$2"
            shift 2
            ;;
        --output-channels)
            OUTPUT_CHANNELS="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --epochs)  # Add new epochs argument
            EPOCHS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: ./tox21.sh --cuda-device <device> --structures <struct1,struct2,...> --datasets <dataset1,dataset2,...> --dataset-args <args> --output-channels <num> --batch-size <size> --epochs <num>"
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
        
        python_args=(
            --output-dir "../output/"
            --model-name 'graph_attention_transformer_nonlinear_l2'
            --input-irreps '5x0e'
            --dataset "$dataset"
            --epochs "$EPOCHS"  # Update epochs to use the variable
            --task-type "class"
            --dataset-root "$dataset_root"
            --feature-type 'one_hot'
            --no-standardize
            --batch-size "$BATCH_SIZE"
            --radius 5.0
            --train-size 0.8
            --num-workers 8
            --val-size 0.1
            --test-size 0.1
            --output-channels "$OUTPUT_CHANNELS"
            --num-basis 128
            --drop-path 0.0
            --weight-decay 5e-3
            --lr 5e-4
            --min-lr 1e-6
            --no-model-ema
            --no-amp
            --structure "$structure"
            --cuda-device "$CUDA_DEVICE"
        )

        # Only add dataset-args if it's not "none"
        if [ "$DATASET_ARGS" != "none" ]; then
            python_args+=(--dataset-args "$DATASET_ARGS")
        fi

        python train.py "${python_args[@]}"
    done
done
