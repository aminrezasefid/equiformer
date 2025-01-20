#!/bin/bash

# Loading the required module
source /etc/profile
module load anaconda/2021a

export PYTHONNOUSERSITE=True    # prevent using packages from base
source activate equiformer

# Define a list of structures to iterate over
structures=("precise3d" "rdkit3d" "optimized3d" "rdkit2d")  # Add your structures here
structures=("precise3d" "optimized3d" "rdkit2d")
datasets=("Bace")
# Iterate over each structure
for structure in "${structures[@]}"; do
    for dataset in "${datasets[@]}"; do
        dataset_root="../data/${dataset,,}"  # Convert dataset name to lowercase
        
        python train.py \
            --output-dir "../output/" \
            --model-name 'graph_attention_transformer_nonlinear_l2' \
            --input-irreps '5x0e' \
            --dataset "$dataset" \
            --epochs 100 \
            --task-type "class"\
            --dataset-root "$dataset_root" \
            --feature-type 'one_hot' \
            --no-standardize\
            --batch-size 8 \
            --radius 5.0 \
            --train-size 0.8\
            --val-size 0.1\
            --test-size 0.1\
            --output-channels 1\
            --num-basis 128 \
            --drop-path 0.0 \
            --weight-decay 5e-3 \
            --lr 5e-4 \
            --min-lr 1e-6 \
            --no-model-ema \
            --no-amp \
            --structure "$structure"
    done
done
