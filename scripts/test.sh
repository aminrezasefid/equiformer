#!/bin/bash

# Load the required module
source /etc/profile
module load anaconda/2021a
export PYTHONNOUSERSITE=True    # prevent using packages from base

source activate equiformer

# Set variables
MODEL_PATH="/home/amin/equiformer/output/QM9-precise3d/best_model.pth"  # Update this to your model's path
OUTPUT_DIR="/home/amin/equiformer/output/QM9-precise3d"       # Update this to your desired output directory
DATASET_ROOT="../data/qm9"                   # Update this to your dataset root if needed
BATCH_SIZE=128
SEED=0
DATASET="QM9"
OUTPUT_CHANNELS="3"
# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"
#--dataset-args homo,lumo,gap\
# Run the test script
python test.py \
    --model-path "$MODEL_PATH" \
    --model-name 'graph_attention_transformer_nonlinear_l2' \
    --standardize True\
    --input-irreps '5x0e' \
    --radius 5.0 \
    --num-basis 128 \
    --drop-path 0.0 \
    --output-channels $OUTPUT_CHANNELS\
    --dataset-args homo,lumo,gap\
    --dataset "$DATASET"\
    --output-dir "$OUTPUT_DIR" \
    --dataset-root "$DATASET_ROOT" \
    --batch-size "$BATCH_SIZE" \
    --seed "$SEED"