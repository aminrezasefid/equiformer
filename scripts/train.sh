#!/bin/bash

# Loading the required module
source /etc/profile
module load anaconda/2021a
export TORCH_CUDA_ARCH_LIST="8.9"
export HYDRA_FULL_ERROR="1"
export PYTHONNOUSERSITE=True    # prevent using packages from base
source activate equiformer

HYDRA_FULL_ERROR="1" TORCH_CUDA_ARCH_LIST="8.9" python train.py \
    --output-dir 'models/qm7/equiformer/' \
    --model-name 'graph_attention_transformer_nonlinear_l2' \
    --input-irreps '5x0e' \
    --dataset 'QM7' \
    --dataset-root '../data/qm7' \
    --feature-type 'one_hot' \
    --standardize 'True' \
    --batch-size 10 \
    --radius 5.0 \
    --num-basis 128 \
    --drop-path 0.0 \
    --weight-decay 5e-3 \
    --lr 5e-4 \
    --min-lr 1e-6 \
    --no-model-ema \
    --no-amp
