import argparse
import os
import torch
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
print(SCRIPT_DIR)
sys.path.append(os.path.dirname(SCRIPT_DIR))
import datasets
from module_utils.test import test_model
from datasets.datamodule import DataModule
from nets import model_entrypoint
from optim_factory import create_optimizer
from timm.utils import NativeScaler
from engine import evaluate

def number(text):
    if text is None or text == "None":
        return None

    try:
        num_int = int(text)
    except ValueError:
        num_int = None
    num_float = float(text)

    if num_int == num_float:
        return num_int
    return num_float
def get_args_parser():
    parser = argparse.ArgumentParser('Testing equivariant networks', add_help=False)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument('--input-irreps', type=str, default=None)
    parser.add_argument('--drop-path', type=float, default=0.0)
    parser.add_argument('--radius', type=float, default=2.0)
    parser.add_argument('--num-basis', type=int, default=32)
    parser.add_argument('--output-channels', type=int, default=1)
    parser.add_argument('--model-name', type=str, default='transformer_ti')
    parser.add_argument("--inference-batch-size", type=int, default=128)
    parser.add_argument('--standardize', type=bool, default=False, help='If true, multiply prediction by dataset std and add mean')
    parser.add_argument('--dataset', default=None, type=str, choices=datasets.__all__, help='Name of the torch_geometric dataset')
    parser.add_argument('--dataset-args', type=str, default=None, help='Comma-separated list of dataset arguments')
    parser.add_argument('--split', type=str, default='scaffold', choices=['random', 'scaffold'], help='Split type')
    parser.add_argument('--train-size', type=number, default=0.8, help='Percentage/number of samples in training set (None to use all remaining samples)')
    parser.add_argument('--val-size', type=number, default=0.1, help='Percentage/number of samples in validation set (None to use all remaining samples)')
    parser.add_argument('--test-size', type=number, default=0.1, help='Percentage/number of samples in test set (None to use all remaining samples)')
    parser.add_argument('--structure', choices=["precise3d", "rdkit3d", "optimized3d", "rdkit2d", "pubchem3d"], default="precise3d", help='Structure of the input data')
    parser.add_argument('--model-path', type=str, required=True, help='Path to the saved model')
    parser.add_argument('--output-dir', type=str, default='.', help='Directory to save results')
    parser.add_argument('--dataset-root', default='data', type=str, help='Data storage directory')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size for testing')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    return parser



if __name__ == "__main__":
    parser = argparse.ArgumentParser('Testing equivariant networks', parents=[get_args_parser()])
    
    args = parser.parse_args()
    if args.dataset_args:
        args.dataset_args = args.dataset_args.split(',')  # Convert comma-separated string to list
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)  # Create output directory if it doesn't exist
    test_model(args)
