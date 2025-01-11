import argparse
import datetime
import itertools
import pickle
import subprocess
import time
import os
import torch
import sys
import pandas as pd
import numpy as np
from torch_geometric.loader import DataLoader
from typing import List
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
print(SCRIPT_DIR)
sys.path.append(os.path.dirname(SCRIPT_DIR))
import datasets
from logger import FileLogger
from pathlib import Path
from datasets.datamodule import DataModule
from contextlib import suppress
from timm.utils import NativeScaler

import nets
from nets import model_entrypoint

from timm.utils import ModelEmaV2
from timm.scheduler import create_scheduler
from optim_factory import create_optimizer

from engine import train_one_epoch, evaluate, compute_stats

def test_model(args):
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load the dataset
    data = DataModule(args)
    data.setup()
    target_names = data.dataset.target_names
    task_mean, task_std = 0, 1
    if args.standardize:
        task_mean, task_std = data.mean, data.std    

    # Load the model
    create_model = model_entrypoint(args.model_name)
    model = create_model(irreps_in=args.input_irreps, 
        radius=args.radius, num_basis=args.num_basis, 
        output_channels=args.output_channels, 
        task_mean=task_mean, 
        task_std=task_std, 
        atomref=None, #train_dataset.atomref(args.target),
        drop_path=args.drop_path,
        unique_atomic_numbers=data.get_unique_atomic_numbers())
    model.load_state_dict(torch.load(args.model_path))  # Load the saved model
    model.eval()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    norm_factor = [task_mean.to(device), task_std.to(device)]
    model.to(device)

    # Prepare to collect results
    
    target_names = data.dataset.target_names
    print(data.dataset.target_names)
    # Combine all target metrics into a single dictionary
    results = {"smiles":[]}
    for target in target_names:
        results[f"pred_{target}"] = []
        results[f"actual_{target}"] = []
        results[f"diff_{target}"] = []

    with torch.no_grad():
        for data_batch in data.test_dataloader():
            data_batch = data_batch.to(device)
            pred = model(f_in=data_batch.x, pos=data_batch.pos, batch=data_batch.batch,
                          node_atom=data_batch.z,
                          edge_d_index=data_batch.edge_d_index, edge_d_attr=data_batch.edge_d_attr)
            pred = pred.squeeze()
            if len(pred.shape)==0:
                pred=pred[None,None]
            if len(pred.shape)==1:
                pred=pred[:,None]
            
            results[f"smiles"].append(data_batch.name)
            # Store predictions and labels for each target
            for i, target in enumerate(target_names):
                pred[:, i] = pred[:, i] * task_std[i] + task_mean[i]  # De-normalize prediction
                results[f"pred_{target}"].append(pred[:,i].cpu().numpy())
                results[f"actual_{target}"].append(data_batch.y[:,i].cpu().numpy())
                results[f"diff_{target}"].append(pred[:,i].cpu().numpy() - data_batch.y[:,i].cpu().numpy())
            # Print shapes of last items added for first target
            #target = target_names[0]
            # print(f"pred_{target}:", results[f"pred_{target}"][-1].shape)
            # print(f"label_{target}:", results[f"label_{target}"][-1].shape)
            # print(f"diff_{target}:", results[f"diff_{target}"][-1].shape)
            # Collect predictions, labels, and SMILES

    # Convert lists to numpy arrays
    results["smiles"] = np.concatenate(results["smiles"])
    for i, target in enumerate(target_names):
        results[f"pred_{target}"] = np.concatenate(results[f"pred_{target}"])
        results[f"actual_{target}"] = np.concatenate(results[f"actual_{target}"])
        results[f"diff_{target}"] = np.concatenate(results[f"diff_{target}"])

    # Calculate differences
    # Create a DataFrame and save to CSV
    results_df = pd.DataFrame.from_dict(results)
    results_df.to_csv(os.path.join(args.output_dir, 'test_results.csv'), index=False)