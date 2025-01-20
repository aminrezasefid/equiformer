from typing import Optional, Callable, List

import sys
import os
import os.path as osp
from tqdm import tqdm
import numpy as np
from typing import Callable, List, Optional, Dict
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    MutableSequence,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)
import torch
import torch.nn.functional as F
from torch_scatter import scatter

from torch_geometric.data.data import BaseData
from torch_geometric.data import (InMemoryDataset, download_url, extract_zip,
                                  Data)
from torch_geometric.nn import radius_graph





URLS = {
    "precise3d": "https://drive.google.com/uc?export=download&id=1aGf10_zoUVtn4c-oczofcgM_nTZQnYnP",
    "optimized3d": "https://drive.google.com/uc?export=download&id=1A9rky0CJAc1xHqVRTbmwvGHPqiOKM7w8",
    "rdkit3d": "https://drive.google.com/uc?export=download&id=1n80aP2vXOKYBB4yaMcqi71gcXBAQGUhY", # Replaced Armin's
    "rdkit2d": "https://drive.google.com/uc?export=download&id=1Yzn6KSeDhxkqZVYHmiClr35r1vO-ShTx"
}

tox21_target_dict = {'NR-AR': 0, 'NR-AR-LBD': 1, 'NR-AhR': 2, 'NR-Aromatase': 3, 'NR-ER': 4, 'NR-ER-LBD': 5, 'NR-PPAR-gamma': 6, 'SR-ARE': 7, 'SR-ATAD5': 8, 'SR-HSE': 9, 'SR-MMP': 10, 'SR-p53': 11}


class Tox21(InMemoryDataset):
    """
    1. This is the QM9 dataset, adapted from Pytorch Geometric to incorporate 
    cormorant data split. (Reference: Geometric and Physical Quantities improve 
    E(3) Equivariant Message Passing)
    2. Add pair-wise distance for each graph. """

    
    @property
    def target_names(self) -> List[str]:
        """Returns the names of the available target properties.
        If dataset_args is specified, returns only those target names.
        Otherwise returns all available target names.
        """
        if hasattr(self, 'labels') and self.labels is not None:
            return [name for name, idx in tox21_target_dict.items() 
                   if idx in self.labels]
        return list(tox21_target_dict.keys())
    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        force_reload: bool = False,
        structure: str = "precise3d",
        dataset_args: List[str] = None,
    ):
        self.structure = structure
        self.raw_url = URLS[structure]
        self.labels = (
            [tox21_target_dict[label] for label in dataset_args]
            if dataset_args is not None
            else list(tox21_target_dict.values())
        )
        transform = self._filter_label
        super().__init__(
            root, transform, pre_transform, pre_filter
        )
        self.data,self.slices=torch.load(self.processed_paths[0])
    
    def _filter_label(self, batch):
        if self.labels:
            batch.y = batch.y[:, self.labels]
        return batch
    # def mean(self, target: int) -> float:
    #     y = torch.cat([self.get(i).y for i in range(len(self))], dim=0)
    #     return float(y[:, target].mean())


    # def std(self, target: int) -> float:
    #     y = torch.cat([self.get(i).y for i in range(len(self))], dim=0)
    #     return float(y[:, target].std())



    @property
    def raw_file_names(self) -> List[str]:
        try:
            import rdkit  # noqa
            file_names = {
                "precise3d": ['pubchem.sdf', 'pubchem.sdf.csv'],
                "optimized3d": ['rdkit_opt.sdf', 'rdkit_opt.sdf.csv'],
                "rdkit3d": ['rdkit_3D.sdf', 'rdkit_3D.sdf.csv'],          ###### CHANGE ######
                "rdkit2d": ['rdkit_graph.sdf', 'rdkit_graph.sdf.csv']
            }
            return file_names[self.structure]
        except ImportError:
            return ImportError("Please install 'rdkit' to download the dataset.")


    @property
    def processed_file_names(self) -> str:
        return "data_v3.pt"


    def download(self):
        try:
            import rdkit  # noqa
            #import gdown
            file_path = download_url(self.raw_url, self.raw_dir)
            #gdown.download(self.raw_url, output=file_path, quiet=False)
            extract_zip(file_path, self.raw_dir)
            os.unlink(file_path)

        except ImportError:
            print("Please install 'rdkit' to download the dataset.", file=sys.stderr)


    def process(self):
        try:
            import rdkit
            from rdkit import Chem
            from rdkit.Chem.rdchem import HybridizationType
            from rdkit.Chem.rdchem import BondType as BT
            from rdkit import RDLogger
            RDLogger.DisableLog('rdApp.*')
        except ImportError:
            assert False, "Install rdkit-pypi"

        
        types = {'C': 0, 'O': 1, 'N': 2, 'S': 3, 'P': 4, 'Cl': 5, 'I': 6, 'Zn': 7, 'F': 8, 'Ca': 9, 'As': 10, 'Br': 11, 'B': 12, 'H': 13, 'K': 14, 'Si': 15, 'Cu': 16, 'Mg': 17, 'Hg': 18, 'Cr': 19, 'Zr': 20, 'Sn': 21, 'Na': 22, 'Ba': 23, 'Au': 24, 'Pd': 25, 'Tl': 26, 'Fe': 27, 'Al': 28, 'Gd': 29, 'Ag': 30, 'Mo': 31, 'V': 32, 'Nd': 33, 'Co': 34, 'Yb': 35, 'Pb': 36, 'Sb': 37, 'In': 38, 'Li': 39, 'Ni': 40, 'Bi': 41, 'Cd': 42, 'Ti': 43, 'Se': 44, 'Dy': 45, 'Mn': 46, 'Sr': 47, 'Be': 48, 'Pt': 49, 'Ge': 50}

        bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3, BT.DATIVE: 4}

        with open(self.raw_paths[1], 'r') as f:
            target = [[float(x) if x != '' else -1
                       for x in line.split(',')[:-2]]
                      for line in f.read().split('\n')[1:-1]]
            y = torch.tensor(target, dtype=torch.float)

        suppl = Chem.SDMolSupplier(self.raw_paths[0], removeHs=False,
                                   sanitize=False)
        data_list = []
        for i, mol in enumerate(tqdm(suppl)):

            N = mol.GetNumAtoms()
            conf = mol.GetConformer()
            pos = conf.GetPositions()
            pos = torch.tensor(pos, dtype=torch.float)
            if torch.unique(pos, dim=0).size(0) != N:
                # print(f"Skipping molecule {mol.GetProp('_Name')} as it contains overlapping atoms.")
                continue
            #edge_index = radius_graph(pos, r=self.radius, loop=False)
            
            # build pair-wise edge graphs
            num_nodes = pos.shape[0]
            node_index = torch.tensor([i for i in range(num_nodes)])
            edge_d_dst_index = torch.repeat_interleave(node_index, repeats=num_nodes)
            edge_d_src_index = node_index.repeat(num_nodes)
            edge_d_attr = pos[edge_d_dst_index] - pos[edge_d_src_index]
            edge_d_attr = edge_d_attr.norm(dim=1, p=2)
            edge_d_dst_index = edge_d_dst_index.view(1, -1)
            edge_d_src_index = edge_d_src_index.view(1, -1)
            edge_d_index = torch.cat((edge_d_dst_index, edge_d_src_index), dim=0)
            
            type_idx = []
            atomic_number = []
            aromatic = []
            sp = []
            sp2 = []
            sp3 = []
            num_hs = []
            for atom in mol.GetAtoms():
                type_idx.append(types[atom.GetSymbol()])
                atomic_number.append(atom.GetAtomicNum())
                aromatic.append(1 if atom.GetIsAromatic() else 0)
                hybridization = atom.GetHybridization()
                sp.append(1 if hybridization == HybridizationType.SP else 0)
                sp2.append(1 if hybridization == HybridizationType.SP2 else 0)
                sp3.append(1 if hybridization == HybridizationType.SP3 else 0)

            z = torch.tensor(atomic_number, dtype=torch.long)

            # from torch geometric
            row, col, edge_type = [], [], []
            for bond in mol.GetBonds():
                start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                row += [start, end]
                col += [end, start]
                edge_type += 2 * [bonds[bond.GetBondType()]]

            edge_index = torch.tensor([row, col], dtype=torch.long)
            edge_type = torch.tensor(edge_type, dtype=torch.long)
            edge_attr = F.one_hot(edge_type,
                                  num_classes=len(bonds)).to(torch.float)

            perm = (edge_index[0] * N + edge_index[1]).argsort()
            edge_index = edge_index[:, perm]
            edge_type = edge_type[perm]
            edge_attr = edge_attr[perm]

            row, col = edge_index
            hs = (z == 1).to(torch.float)
            num_hs = scatter(hs[row], col, dim_size=N, reduce="sum").tolist()

            x1 = F.one_hot(torch.tensor(type_idx), num_classes=len(types))
            x2 = (
                torch.tensor(
                    [atomic_number, aromatic, sp, sp2, sp3, num_hs], dtype=torch.float
                )
                .t()
                .contiguous()
            )
            x = torch.cat([x1, x2], dim=-1)

            name = mol.GetProp('_Name')
            

            smiles = Chem.MolToSmiles(mol, isomericSmiles=True)      

            data = Data(x=x, pos=pos, z=z, edge_index=edge_index, 
                edge_attr=edge_attr, name=name, index=i, 
                smiles=smiles,
                y=y[i].unsqueeze(0),
                edge_d_index=edge_d_index, edge_d_attr=edge_d_attr)
            data_list.append(data)
            
        torch.save(self.collate(data_list), self.processed_paths[0])

    
def get_cormorant_features(one_hot, charges, charge_power, charge_scale):
    """ Create input features as described in section 7.3 of https://arxiv.org/pdf/1906.04015.pdf """
    charge_tensor = (charges.unsqueeze(-1) / charge_scale).pow(
        torch.arange(charge_power + 1., dtype=torch.float32))
    charge_tensor = charge_tensor.view(charges.shape + (1, charge_power + 1))
    atom_scalars = (one_hot.unsqueeze(-1) * charge_tensor).view(charges.shape[:2] + (-1,))
    return atom_scalars


if __name__ == "__main__":
    
    from torch_geometric.loader import DataLoader
    import matplotlib.pyplot as plt
    
    #dataset = QM9("temp", "valid", feature_type="one_hot")
    #print("length", len(dataset))
    #dataloader = DataLoader(dataset, batch_size=4)
    
    '''
    _target = 1
    
    dataset = QM9("test_atom_ref/with_atomrefs", "test", feature_type="one_hot", update_atomrefs=True)
    mean = dataset.mean(_target)
    _, std = dataset.calc_stats(_target)
    
    dataset_original = QM9("test_atom_ref/without_atomrefs", "test", feature_type="one_hot", update_atomrefs=False)
    
    for i in range(12):
        mean = dataset.mean(i)
        std = dataset.std(i)
        
        mean_original = dataset_original.mean(i)
        std_original = dataset_original.std(i)
        
        print('Target: {}, mean diff = {}, std diff = {}'.format(i, 
            mean - mean_original, std - std_original))
    '''

    #dataset = QM7("test_torchmd_net_splits", "train", feature_type="one_hot", update_atomrefs=True, torchmd_net_split=True)