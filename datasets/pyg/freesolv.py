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
from torch_geometric.data import (InMemoryDataset, download_url, extract_tar,
                                  Data)
from torch_geometric.nn import radius_graph





URLS = {
    "precise3d": "https://drive.google.com/uc?export=download&id=1FRn6e_a-BVlfD8-rnemviHeeTBDUnC6P",
    "optimized3d": "https://drive.google.com/uc?export=download&id=1Fa39l9QUvJzuNi3fD_so92YdGzPXbNi5",
    "rdkit3d": "https://drive.google.com/uc?export=download&id=1XzATIcN_3Qtjzxqp-r4bKX1bqku4jpq9",
    "rdkit2d": "https://drive.google.com/uc?export=download&id=1rnxECBjASyIRuD9KfbL2YOuyTJeyjpzL"
}

freesolv_taget_dict = {
    "expt" : 0,
}

class Freesolv(InMemoryDataset):
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
            return [name for name, idx in freesolv_taget_dict.items() 
                   if idx in self.labels]
        return list(freesolv_taget_dict.keys())
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
            [freesolv_taget_dict[label] for label in dataset_args]
            if dataset_args is not None
            else list(freesolv_taget_dict.keys())
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
    def mean(self, target: int) -> float:
        y = torch.cat([self.get(i).y for i in range(len(self))], dim=0)
        return float(y[:, target].mean())


    def std(self, target: int) -> float:
        y = torch.cat([self.get(i).y for i in range(len(self))], dim=0)
        return float(y[:, target].std())



    @property
    def raw_file_names(self) -> List[str]:
        try:
            import rdkit  # noqa
            return ['FreeSolv.sdf', 'FreeSolv.sdf.csv']
        except ImportError:
            return ImportError("Please install 'rdkit' to download the dataset.")


    @property
    def processed_file_names(self) -> str:
        return "data_v3.pt"


    def download(self):
        try:
            import rdkit  # noqa

            file_path = download_url(self.raw_url, self.raw_dir)
            extract_tar(file_path, self.raw_dir)
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

        
        types = {'C': 0, 'N': 1, 'O': 2, 'H': 3, 'S': 4, 'Cl': 5, 'Br': 6, 'P': 7, 'F': 8, 'I': 9}
        bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}

        with open(self.raw_paths[1], 'r') as f:
            target = [[float(x)
                       for x in line.split(',')]
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