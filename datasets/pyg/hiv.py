from typing import Optional, Callable, List

import sys
import os
import os.path as osp
from tqdm import tqdm
import numpy as np
import gdown
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
import errno
from torch_geometric.data.data import BaseData
from torch_geometric.data import (InMemoryDataset, download_url, extract_zip, extract_gz,extract_tar,
                                  Data)
from torch_geometric.nn import radius_graph
def makedirs(path):
    try:
        os.makedirs(osp.expanduser(osp.normpath(path)))
    except OSError as e:
        if e.errno != errno.EEXIST and osp.isdir(path):
            raise e

def gdown_download_url(id: str, folder: str, log: bool = True):
    filename = id + ".zip"
    path = osp.join(folder, filename)

    if osp.exists(path):  # pragma: no cover
        if log:
            print(f"Using existing file {filename}", file=sys.stderr)
        return path

    if log:
        print(f"Downloading {id}", file=sys.stderr)

    makedirs(folder)

    data = gdown.download(id=id, output=path)

    return path
URLS = {
    "precise3d": "https://drive.google.com/uc?export=download&id=1NolfTVAKGEJAxwBpLwkV2Ax6EmJiSnpc",
    "optimized3d": "https://drive.google.com/uc?export=download&id=1G9FuFL4yzfcnQ-x99WZPy31C6QwMjSQJ",
    "rdkit3d": "https://drive.google.com/uc?export=download&id=1ifQmTROUBvIMt0UOiXU2QwtRk8PpR_25",
    "rdkit2d": "https://drive.google.com/uc?export=download&id=1CmDIsgf58UzHnLTEHJUrbUZwMG6KDJoL",
}

hiv_target_dict = {"HIV_active": 0}

class HIV(InMemoryDataset):
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
            return [name for name, idx in hiv_target_dict.items() 
                   if idx in self.labels]
        return list(hiv_target_dict.keys())
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
            [hiv_target_dict[label] for label in dataset_args]
            if dataset_args is not None
            else list(hiv_target_dict.values())
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
                "precise3d": ["HIV_exp.sdf", "HIV_exp.sdf.csv"],
                "optimized3d": ["HIV_opt.sdf", "HIV_opt.sdf.csv"],
                "rdkit3d": ["HIV.sdf", "HIV.sdf.csv"],
                "rdkit2d": ["HIV_graph.sdf", "HIV_graph.sdf.csv"],
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

            # import gdown
            file_path = gdown_download_url(self.raw_url.split("id=")[1], self.raw_dir)
            # gdown.download(self.raw_url, output=file_path, quiet=False)
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

        
        types = {
            "C": 0,
            "O": 1,
            "Cu": 2,
            "N": 3,
            "S": 4,
            "P": 5,
            "Cl": 6,
            "Zn": 7,
            "B": 8,
            "Br": 9,
            "Co": 10,
            "Mn": 11,
            "As": 12,
            "Al": 13,
            "Ni": 14,
            "Se": 15,
            "Si": 16,
            "V": 17,
            "Zr": 18,
            "Sn": 19,
            "I": 20,
            "F": 21,
            "Li": 22,
            "Sb": 23,
            "Fe": 24,
            "Pd": 25,
            "Hg": 26,
            "Bi": 27,
            "Na": 28,
            "Ca": 29,
            "Ti": 30,
            "H": 31,
            "Ho": 32,
            "Ge": 33,
            "Pt": 34,
            "Ru": 35,
            "Rh": 36,
            "Cr": 37,
            "Ga": 38,
            "K": 39,
            "Ag": 40,
            "Au": 41,
            "Tb": 42,
            "Ir": 43,
            "Te": 44,
            "Mg": 45,
            "Pb": 46,
            "W": 47,
            "Cs": 48,
            "Mo": 49,
            "Re": 50,
            "U": 51,
            "Gd": 52,
            "Tl": 53,
            "Ac": 54,
        }
        bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3, BT.DATIVE: 4}
        with open(self.raw_paths[1], "r") as f:
            target = [
                [
                    float(x) if x != "-100" and x != "" else -1
                    for x in line.split(",")[2]
                ]
                for line in f.read().split("\n")[1:-1]
            ]
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
            
            if self.structure != "precise3d":
                try:
                    name = Chem.MolToSmiles(mol, isomericSmiles=False)
                    mol.UpdatePropertyCache()
                except:
                    continue

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