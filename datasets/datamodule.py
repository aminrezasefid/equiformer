from tqdm import tqdm
import torch
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
from rdkit.Chem.Scaffolds import MurckoScaffold
import datasets 
def generate_scaffold(smiles, include_chirality=False):
    """
    Obtain Bemis-Murcko scaffold from smiles

    Args:
        smiles: smiles sequence
        include_chirality: Default=False
    
    Return: 
        the scaffold of the given smiles.
    """
    try:
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(
            smiles=smiles, includeChirality=include_chirality
        )
    except Exception as e:
        print("asd",e)
        return None
    return scaffold

def scaffold_split( dataset, 
            frac_train=None, 
            frac_val=None, 
            frac_test=None):
    N = len(dataset)

    # create dict of the form {scaffold_i: [idx1, idx....]}
    all_scaffolds = {}
    failed_scaffolds = []
    for i in range(N):
        datapoint=dataset[i]
        smi=datapoint['name']
        scaffold = generate_scaffold(smi)
        if scaffold is None:
            failed_scaffolds.append(smi)
            continue
        if scaffold not in all_scaffolds:
            all_scaffolds[scaffold] = [i]
        else:
            all_scaffolds[scaffold].append(i)

    N = N - len(failed_scaffolds)
    
    # sort from largest to smallest sets
    all_scaffolds = {key: sorted(value) for key, value in all_scaffolds.items()}
    all_scaffold_sets = [
        scaffold_set for (scaffold, scaffold_set) in sorted(
            all_scaffolds.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True)
    ]

    # get train, valid test indices
    train_cutoff = frac_train * N
    valid_cutoff = (frac_train + frac_val) * N
    train_idx, valid_idx, test_idx = [], [], []
    for scaffold_set in all_scaffold_sets:
        if len(train_idx) + len(scaffold_set) > train_cutoff:
            if len(train_idx) + len(valid_idx) + len(scaffold_set) > valid_cutoff:
                test_idx.extend(scaffold_set)
            else:
                valid_idx.extend(scaffold_set)
        else:
            train_idx.extend(scaffold_set)

    assert len(set(train_idx).intersection(set(valid_idx))) == 0
    # Shouldn't this be between train and test?
    assert len(set(test_idx).intersection(set(valid_idx))) == 0

    return (
        torch.tensor(train_idx, dtype=torch.long),
        torch.tensor(valid_idx, dtype=torch.long),
        torch.tensor(test_idx, dtype=torch.long),
    )

class DataModule:
    def __init__(self, hparams):
        super(DataModule, self).__init__()
        self._hparams = hparams.__dict__ if hasattr(hparams, "__dict__") else hparams
        self._mean, self._std = None, None
        self._saved_dataloaders = dict()
    @property
    def target_idx(self):
        target_idx=self.dataset.labels
        return target_idx
    @property
    def hparams(self):
        return self._hparams
    
    def get_unique_atomic_numbers(self):
        unique_z = set()
        for data in self.dataset:
            unique_z.update(data.z.tolist())
        return sorted(list(unique_z))

    def setup(self):
        self.dataset = getattr(datasets, self.hparams["dataset"])(
                    root=self.hparams["dataset_root"] + "-" + self.hparams["structure"],
                    dataset_args=self.hparams["dataset_args"],
                    transform=None,
                    structure=self.hparams["structure"],
                )
        if self.hparams["split"] == "scaffold":
            self.idx_train, self.idx_val, self.idx_test = scaffold_split(
                dataset=self.dataset,
                frac_train=self.hparams["train_size"],
                frac_val=self.hparams["val_size"],
                frac_test=self.hparams["test_size"],
            )
        print(
            f"train {len(self.idx_train)}, val {len(self.idx_val)}, test {len(self.idx_test)}"
        )
        self.train_dataset = Subset(self.dataset, self.idx_train)
        self.val_dataset = Subset(self.dataset, self.idx_val)
        self.test_dataset = Subset(self.dataset, self.idx_test)
        if self.hparams["standardize"]:
            self._standardize()
    def train_dataloader(self):
        return self._get_dataloader(self.train_dataset, "train")
    def val_dataloader(self):
        loaders = self._get_dataloader(self.val_dataset, "val")
        return loaders
    def test_dataloader(self):
        return self._get_dataloader(self.test_dataset, "test")
    def predict_dataloader(self):
        return self._get_dataloader(self.test_dataset, "test")
    
    @property
    def mean(self):
        return self._mean

    @property
    def std(self):
        return self._std

    def _get_dataloader(self, dataset, stage):
        if stage == "train":
            batch_size = self.hparams["batch_size"]
            shuffle = True
            
            # Add balanced sampling for BBBP dataset
            unbalanced_datasets=["BBBP","Bace","Clintox","HIV"]
            if self.hparams["dataset"] in unbalanced_datasets:
                # Get labels for the subset
                labels = torch.tensor([dataset[i]['y'] for i in range(len(dataset))])
                class_sample_count = torch.bincount(labels.long())
                weights = 1. / class_sample_count.float()
                samples_weights = weights[labels.long()]
                
                sampler = torch.utils.data.WeightedRandomSampler(
                    weights=samples_weights,
                    num_samples=len(samples_weights),
                    replacement=True
                )
                shuffle = False  # Don't shuffle when using sampler
            else:
                sampler = None
        elif stage in ["val", "test"]:
            batch_size = self.hparams["inference_batch_size"]
            shuffle = False
            sampler = None

        dl = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.hparams["num_workers"],
            pin_memory=True,
            sampler=sampler,  # Add sampler parameter
        )

        return dl

    def _standardize(self):
        def get_energy(batch, atomref):
            return batch.y.clone()


        data = tqdm(
            self._get_dataloader(self.train_dataset, "train"),
            desc="computing mean and std",
        )
        atomref = None
        ys = torch.cat([get_energy(batch, atomref) for batch in data])
        self._mean = ys.mean(dim=0)
        self._std = ys.std(dim=0)