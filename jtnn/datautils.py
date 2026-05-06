from torch.utils.data import Dataset
from .mol_tree import MolTree
import numpy as np


class MoleculeDataset(Dataset):

    def __init__(self, data_file):
        with open(data_file, "r") as f:
            self.data = [
                line.strip().split()[0]
                for line in f
                if line.strip()
            ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        smiles = self.data[idx]

        mol_tree = MolTree(smiles)

        # Safety: RDKit parsing can fail silently in pipelines
        mol_tree.recover()
        mol_tree.assemble()

        return mol_tree


class PropDataset(Dataset):

    def __init__(self, data_file, prop_file):
        # FIX: ensure float dtype (np.loadtxt defaults can vary)
        self.prop_data = np.loadtxt(prop_file, dtype=np.float32)

        with open(data_file, "r") as f:
            self.data = [
                line.strip().split()[0]
                for line in f
                if line.strip()
            ]

        # Safety check: alignment bug prevention
        if len(self.data) != len(self.prop_data):
            raise ValueError(
                f"Mismatch: {len(self.data)} SMILES vs {len(self.prop_data)} properties"
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        smiles = self.data[idx]

        mol_tree = MolTree(smiles)
        mol_tree.recover()
        mol_tree.assemble()

        prop = self.prop_data[idx]

        # ensure numpy scalar → python float if single value
        if isinstance(prop, np.ndarray) and prop.shape == ():
            prop = float(prop)

        return mol_tree, prop