import random
from torch.utils.data import Dataset
from jtnn.mol_tree import MolTree

class SmilesDataset(Dataset):

    def __init__(self, path):
        with open(path, "r") as f:
            self.smiles = [line.strip().split()[0] for line in f if line.strip()]

        self.smiles = [s for s in self.smiles if len(s) > 0]

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        smi = self.smiles[idx]

        mol_tree = MolTree(smi)
        mol_tree.recover()

        return mol_tree, 0.0