import torch
import torch.nn as nn
from .nnutils import create_var, index_select_ND
from .chemutils import get_mol
import rdkit.Chem as Chem

ELEM_LIST = [
    'C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg',
    'Na', 'Ca', 'Fe', 'Al', 'I', 'B', 'K', 'Se', 'Zn',
    'H', 'Cu', 'Mn', 'unknown'
]

ATOM_FDIM = len(ELEM_LIST) + 6 + 5 + 1
BOND_FDIM = 5
MAX_NB = 10


def onek_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]


def atom_features(atom):
    return torch.tensor(
        list(onek_encoding_unk(atom.GetSymbol(), ELEM_LIST)) +
        list(onek_encoding_unk(atom.GetDegree(), [0, 1, 2, 3, 4, 5])) +
        list(onek_encoding_unk(atom.GetFormalCharge(), [-1, -2, 1, 2, 0])) +
        [atom.GetIsAromatic()],
        dtype=torch.float
    )


def bond_features(bond):
    bt = bond.GetBondType()
    return torch.tensor([
        bt == Chem.rdchem.BondType.SINGLE,
        bt == Chem.rdchem.BondType.DOUBLE,
        bt == Chem.rdchem.BondType.TRIPLE,
        bt == Chem.rdchem.BondType.AROMATIC,
        bond.IsInRing()
    ], dtype=torch.float)


class JTMPN(nn.Module):

    def __init__(self, hidden_size, depth):
        super().__init__()
        self.hidden_size = hidden_size
        self.depth = depth

        self.W_i = nn.Linear(ATOM_FDIM + BOND_FDIM, hidden_size, bias=False)
        self.W_h = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_o = nn.Linear(ATOM_FDIM + hidden_size, hidden_size)

    def forward(self, cand_batch, tree_mess):

        fatoms, fbonds = [], []
        in_bonds, all_bonds = [], []

        mess_dict = {}
        all_mess = [create_var(torch.zeros(self.hidden_size))]

        total_atoms = 0
        scope = []

        # tree messages
        for e, vec in tree_mess.items():
            mess_dict[e] = len(all_mess)
            all_mess.append(vec)

        for mol, all_nodes, ctr_node in cand_batch:

            n_atoms = mol.GetNumAtoms()
            atom_offset = len(fatoms)

            for _ in range(n_atoms):
                in_bonds.append([])

            # atom features
            for atom in mol.GetAtoms():
                fatoms.append(atom_features(atom))

            # bonds
            for bond in mol.GetBonds():

                a1 = bond.GetBeginAtom()
                a2 = bond.GetEndAtom()

                x = a1.GetIdx() + atom_offset
                y = a2.GetIdx() + atom_offset

                x_nid = a1.GetAtomMapNum()
                y_nid = a2.GetAtomMapNum()

                x_bid = all_nodes[x_nid - 1].idx if x_nid > 0 else -1
                y_bid = all_nodes[y_nid - 1].idx if y_nid > 0 else -1

                bfeature = bond_features(bond)

                b = len(all_mess) + len(all_bonds)
                all_bonds.append((x, y))
                fbonds.append(torch.cat([fatoms[x], bfeature], 0))
                in_bonds[y].append(b)

                b = len(all_mess) + len(all_bonds)
                all_bonds.append((y, x))
                fbonds.append(torch.cat([fatoms[y], bfeature], 0))
                in_bonds[x].append(b)

                if x_bid >= 0 and y_bid >= 0 and x_bid != y_bid:
                    if (x_bid, y_bid) in mess_dict:
                        in_bonds[y].append(mess_dict[(x_bid, y_bid)])
                    if (y_bid, x_bid) in mess_dict:
                        in_bonds[x].append(mess_dict[(y_bid, x_bid)])

            scope.append((total_atoms, n_atoms))
            total_atoms += n_atoms

        total_bonds = len(all_bonds)

        fatoms = torch.stack(fatoms, 0) if len(fatoms) > 0 else torch.zeros(0, ATOM_FDIM)
        fbonds = torch.stack(fbonds, 0) if len(fbonds) > 0 else torch.zeros(0, ATOM_FDIM + BOND_FDIM)

        agraph = torch.zeros(total_atoms, MAX_NB, dtype=torch.long)
        bgraph = torch.zeros(total_bonds, MAX_NB, dtype=torch.long)

        tree_message = torch.stack(all_mess, dim=0)

        for a in range(total_atoms):
            for i, b in enumerate(in_bonds[a]):
                if i < MAX_NB:
                    agraph[a, i] = b

        for b1 in range(total_bonds):
            x, y = all_bonds[b1]
            for i, b2 in enumerate(in_bonds[x]):
                if i < MAX_NB:
                    if b2 < len(all_mess) or all_bonds[b2 - len(all_mess)][0] != y:
                        bgraph[b1, i] = b2

        fatoms = create_var(fatoms)
        fbonds = create_var(fbonds)
        agraph = create_var(agraph)
        bgraph = create_var(bgraph)

        binput = self.W_i(fbonds)
        graph_message = torch.relu(binput)

        for _ in range(self.depth - 1):
            message = torch.cat([tree_message, graph_message], dim=0)
            nei_message = index_select_ND(message, 0, bgraph)
            nei_message = nei_message.sum(dim=1)
            nei_message = self.W_h(nei_message)
            graph_message = torch.relu(binput + nei_message)

        message = torch.cat([tree_message, graph_message], dim=0)
        nei_message = index_select_ND(message, 0, agraph)
        nei_message = nei_message.sum(dim=1)

        ainput = torch.cat([fatoms, nei_message], dim=1)
        atom_hiddens = torch.relu(self.W_o(ainput))

        mol_vecs = []
        for st, le in scope:
            mol_vecs.append(atom_hiddens.narrow(0, st, le).mean(dim=0))

        if len(mol_vecs) == 0:
            return torch.zeros(0, self.hidden_size)

        return torch.stack(mol_vecs, dim=0)