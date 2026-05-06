import torch
import torch.nn as nn
from .mol_tree import Vocab, MolTree
from .nnutils import create_var
from .jtnn_enc import JTNNEncoder
from .jtnn_dec import JTNNDecoder
from .mpn import MPN, mol2graph
from .jtmpn import JTMPN

from .chemutils import (
    enum_assemble,
    set_atommap,
    copy_edit_mol,
    attach_mols,
    atom_equal,
    decode_stereo,
)

import rdkit
import rdkit.Chem as Chem
import copy


def set_batch_nodeID(mol_batch, vocab):
    tot = 0
    for mol_tree in mol_batch:
        for node in mol_tree.nodes:
            node.idx = tot
            node.wid = vocab.get_index(node.smiles)
            tot += 1


class JTNNVAE(nn.Module):

    def __init__(self, vocab, hidden_size, latent_size, depth, stereo=True):
        super().__init__()

        self.vocab = vocab
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.depth = depth
        self.use_stereo = stereo

        self.embedding = nn.Embedding(vocab.size(), hidden_size)

        self.jtnn = JTNNEncoder(vocab, hidden_size, self.embedding)
        self.jtmpn = JTMPN(hidden_size, depth)
        self.mpn = MPN(hidden_size, depth)
        self.decoder = JTNNDecoder(vocab, hidden_size, latent_size // 2, self.embedding)

        self.T_mean = nn.Linear(hidden_size, latent_size // 2)
        self.T_var = nn.Linear(hidden_size, latent_size // 2)
        self.G_mean = nn.Linear(hidden_size, latent_size // 2)
        self.G_var = nn.Linear(hidden_size, latent_size // 2)

        # FIX: deprecated size_average
        self.assm_loss = nn.CrossEntropyLoss(reduction="sum")

        if stereo:
            self.stereo_loss = nn.CrossEntropyLoss(reduction="sum")

    def encode(self, mol_batch):
        set_batch_nodeID(mol_batch, self.vocab)

        root_batch = [mol_tree.nodes[0] for mol_tree in mol_batch]
        tree_mess, tree_vec = self.jtnn(root_batch)

        smiles_batch = [mol_tree.smiles for mol_tree in mol_batch]
        mol_vec = self.mpn(mol2graph(smiles_batch))

        return tree_mess, tree_vec, mol_vec

    def encode_latent_mean(self, smiles_list):
        mol_batch = [MolTree(s) for s in smiles_list]
        for mol_tree in mol_batch:
            mol_tree.recover()

        _, tree_vec, mol_vec = self.encode(mol_batch)

        tree_mean = self.T_mean(tree_vec)
        mol_mean = self.G_mean(mol_vec)

        return torch.cat([tree_mean, mol_mean], dim=1)

    def forward(self, mol_batch, beta=0.0):
        batch_size = len(mol_batch)

        tree_mess, tree_vec, mol_vec = self.encode(mol_batch)

        tree_mean = self.T_mean(tree_vec)
        tree_log_var = -torch.abs(self.T_var(tree_vec))

        mol_mean = self.G_mean(mol_vec)
        mol_log_var = -torch.abs(self.G_var(mol_vec))

        z_mean = torch.cat([tree_mean, mol_mean], dim=1)
        z_log_var = torch.cat([tree_log_var, mol_log_var], dim=1)

        kl_loss = -0.5 * torch.sum(
            1 + z_log_var - z_mean.pow(2) - torch.exp(z_log_var)
        ) / batch_size

        # FIX: integer division + correct dtype
        eps_dim = self.latent_size // 2

        epsilon = create_var(torch.randn(batch_size, eps_dim), False)
        tree_vec = tree_mean + torch.exp(tree_log_var / 2) * epsilon

        epsilon = create_var(torch.randn(batch_size, eps_dim), False)
        mol_vec = mol_mean + torch.exp(mol_log_var / 2) * epsilon

        word_loss, topo_loss, word_acc, topo_acc = self.decoder(mol_batch, tree_vec)

        assm_loss, assm_acc = self.assm(mol_batch, mol_vec, tree_mess)

        stereo_loss, stereo_acc = 0.0, 0.0
        if self.use_stereo:
            stereo_loss, stereo_acc = self.stereo(mol_batch, mol_vec)

        loss = word_loss + topo_loss + assm_loss + 2 * stereo_loss + beta * kl_loss

        return loss, kl_loss.item(), word_acc, topo_acc, assm_acc, stereo_acc

    def assm(self, mol_batch, mol_vec, tree_mess):
        cands = []
        batch_idx = []

        for i, mol_tree in enumerate(mol_batch):
            for node in mol_tree.nodes:
                if node.is_leaf or len(node.cands) == 1:
                    continue
                cands.extend([(cand, mol_tree.nodes, node) for cand in node.cand_mols])
                batch_idx.extend([i] * len(node.cands))

        cand_vec = self.jtmpn(cands, tree_mess)
        cand_vec = self.G_mean(cand_vec)

        batch_idx = create_var(torch.LongTensor(batch_idx))
        mol_vec = mol_vec.index_select(0, batch_idx)

        mol_vec = mol_vec.view(-1, 1, self.latent_size // 2)
        cand_vec = cand_vec.view(-1, self.latent_size // 2, 1)

        scores = torch.bmm(mol_vec, cand_vec).squeeze()

        cnt, acc = 0, 0
        all_loss = []

        for i, mol_tree in enumerate(mol_batch):
            comp_nodes = [
                node for node in mol_tree.nodes
                if len(node.cands) > 1 and not node.is_leaf
            ]
            cnt += len(comp_nodes)

            for node in comp_nodes:
                label = node.cands.index(node.label)
                ncand = len(node.cands)

                cur_score = scores.narrow(0, 0, ncand)

                if cur_score[label].item() >= cur_score.max().item():
                    acc += 1

                label_t = create_var(torch.LongTensor([label]))
                all_loss.append(self.assm_loss(cur_score.view(1, -1), label_t))

        return sum(all_loss) / max(1, len(mol_batch)), acc / max(1, cnt)

    def stereo(self, mol_batch, mol_vec):
        stereo_cands, batch_idx, labels = [], [], []

        for i, mol_tree in enumerate(mol_batch):
            cands = mol_tree.stereo_cands
            if len(cands) == 1:
                continue

            if mol_tree.smiles3D not in cands:
                cands.append(mol_tree.smiles3D)

            stereo_cands.extend(cands)
            batch_idx.extend([i] * len(cands))
            labels.append((cands.index(mol_tree.smiles3D), len(cands)))

        if len(labels) == 0:
            return 0.0, 1.0

        batch_idx = create_var(torch.LongTensor(batch_idx))

        stereo_cands = self.mpn(mol2graph(stereo_cands))
        stereo_cands = self.G_mean(stereo_cands)

        stereo_labels = mol_vec.index_select(0, batch_idx)

        cos = nn.CosineSimilarity(dim=1)
        scores = cos(stereo_cands, stereo_labels)

        st, acc = 0, 0
        all_loss = []

        for label, le in labels:
            cur_scores = scores.narrow(0, st, le)

            if cur_scores.data[label] >= cur_scores.max().data:
                acc += 1

            label_t = create_var(torch.LongTensor([label]))
            all_loss.append(self.stereo_loss(cur_scores.view(1, -1), label_t))
            st += le

        return sum(all_loss) / max(1, len(labels)), acc / max(1, len(labels))

    def decode(self, tree_vec, mol_vec, prob_decode):
        return self.decoder.decode(tree_vec, prob_decode)