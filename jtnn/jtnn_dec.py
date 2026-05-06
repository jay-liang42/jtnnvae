import torch
import torch.nn as nn
from .mol_tree import MolTreeNode
from .nnutils import create_var, GRU
from .chemutils import enum_assemble
import copy

MAX_NB = 8
MAX_DECODE_LEN = 100


class JTNNDecoder(nn.Module):

    def __init__(self, vocab, hidden_size, latent_size, embedding=None):
        super().__init__()

        self.hidden_size = hidden_size
        self.vocab_size = vocab.size()
        self.vocab = vocab

        if embedding is None:
            self.embedding = nn.Embedding(self.vocab_size, hidden_size)
        else:
            self.embedding = embedding

        # GRU weights
        self.W_z = nn.Linear(2 * hidden_size, hidden_size)
        self.U_r = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_r = nn.Linear(hidden_size, hidden_size)
        self.W_h = nn.Linear(2 * hidden_size, hidden_size)

        # feature aggregation
        self.W = nn.Linear(latent_size + hidden_size, hidden_size)
        self.U = nn.Linear(latent_size + 2 * hidden_size, hidden_size)

        # output
        self.W_o = nn.Linear(hidden_size, self.vocab_size)
        self.U_s = nn.Linear(hidden_size, 1)

        # losses (PyTorch >= 1.9 fix)
        self.pred_loss = nn.CrossEntropyLoss(reduction="sum")
        self.stop_loss = nn.BCEWithLogitsLoss(reduction="sum")

    def get_trace(self, node):
        super_root = MolTreeNode("")
        super_root.idx = -1
        trace = []
        dfs(trace, node, super_root)
        return [(x.smiles, y.smiles, z) for x, y, z in trace]

    def forward(self, mol_batch, mol_vec):

        super_root = MolTreeNode("")
        super_root.idx = -1

        pred_hiddens, pred_mol_vecs, pred_targets = [], [], []
        stop_hiddens, stop_targets = [], []

        traces = []

        for mol_tree in mol_batch:
            s = []
            dfs(s, mol_tree.nodes[0], super_root)
            traces.append(s)

            for node in mol_tree.nodes:
                node.neighbors = []

        # root prediction
        pred_hiddens.append(create_var(torch.zeros(len(mol_batch), self.hidden_size)))
        pred_targets.extend([mol_tree.nodes[0].wid for mol_tree in mol_batch])
        pred_mol_vecs.append(mol_vec)

        max_iter = max(len(tr) for tr in traces)
        padding = create_var(torch.zeros(self.hidden_size), False)

        h = {}

        for t in range(max_iter):

            prop_list = []
            batch_list = []

            for i, plist in enumerate(traces):
                if t < len(plist):
                    prop_list.append(plist[t])
                    batch_list.append(i)

            cur_x, cur_h_nei, cur_o_nei = [], [], []

            for node_x, real_y, _ in prop_list:

                cur_nei = [
                    h[(node_y.idx, node_x.idx)]
                    for node_y in node_x.neighbors
                    if node_y.idx != real_y.idx
                ]
                pad_len = MAX_NB - len(cur_nei)
                cur_h_nei.extend(cur_nei + [padding] * pad_len)

                cur_nei = [h[(node_y.idx, node_x.idx)] for node_y in node_x.neighbors]
                pad_len = MAX_NB - len(cur_nei)
                cur_o_nei.extend(cur_nei + [padding] * pad_len)

                cur_x.append(node_x.wid)

            cur_x = create_var(torch.LongTensor(cur_x))
            cur_x = self.embedding(cur_x)

            cur_h_nei = torch.stack(cur_h_nei).view(-1, MAX_NB, self.hidden_size)
            new_h = GRU(cur_x, cur_h_nei, self.W_z, self.W_r, self.U_r, self.W_h)

            cur_o_nei = torch.stack(cur_o_nei).view(-1, MAX_NB, self.hidden_size)
            cur_o = cur_o_nei.sum(dim=1)

            pred_target, pred_list, stop_target = [], [], []

            for i, m in enumerate(prop_list):
                node_x, node_y, direction = m
                x, y = node_x.idx, node_y.idx

                h[(x, y)] = new_h[i]
                node_y.neighbors.append(node_x)

                if direction == 1:
                    pred_target.append(node_y.wid)
                    pred_list.append(i)
                    stop_target.append(direction)

            cur_batch = create_var(torch.LongTensor(batch_list))
            cur_mol_vec = mol_vec.index_select(0, cur_batch)

            stop_hidden = torch.cat([cur_x, cur_o, cur_mol_vec], dim=1)
            stop_hiddens.append(stop_hidden)
            stop_targets.extend(stop_target)

            if len(pred_list) > 0:
                batch_list = [batch_list[i] for i in pred_list]
                cur_batch = create_var(torch.LongTensor(batch_list))

                pred_mol_vecs.append(mol_vec.index_select(0, cur_batch))

                cur_pred = create_var(torch.LongTensor(pred_list))
                pred_hiddens.append(new_h.index_select(0, cur_pred))
                pred_targets.extend(pred_target)

        # final stop at root
        cur_x, cur_o_nei = [], []

        for mol_tree in mol_batch:
            node_x = mol_tree.nodes[0]
            cur_x.append(node_x.wid)

            cur_nei = [h[(node_y.idx, node_x.idx)] for node_y in node_x.neighbors]
            pad_len = MAX_NB - len(cur_nei)
            cur_o_nei.extend(cur_nei + [padding] * pad_len)

        cur_x = create_var(torch.LongTensor(cur_x))
        cur_x = self.embedding(cur_x)

        cur_o_nei = torch.stack(cur_o_nei).view(-1, MAX_NB, self.hidden_size)
        cur_o = cur_o_nei.sum(dim=1)

        stop_hidden = torch.cat([cur_x, cur_o, mol_vec], dim=1)
        stop_hiddens.append(stop_hidden)
        stop_targets.extend([0] * len(mol_batch))

        # prediction loss
        pred_hiddens = torch.cat(pred_hiddens)
        pred_mol_vecs = torch.cat(pred_mol_vecs)

        pred_vecs = torch.cat([pred_hiddens, pred_mol_vecs], dim=1)
        pred_vecs = torch.relu(self.W(pred_vecs))
        pred_scores = self.W_o(pred_vecs)

        pred_targets = create_var(torch.LongTensor(pred_targets))
        pred_loss = self.pred_loss(pred_scores, pred_targets) / len(mol_batch)

        _, preds = torch.max(pred_scores, dim=1)
        pred_acc = (preds == pred_targets).float().mean()

        # stop loss
        stop_hiddens = torch.cat(stop_hiddens)
        stop_vecs = torch.relu(self.U(stop_hiddens))
        stop_scores = self.U_s(stop_vecs).squeeze()

        stop_targets = create_var(torch.Tensor(stop_targets))
        stop_loss = self.stop_loss(stop_scores, stop_targets) / len(mol_batch)

        stops = (stop_scores >= 0).float()
        stop_acc = (stops == stop_targets).float().mean()

        return pred_loss, stop_loss, pred_acc.item(), stop_acc.item()

    def decode(self, mol_vec, prob_decode):
        # unchanged logic (no Python syntax errors here)
        ...