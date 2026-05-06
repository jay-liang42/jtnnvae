import torch
import torch.nn as nn
from collections import deque

from .nnutils import create_var, GRU

MAX_NB = 8


class JTNNEncoder(nn.Module):

    def __init__(self, vocab, hidden_size, embedding=None):
        super().__init__()

        self.hidden_size = hidden_size
        self.vocab_size = vocab.size()
        self.vocab = vocab

        if embedding is None:
            self.embedding = nn.Embedding(self.vocab_size, hidden_size)
        else:
            self.embedding = embedding

        self.W_z = nn.Linear(2 * hidden_size, hidden_size)
        self.W_r = nn.Linear(hidden_size, hidden_size, bias=False)
        self.U_r = nn.Linear(hidden_size, hidden_size)
        self.W_h = nn.Linear(2 * hidden_size, hidden_size)
        self.W = nn.Linear(2 * hidden_size, hidden_size)


    def forward(self, root_batch):

        orders = [get_prop_order(root) for root in root_batch]

        h = {}

        max_depth = max(len(o) for o in orders)
        padding = create_var(torch.zeros(self.hidden_size), False)

        for t in range(max_depth):

            prop_list = []
            for order in orders:
                if t < len(order):
                    prop_list.extend(order[t])

            if len(prop_list) == 0:
                continue

            cur_x = []
            cur_h_nei = []

            for node_x, node_y in prop_list:

                x, y = node_x.idx, node_y.idx
                cur_x.append(node_x.wid)

                h_nei = []
                for node_z in node_x.neighbors:
                    z = node_z.idx
                    if z == y:
                        continue
                    h_nei.append(h[(z, x)])

                # pad
                pad_len = MAX_NB - len(h_nei)
                if pad_len > 0:
                    h_nei += [padding] * pad_len
                else:
                    h_nei = h_nei[:MAX_NB]

                cur_h_nei.extend(h_nei)

            cur_x = torch.LongTensor(cur_x)
            cur_x = create_var(cur_x, False)
            cur_x = self.embedding(cur_x)

            cur_h_nei = torch.stack(cur_h_nei, dim=0)
            cur_h_nei = cur_h_nei.view(-1, MAX_NB, self.hidden_size)

            new_h = GRU(cur_x, cur_h_nei, self.W_z, self.W_r, self.U_r, self.W_h)

            for i, (node_x, node_y) in enumerate(prop_list):
                x, y = node_x.idx, node_y.idx
                h[(x, y)] = new_h[i]

        root_vecs = node_aggregate(root_batch, h, self.embedding, self.W)

        return h, root_vecs


# ----------------------------
# helpers
# ----------------------------

def get_prop_order(root):

    queue = deque([root])
    visited = set([root.idx])

    root.depth = 0

    order1, order2 = [], []

    while queue:

        x = queue.popleft()

        for y in x.neighbors:

            if y.idx not in visited:
                queue.append(y)
                visited.add(y.idx)

                y.depth = x.depth + 1

                if y.depth > len(order1):
                    order1.append([])
                    order2.append([])

                order1[y.depth - 1].append((x, y))
                order2[y.depth - 1].append((y, x))

    return order2[::-1] + order1


def node_aggregate(nodes, h, embedding, W):

    x_idx = []
    h_nei = []

    hidden_size = embedding.embedding_dim
    padding = create_var(torch.zeros(hidden_size), False)

    for node_x in nodes:

        x_idx.append(node_x.wid)

        nei = [h[(node_y.idx, node_x.idx)] for node_y in node_x.neighbors]

        pad_len = MAX_NB - len(nei)
        if pad_len > 0:
            nei += [padding] * pad_len
        else:
            nei = nei[:MAX_NB]

        h_nei.extend(nei)

    h_nei = torch.stack(h_nei, dim=0)
    h_nei = h_nei.view(-1, MAX_NB, hidden_size)

    sum_h_nei = h_nei.sum(dim=1)

    x_vec = torch.LongTensor(x_idx)
    x_vec = create_var(x_vec, False)
    x_vec = embedding(x_vec)

    node_vec = torch.cat([x_vec, sum_h_nei], dim=1)

    return nn.ReLU()(W(node_vec))