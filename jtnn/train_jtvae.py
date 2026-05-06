import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

import math
import argparse
import numpy as np
import os
import traceback


from torch.utils.data import DataLoader

from jtnn import Vocab, JTNNVAE
from jtnn.simple_dataset import SmilesDataset

import rdkit
import rdkit.RDLogger as RDLogger

# ----------------------------
# Silence RDKit
# ----------------------------
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)


# ----------------------------
# Args
# ----------------------------
parser = argparse.ArgumentParser()

parser.add_argument('--train', required=True)
parser.add_argument('--vocab', required=True)
parser.add_argument('--save_dir', required=True)
parser.add_argument('--load_epoch', type=int, default=0)

parser.add_argument('--hidden_size', type=int, default=450)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--latent_size', type=int, default=56)
parser.add_argument('--depthT', type=int, default=20)
parser.add_argument('--depthG', type=int, default=3)

parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--clip_norm', type=float, default=50.0)

parser.add_argument('--beta', type=float, default=0.0)
parser.add_argument('--step_beta', type=float, default=0.002)
parser.add_argument('--max_beta', type=float, default=1.0)

parser.add_argument('--warmup', type=int, default=40000)
parser.add_argument('--epoch', type=int, default=20)

parser.add_argument('--anneal_rate', type=float, default=0.9)
parser.add_argument('--anneal_iter', type=int, default=40000)
parser.add_argument('--kl_anneal_iter', type=int, default=2000)

parser.add_argument('--print_iter', type=int, default=50)
parser.add_argument('--save_iter', type=int, default=5000)

args = parser.parse_args()
print(args)


# ----------------------------
# Setup
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(args.save_dir, exist_ok=True)


# ----------------------------
# Vocab
# ----------------------------
vocab_list = [x.strip() for x in open(args.vocab)]
vocab = Vocab(vocab_list)


# ----------------------------
# Model
# ----------------------------
model = JTNNVAE(
    vocab,
    args.hidden_size,
    args.latent_size,
    args.depthT,
    args.depthG
).to(device)


# init weights
for p in model.parameters():
    if p.dim() == 1:
        nn.init.constant_(p, 0)
    else:
        nn.init.xavier_normal_(p)


# load checkpoint
if args.load_epoch > 0:
    path = f"{args.save_dir}/model.iter-{args.load_epoch}"
    model.load_state_dict(torch.load(path, map_location=device))


print(f"Model params: {sum(p.numel() for p in model.parameters()) // 1000}K")


# ----------------------------
# Optimizer
# ----------------------------
optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=args.anneal_rate)


# ----------------------------
# Helpers
# ----------------------------
def param_norm(m):
    return math.sqrt(sum(p.norm().item() ** 2 for p in m.parameters()))

def grad_norm(m):
    return math.sqrt(sum(p.grad.norm().item() ** 2 for p in m.parameters() if p.grad is not None))


# ----------------------------
# Training state
# ----------------------------
total_step = args.load_epoch
beta = args.beta

meters = np.zeros(4)
meter_count = 0


# ----------------------------
# Training loop
# ----------------------------
for epoch in range(args.epoch):

    dataset = SmilesDataset(args.train)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda x: list(x),
        num_workers=0
    )

    for batch in loader:

        total_step += 1

        try:
            model.zero_grad()

            mol_batch = [x[0] for x in batch]

            loss, kl_div, wacc, tacc, sacc = model(mol_batch, beta)

            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)

            optimizer.step()

        except Exception as e:
            print("SMILES failed:", batch)
            traceback.print_exc()
            break


        # ----------------------------
        # metrics
        # ----------------------------
        meters += np.array([
            kl_div,
            wacc * 100,
            tacc * 100,
            sacc * 100
        ])
        meter_count += 1


        # ----------------------------
        # print
        # ----------------------------
        if total_step % args.print_iter == 0 and meter_count > 0:

            meters /= meter_count
            meter_count = 0

            print(
                f"[{total_step}] "
                f"Beta: {beta:.3f}, KL: {meters[0]:.2f}, "
                f"Word: {meters[1]:.2f}, Topo: {meters[2]:.2f}, "
                f"Assm: {meters[3]:.2f}, "
                f"PNorm: {param_norm(model):.2f}, "
                f"GNorm: {grad_norm(model):.2f}"
            )

            meters[:] = 0


        # ----------------------------
        # save
        # ----------------------------
        if total_step % args.save_iter == 0:
            torch.save(
                model.state_dict(),
                f"{args.save_dir}/model.iter-{total_step}"
            )


        # ----------------------------
        # LR schedule
        # ----------------------------
        if total_step % args.anneal_iter == 0:
            scheduler.step()
            print("lr:", scheduler.get_last_lr()[0])


        # ----------------------------
        # beta schedule
        # ----------------------------
        if total_step % args.kl_anneal_iter == 0 and total_step >= args.warmup:
            beta = min(args.max_beta, beta + args.step_beta)