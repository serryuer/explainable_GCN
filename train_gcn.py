import os
import sys
import json
import torch
import torch.nn.functional as F
import pickle as pkl
import numpy as np
import logging
import argparse
from models.GCNNet import GCNNet

# log format
C_LogFormat = '%(asctime)s - %(levelname)s - %(message)s'
# setting log format
logging.basicConfig(level=logging.INFO, format=C_LogFormat)

node_size_map = {
    'mr_id': 29426,
    'mr_onehot': 29426
}

# Training settings
parser = argparse.ArgumentParser(
    description="text gcn with pytorch + torch_geometric")
parser.add_argument('-dataset_name', type=str,
                    default='mr', help='data sets name')
parser.add_argument('-lr', type=float, default=0.01,
                    help='initial learning rate [default: 0.001]')
parser.add_argument('-dropout', type=float, default=0.5,
                    help='the probability for dropout [default: 0.5]')
parser.add_argument('-epochs', type=int, default=16,
                    help='number of epochs for train [default: 20]')
parser.add_argument('-gcn_layers', type=int, default=4,
                    help='the network layer [default 2]')
parser.add_argument('-embed_dim', type=int, default=1024,
                    help='the size of node embedding')
parser.add_argument('-embed_fintune', type=bool, default=False,
                    help='whether finetune embedding layer')
parser.add_argument('-hidden_size', type=int, default=512,
                    help='number of hidden size for one rnn cell [default: 512]')
parser.add_argument('-node_size', type=int, default=29426,
                    help='number of embedding dimension [default: 128]')
parser.add_argument('-random_seed', type=int,
                    default=1024, help='attention size')
parser.add_argument('-device', type=int, default=0,
                    help='device to use for iterate data, -1 mean cpu,1 mean gpu [default: -1]')
parser.add_argument('-save_dir', type=str,
                    default='experiments', help='where to save the snapshot')
parser.add_argument('-model_name', type=str,
                    default='mr-gcn', help='model name')
parser.add_argument('-early_stop_patience', type=int,
                    default=30, help='early stop patience')
parser.add_argument('-output_size', type=int, default=2,
                    help='number of classification [default: 2]')
parser.add_argument('-save_best_model', type=bool,
                    default=False, help='whether save best model')

args = parser.parse_args()

torch.manual_seed(args.random_seed)
torch.cuda.manual_seed(args.random_seed)
np.random.seed(args.random_seed)

data = pkl.load(open(
    f"data/{args.dataset_name}/graph/ind.{args.dataset_name}_id", 'rb'), encoding='latin1')

device = torch.device(
    f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
model = GCNNet(node_size=args.node_size,
               embed_dim=args.embed_dim,
               embedding_finetune=args.embed_fintune,
               hidden_dim=args.hidden_size,
               num_class=args.output_size,
               dropout=args.dropout,
               layers=args.gcn_layers).to(device)
data = data.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)


def train():
    model.train()
    optimizer.zero_grad()
    loss = F.nll_loss(model(data.x, data.edge_index, data.edge_attr)[
               data.train_mask], data.y[data.train_mask])
    loss.backward()
    print(loss)
    optimizer.step()


def test():
    model.eval()
    logits, accs = model(data.x, data.edge_index, data.edge_attr), []
    for _, mask in data('train_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs


def save_model():
    torch.save(model.state_dict(), os.path.join(
        args.save_dir, args.model_name + '.pt'))


best_test_acc, best_epoch = 0.0, 0
patience = 0
for epoch in range(1, args.epochs):
    train()
    train_acc, test_acc = test()
    print(f"Epoch: {epoch}, Train: {train_acc}, Test: {test_acc}")
    if test_acc > best_test_acc:
        print('update best model')
        if args.save_best_model:
            save_model()
        best_test_acc = test_acc
        best_epoch = epoch
        patience = 0
    else:
        patience += 1
        if patience > args.early_stop_patience:
            print(
                f'early stop, best test acc is: {best_test_acc}, best epoch is {best_epoch}')
            exit(0)
