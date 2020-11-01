
import os, sys, json
import numpy as np
from collections import defaultdict
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys
import re
from tqdm import tqdm
from itertools import repeat
from torch_geometric.io import read_txt_array
from torch_geometric.utils import remove_self_loops
from torch_sparse import coalesce
from torch_geometric.data import Data
import torch
    
def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)
        
def load_data(dataset_str):
    print('start load data')
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'edge_index', 'edge_weight']
    objects = []
    for i in range(len(names)):
        with open(f"data/{dataset_str}/graph/ind.{dataset_str}.{names[i]}", 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))
    x, y, tx, ty, allx, ally, edge_index, edge_weight = tuple(objects)
    
    print(x.shape, y.shape, tx.shape, ty.shape, allx.shape, ally.shape, edge_index.shape, edge_weight.shape)

    features = sp.vstack((allx, tx)).tolil().toarray()

    labels = np.append(ally, ty)

    idx_test = range(len(ally), len(labels))
    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])
    
    data = Data(x=torch.tensor(features, dtype=torch.float32), edge_index=torch.tensor(edge_index, dtype=torch.long), y=torch.tensor(labels))
    data.train_mask = torch.tensor(train_mask)
    data.val_mask = torch.tensor(val_mask)
    data.test_mask = torch.tensor(test_mask)
    data.edge_attr = torch.tensor(edge_weight, dtype=torch.float32).unsqueeze(-1)
    print('load data succ')
    return data


def load_corpus(dataset_str):
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'adj']
    objects = []
    for i in range(len(names)):
        with open(f"data/{dataset_str}/graph/ind.{dataset_str}.{names[i]}", 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, adj = tuple(objects)
    print(x.shape, y.shape, tx.shape, ty.shape, allx.shape, ally.shape)

    features = sp.vstack((allx, tx)).tolil()
    labels = np.vstack((ally, ty))
    labels = labels.argmax(axis=1)
    print(len(labels))

    train_idx_orig = parse_index_file(f"data/{dataset_str}/graph/{dataset_str}.train.index")
    train_size = len(train_idx_orig)

    val_size = train_size - x.shape[0]
    test_size = tx.shape[0]

    idx_train = range(len(y))
    idx_val = range(len(y), len(y) + val_size)
    idx_test = range(allx.shape[0], allx.shape[0] + test_size)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask] = labels[train_mask]
    y_val[val_mask] = labels[val_mask]
    y_test[test_mask] = labels[test_mask]

    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    
    features = sp.identity(features.shape[0])
    features = features.toarray()
    # features = preprocess_features(features)
    adj = preprocess_adj(adj)
    
    edge_index = adj[0].transpose()
    edge_attr = adj[1]
    print(edge_index.shape)
    print(edge_attr.shape)
    
    data = Data(x=torch.tensor(features, dtype=torch.float32), edge_index=torch.tensor(edge_index, dtype=torch.long), y=torch.tensor(labels))
    data.train_mask = torch.tensor(train_mask)
    data.val_mask = torch.tensor(val_mask)
    data.test_mask = torch.tensor(test_mask)
    data.edge_attr = torch.tensor(edge_attr, dtype=torch.float32)
    
    return data


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return sparse_to_tuple(features)


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx

def draw_graph(nodes, edges, name):
    import networkx as nx
    from torch_geometric.utils.convert import to_networkx
    import matplotlib.pyplot as plt
    # G = to_networkx(data)
    # nx.draw(G)
    G=nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    nx.draw_networkx(G)

    plt.savefig(f"{name}.png")
    plt.show()

def parse_single_data(data):
    x, edge_index, edge_attr, train_mask, test_mask = data.x, data.edge_index.numpy(), data.edge_attr.numpy(), data.train_mask.numpy(), data.test_mask.numpy()
    adj = np.zeros([x.shape[0], x.shape[0]])
    for i in tqdm(range(edge_index.shape[1])):
        adj[edge_index[0, i], edge_index[1, i]] = 1
    for i in range(train_mask.shape[0]):
        nodes = [i]
        edges = []
        if not train_mask[i]:
            continue
        neighbors_l1 = [j for j in range(len(train_mask)) if adj[i][j] == 1]
        nodes.extend(neighbors_l1)
        edges.extend([[i, j] for j in neighbors_l1])
        draw_graph(nodes, edges, f'{i}_l1')
        neighbors_l2 = []
        for n in neighbors_l1:
            neighbors_l2.extend([j for j in range(len(train_mask)) if adj[n][j] == 1])
            edges.extend([[n, j] for j in range(len(train_mask)) if adj[n][j] == 1])
        
        nodes.extend(neighbors_l2)
        draw_graph(nodes, edges, f'{i}_l2')
        
    

if __name__ == '__main__':
    data = load_data('mr')
    parse_single_data(data)
    # draw_graph(None)