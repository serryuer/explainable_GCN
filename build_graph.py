import os, sys, json, random
import numpy as np
import pickle as pkl
import networkx as nx
from collections import defaultdict
import scipy.sparse as sp
from math import log
from sklearn import svm
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial.distance import cosine
from tqdm import tqdm
import torch
from torch_geometric.data import Data

sys.argv.append('mr')

if len(sys.argv) != 2:
	sys.exit("Use: python build_graph.py <dataset>")

datasets = ['20ng', 'R8', 'R52', 'ohsumed', 'mr']
dataset = sys.argv[1]

if dataset not in datasets:
	sys.exit("wrong dataset name")

word_embeddings_dim = 300

vocab = list([line.strip() for line in open(f'data/{dataset}/raw/{dataset}_vocab.txt').readlines()])
word_id_map = {vocab[i] : i for i in range(len(vocab))}

docs = list([line.strip() for line in open(f'data/{dataset}/raw/{dataset}.clean.txt').readlines()])
doc_metas = list([line.strip() for line in open(f'data/{dataset}/raw/{dataset}.txt').readlines()])

doc_labels = list(set([doc_meta.split('\t')[2] for doc_meta in doc_metas]))
label_id_map = {doc_labels[i]: i for i in range(len(doc_labels))}

train_set = [[docs[i], label_id_map[doc_metas[i].split('\t')[-1]]] for i in range(len(docs)) if doc_metas[i].split('\t')[1].find('train') != -1]
test_set = [[docs[i], label_id_map[doc_metas[i].split('\t')[-1]]] for i in range(len(docs)) if doc_metas[i].split('\t')[1].find('test') != -1]

train_size = len(train_set)
test_size = len(test_set)
doc_size = len(docs)
vocab_size = len(vocab)

node_size = len(vocab) + len(docs)
train_mask = torch.tensor(np.array([1] * train_size + [0] * (node_size - train_size), dtype=np.bool))
test_mask = torch.tensor(np.array([0] * train_size + [1] * test_size + [0] * vocab_size, dtype=np.bool))
labels = torch.tensor([item[1] for item in train_set] + [item[1] for item in test_set] + [-1] * vocab_size)

edge_index_x = []
edge_index_y = []
edge_weight = []

'''
Doc word heterogeneous graph
'''
window_size = 20
windows = []
print(f"parse window of sequence, window size {window_size}")
for doc_words in tqdm(docs, desc='parse window'):
    words = doc_words.split()
    length = len(words)
    if length <= window_size:
        windows.append(words)
    else:
        for j in range(length - window_size + 1):
            window = words[j: j + window_size]
            windows.append(window)
print(f'parsed window size {len(windows)}')

word_window_freq = defaultdict(int)
for window in tqdm(windows, desc='parse word frequency in a window'):
    appeared = set()
    for i in range(len(window)):
        if window[i] in appeared:
            continue
        word_window_freq[window[i]] += 1
        appeared.add(window[i])

word_pair_count = defaultdict(int)
for window in tqdm(windows, desc='parse word pair in a window'):
    for i in range(1, len(window)):
        for j in range(0, i):
            word_i_id = word_id_map[window[i]]
            word_j_id = word_id_map[window[j]]
            if word_i_id == word_j_id:
                continue
            word_pair_count[str(word_i_id) + ',' + str(word_j_id)] += 1
            word_pair_count[str(word_j_id) + ',' + str(word_i_id)] += 1


# pmi as weights
num_window = len(windows)
for key in tqdm(word_pair_count, desc='cal PMI wight for word pair'):
    i, j = list(map(int, key.split(',')))
    count = word_pair_count[key]
    word_freq_i = word_window_freq[vocab[i]]
    word_freq_j = word_window_freq[vocab[j]]
    pmi = log((1.0 * count / num_window) /
              (1.0 * word_freq_i * word_freq_j/(num_window * num_window)))
    if pmi <= 0:
        continue
    edge_index_x.append(doc_size + i)
    edge_index_y.append(doc_size + j)
    edge_weight.append(pmi)

doc_word_freq = defaultdict(int)
for doc_id in tqdm(range(doc_size), desc='cal doc word freq for every doc'):
    doc_words = docs[doc_id].split()
    for word in doc_words:
        word_id = word_id_map[word]
        doc_word_freq[str(doc_id) + ',' + str(word_id)] += 1
        
word_doc_list = defaultdict(list)
for i in tqdm(range(doc_size), desc='parse word doc list'):
    doc_words = docs[i]
    words = doc_words.split()
    appeared = set()
    for word in words:
        if word in appeared:
            continue
        word_doc_list[word].append(i)
        appeared.add(word)

word_doc_freq = {word: len(word_doc_list[word]) for word in word_doc_list}

for i in tqdm(range(doc_size), desc='cal word-doc weight for graph'):
    doc_words = docs[i]
    words = doc_words.split()
    doc_word_set = set()
    for word in words:
        if word in doc_word_set:
            continue
        j = word_id_map[word]
        key = str(i) + ',' + str(j)
        freq = doc_word_freq[key]
        edge_index_x.append(i)
        edge_index_y.append(doc_size + j)
        idf = log(1.0 * doc_size /
                  word_doc_freq[vocab[j]])
        edge_weight.append(freq * idf)
        doc_word_set.add(word)

adj = sp.csr_matrix((edge_weight, (edge_index_x, edge_index_y)), shape=(node_size, node_size))
adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
adj = adj + sp.eye(adj.shape[0])
adj = sp.coo_matrix(adj)
rowsum = np.array(adj.sum(1))
d_inv_sqrt = np.power(rowsum, -0.5).flatten()
d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
adj = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def to_tuple(mx):
    if not sp.isspmatrix_coo(mx):
        mx = mx.tocoo()
    coords = np.vstack((mx.row, mx.col)).transpose()
    values = mx.data
    shape = mx.shape
    return coords, values, shape

if isinstance(adj, list):
    for i in range(len(adj)):
        adj[i] = to_tuple(adj[i])
else:
    adj = to_tuple(adj)

edge_index = adj[0].transpose()
edge_weight = adj[1]

print('parse edge and edge attr succ, shape is:')
print(edge_index.shape, edge_weight.shape)

data_onehot = Data(x=torch.tensor(torch.eye(node_size, node_size), dtype=torch.float32),
            edge_index=torch.tensor(edge_index, dtype=torch.long), 
            y=torch.tensor(labels), 
            train_mask=torch.tensor(train_mask),
            test_mask = torch.tensor(test_mask),
            edge_attr = torch.tensor(edge_weight, dtype=torch.float32))
            
data_id = Data(x=torch.tensor([i for i in range(node_size)], dtype=torch.long),
            edge_index=torch.tensor(edge_index, dtype=torch.long), 
            y=torch.tensor(labels), 
            train_mask=torch.tensor(train_mask),
            test_mask = torch.tensor(test_mask),
            edge_attr = torch.tensor(edge_weight, dtype=torch.float32))

# dump objects
with open(f"data/{dataset}/graph/ind.{dataset}_onehot", 'wb') as f:
    pkl.dump(data_onehot, f)
with open(f"data/{dataset}/graph/ind.{dataset}_id", 'wb') as f:
    pkl.dump(data_id, f)