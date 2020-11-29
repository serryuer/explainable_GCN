import os, sys, json
import pandas as pd
import matplotlib.pyplot as plt
import torch
import pickle as pkl
from captum.attr import LayerIntegratedGradients, TokenReferenceBase, visualization, IntegratedGradients
from models.GCNNet import GCNNet
import scipy.sparse as sp
from torch_geometric.data import Data
import numpy as np


alg_map = {
     
}


dataset = 'mr'
vocab = list([line.strip() for line in open(f'data/{dataset}/raw/{dataset}_vocab.txt').readlines()])
word_id_map = {vocab[i] : i for i in range(len(vocab))}
id_word_map = {i : vocab[i] for i in range(len(vocab))}

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
labels = torch.tensor([item[1] for item in train_set] + [item[1] for item in test_set] + [-1] * vocab_size)

datasets = 'mr'

def predict(x, edge_index, edge_attr):
    return model(x, edge_index, edge_attr)
    
def text_gcn_forward_func(x, edge_index, edge_attr, docid=0):
    pred = predict(x, edge_index, edge_attr)
    pred = pred[docid]
    return pred

def get_tokens_by_docid(id):
    assert id < len(docs), 'invlaid id'
    tokens = docs[id].split()
    token_ids = [word_id_map[word] for word in tokens]
    if id < len(test_set):
        label = train_set[id][1]
    else:
        label = test_set[id - len(train_set)][1]
    return tokens, token_ids, label
        
def forward_with_sigmoid(x, edge_index, edge_attr, docid):
    return torch.sigmoid(text_gcn_forward_func(x, edge_index, edge_attr, docid))


def plot_with_networkx(node_indexes, node_texts, node_weights):
    import networkx as nx
    import matplotlib.pyplot as plt
    # 解决图像中的中文乱码问题
    plt.rcParams['font.sans-serif'] = ['SimHei'] 
    plt.rcParams['font.family']='sans-serif'

    g = nx.DiGraph()
    g.clear()
    
    for i, index in enumerate(node_indexes):
        for neighbor, weight in enumerate(adj[index]):
            if neighbor in node_indexes and neighbor != index:
                # g.add_edge(f"{node_weights[i]}:{node_texts[i]}", f"{node_weights[node_indexes.index(neighbor)]}:{node_texts[node_indexes.index(neighbor)]}", weight=weight)
                g.add_edge(node_indexes.index(index), node_indexes.index(neighbor), weight=weight)
    print(g)
    print(g.nodes())
    print(g.nodes().data()) # 显示边的数据
    print(g.edges().data())

    pos=nx.spring_layout(g);
    nx.draw_spring(g,with_labels=True); # 显示节点的名称
    nx.draw_networkx_edge_labels(g,pos,font_size=14,alpha=0.5,rotate=True); 

    plt.axis('off')
    plt.show()

def add_attributions_to_visualizer(attributions, text, token_ids, pred, pred_ind, label, delta, vis_data_records, top_k=10):
    attributions = attributions.sum(dim=2).squeeze(0)
    attributions = attributions / torch.norm(attributions)
    attributions = attributions.cpu().detach().numpy()
    node_ids = [id + len(docs) for id in token_ids]
    tokens_attributions = attributions[node_ids]
    sorted_index_h = np.argsort(-attributions)
    sorted_index_l = np.argsort(attributions)
    top_k_index_h = sorted_index_h[:top_k]
    top_k_index_l = sorted_index_l[:top_k]
    other_tokens_index_h = np.array([index for index in top_k_index_h if index >= doc_size and index not in node_ids])
    other_tokens_index_l = np.array([index for index in top_k_index_l if index >= doc_size and index not in node_ids])
    other_tokens_attributions_h = {id_word_map[index - doc_size]: float(attributions[index]) for index in other_tokens_index_h if index not in node_ids}
    other_tokens_attributions_l = {id_word_map[index - doc_size]: float(attributions[index]) for index in other_tokens_index_l if index not in node_ids}
    doc_index_h = np.array([index for index in top_k_index_h if index < doc_size])
    doc_index_l = np.array([index for index in top_k_index_l if index < doc_size])
    doc_attributions_h = {docs[index]: float(attributions[index]) for index in doc_index_h}
    doc_attributions_l = {docs[index]: float(attributions[index]) for index in doc_index_l}
    print('all tokens with attributions in this doc')
    print(json.dumps({token: float(attribution) for token, attribution in zip(text, tokens_attributions)}, indent=4, ensure_ascii=False))
    print(f'other words -{len(other_tokens_attributions_h)}- and attribution in top {top_k} nodes')
    print(json.dumps(other_tokens_attributions_h, indent=4, ensure_ascii=False))
    print(json.dumps(other_tokens_attributions_l, indent=4, ensure_ascii=False))
    print(f'docs -{len(doc_attributions_h)}- and attribution in top {top_k} nodes')
    print(json.dumps(doc_attributions_h, indent=4, ensure_ascii=False))
    print(json.dumps(doc_attributions_l, indent=4, ensure_ascii=False))
    vis_data_records.append(visualization.VisualizationDataRecord(
                            tokens_attributions,
                            pred,
                            pred_ind,
                            label,
                            '1',
                            tokens_attributions.sum(),       
                            text,
                            delta))
    node_indexes = np.concatenate([top_k_index_h, top_k_index_l, node_ids])
    node_texts = [str(index) if index < doc_size else id_word_map[index - doc_size] for index in node_indexes]
    
    node_weights = attributions[node_indexes]
    plot_with_networkx(node_indexes.tolist(), node_texts, node_weights.tolist())



def interpret_sentence(docid):
    vis_data_records_ig = []
    model.zero_grad()
    tokens, token_ids, label = get_tokens_by_docid(docid)
    pred = forward_with_sigmoid(data.x, data.edge_index, data.edge_attr, docid)[label]
    pred_ind = round(pred.detach().cpu().item())
    # compute attributions and approximation delta using layer integrated gradients
    token_reference = TokenReferenceBase(reference_token_idx=0)
    reference_indices = token_reference.generate_reference(data.x.shape[0], device='cuda:3').unsqueeze(0)
    attributions_ig, delta = lig.attribute(data.x.unsqueeze(0), 
                                           reference_indices, 
                                           additional_forward_args=(data.edge_index.unsqueeze(0), data.edge_attr.unsqueeze(0), docid), 
                                           n_steps=50, 
                                           return_convergence_delta=True, 
                                           internal_batch_size=1)
    print(f'pred: {pred}, delta: {abs(delta)}')
    print(attributions_ig)
    add_attributions_to_visualizer(attributions_ig, tokens, token_ids, pred, pred_ind, label, delta, vis_data_records_ig)
    visualization.visualize_text(vis_data_records_ig)
    
device = torch.device(f'cuda:3' if torch.cuda.is_available() else 'cpu')
data = pkl.load(open(f"data/{datasets}/graph/ind.{datasets}_id", 'rb'), encoding='latin1')
adj = sp.csr_matrix((data.edge_attr.cpu().numpy(), (data.edge_index[0].cpu().numpy(), data.edge_index[1].cpu().numpy())), shape=(data.x.shape[0], data.x.shape[0]))
adj = adj.toarray()

model = GCNNet(node_size = 29426, embed_dim=200, hidden_dim=256, embedding_finetune=True, num_class = 2, dropout = 0.5, layers = 2).to(device)
model.eval()
model.load_state_dict(torch.load('/mnt/nlp-lq/yujunshuai/code/explainable_GCN/experiments/mr_id.pt'))
data = data.to(device)
lig = LayerIntegratedGradients(text_gcn_forward_func, model.embed)

interpret_sentence(0)
