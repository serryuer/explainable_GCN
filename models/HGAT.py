from typing import List

import torch
import torch.nn as nn
import torchvision
from torch.nn import CrossEntropyLoss
import numpy as np
from torch_geometric.nn import GATConv, global_mean_pool
from transformers import AutoTokenizer, AutoModelForMaskedLM, BertModel
from torch.autograd import Variable
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, class_num, alpha=[1,8], gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(torch.tensor([alpha]))
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)

        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]

        probs = (P*class_mask).sum(1).view(-1,1)

        log_p = probs.log()

        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p 

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss

class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super(SemanticAttention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z)
        beta = torch.softmax(w, dim=1)

        return (beta * z).sum(1)


class HGATLayer(nn.Module):

    def __init__(self, num_meta_paths, in_size, out_size, layer_num_heads, dropout):
        super(HGATLayer, self).__init__()

        # One GAT layer for each meta path based adjacency matrix
        self.gat_layers = nn.ModuleList()
        for i in range(num_meta_paths):
            self.gat_layers.append(GATConv(in_channels=in_size, out_channels=out_size, heads=layer_num_heads,
                                           dropout=dropout))
        self.semantic_attention = SemanticAttention(in_size=out_size * layer_num_heads)
        self.num_meta_paths = num_meta_paths

    def forward(self, h, gs):
        semantic_embeddings = []

        for i, g in enumerate(gs):
            semantic_embeddings.append(self.gat_layers[i](h, g).flatten(1))
        semantic_embeddings = torch.stack(semantic_embeddings, dim=1)  # (N, M, D * K)

        return self.semantic_attention(semantic_embeddings)  # (N, D * K)


class HGAT(nn.Module):
    def __init__(self, num_meta_paths, in_size, hidden_size, out_size, num_heads, dropout):
        super(HGAT, self).__init__()

        self.layers = nn.ModuleList()
        self.layers.append(HGATLayer(num_meta_paths, in_size, hidden_size, num_heads[0], dropout))
        for l in range(1, len(num_heads)):
            self.layers.append(HGATLayer(
                                         num_meta_paths, 
                                         hidden_size * num_heads[l - 1], 
                                         hidden_size, num_heads[l], dropout)
                               )

    def forward(self, g, h):
        for gnn in self.layers:
            h = gnn(h, g)
        return h


class HGATForTextClassification(torch.nn.Module):
    def __init__(self, 
                 num_class, 
                 dropout, 
                 node_size,
                 embed_dim,
                 hidden_size=768, 
                 layer=2, 
                 edge_mask=[1,1,1,1,1],
                 gat=True):
        super(HGATForTextClassification, self).__init__()
        
        self.num_class = num_class
        self.num_heads = [1] * np.sum(edge_mask)
        self.edge_types = np.sum(edge_mask)
        self.edge_mask = edge_mask
        self.hgat_nets = nn.ModuleList()
        self.hidden_size = hidden_size
        
        self.node_size = node_size
        self.embed_dim = hidden_size
        self.embed = torch.nn.Embedding(num_embeddings=node_size, embedding_dim=embed_dim)

        self.hgat_nets.append(
            HGAT(self.edge_types, 
                 self.hidden_size, 
                 self.hidden_size, 
                 self.num_class, 
                 self.num_heads, 
                 dropout))
        for l in range(1, layer):
            self.hgat_nets.append(
                HGAT(self.edge_types, 
                     self.hidden_size * self.num_heads[-1], 
                     self.hidden_size, 
                     self.num_class,
                     self.num_heads, 
                     dropout))

        self.classifier = nn.Linear(self.hidden_size * (self.num_heads[-1]) * 2, self.num_class)

        self.__init_weights__()

    def __init_weights__(self):
        torch.nn.init.xavier_uniform_(self.classifier.weight)

    def forward(self, x, input):
        node_features = self.embed(x).squeeze()
        edge_indexes = [input.l2d_edge_index, 
                        input.d2l_edge_index,
                        input.w2w_edge_index, 
                        input.w2d_edge_index,
                        input.d2w_edge_index]

        for layer in self.hgat_nets:
            node_features = layer([edge_indexes[i] for i in range(len(edge_indexes)) if self.edge_mask[i] == 1], node_features)
            
        return F.log_softmax(node_features, dim=1)
