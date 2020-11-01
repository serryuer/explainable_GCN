import os, sys
sys.path.append('/data03/yujunshuai/code/text_gcn/')

import pandas as pd
import torch
from torch.utils.data import Dataset
from data_utils.vocabulary import Vocabulary

class TextDataset(Dataset):
    def __init__(self, data_index_path, data_path, meta_data_path, vocab_path, max_sequence_length, dataset_name, is_test):
        self.data_index_path = data_index_path
        self.data_path = data_path
        self.vocab_path = vocab_path
        self.meta_data_path = meta_data_path
        self.max_sequence_length = max_sequence_length
        self.dataset_name = dataset_name
        self.is_test = is_test

        self.token_vocab = Vocabulary(self.vocab_path, is_padded = True)
        
        self._load_data_()

    def _load_data_(self):
        self.text_len = []
        self.label_vocab = Vocabulary()
        # load labels
        self.labels = []
        with open(self.meta_data_path, mode='r') as f:
            for line in f:
                label_token = line.strip().split('\t')[-1]
                self.label_vocab.add_token(label_token)
                self.labels.append(self.label_vocab.token_to_id[label_token])
        # load index
        self.data_index = []
        with open(self.data_index_path, mode='r') as f:
            for line in f:
                if self.dataset_name == 'R8' or self.dataset_name == 'R52' or self.dataset_name == 'ohsumed' or self.dataset_name == 'MR':
                    if self.is_test and line.find('test') != -1:
                        self.data_index.append(len(self.data_index))
                    elif not self.is_test and line.find('train') != -1:
                        self.data_index.append(len(self.data_index))
                elif self.dataset_name == '20ng':
                    self.data_index.append(int(line.strip()))
                    
        # load data
        self.data = []
        with open(self.data_path, mode='r') as f:
            for line in f:
                ids = self.token_vocab.index_sentence(line.strip())
                if len(ids) > self.max_sequence_length:
                    ids = ids[:self.max_sequence_length]
                    self.text_len.append(self.max_sequence_length)
                elif len(ids) < self.max_sequence_length:
                    self.text_len.append(len(ids))
                    ids.extend([0] * (self.max_sequence_length - len(ids)))
                self.data.append(ids)
        
    def __getitem__(self, item):
        input_ids = torch.LongTensor(self.data[self.data_index[item]])
        text_len = torch.LongTensor([self.text_len[item]])
        label = torch.LongTensor([self.labels[self.data_index[item]]])
        return input_ids, text_len, label

    def __len__(self):
        return len(self.data_index)


if __name__ == '__main__':
    dataset = TextDataset(data_index_path = '/data03/yujunshuai/code/text_gcn/data/text_gcn-master/data/20ng.train.index',
                          data_path = '/data03/yujunshuai/code/text_gcn/data/text_gcn-master/data/corpus/20ng.clean.txt',
                          meta_data_path = '/data03/yujunshuai/code/text_gcn/data/text_gcn-master/data/20ng.txt',
                          vocab_path = '/data03/yujunshuai/code/text_gcn/data/text_gcn-master/data/corpus/20ng_vocab.txt',
                          max_sequence_length = 512)
    print(f'dataset length : {len(dataset)}')
    print(dataset[0])