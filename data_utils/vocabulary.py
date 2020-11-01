import os, sys, json

UNK_TOKEN = '[UNK]'
PAD_TOKEN = '[PAD]'

class Vocabulary:
    def __init__(self, vocab_file_path = None, is_padded = False):
        self.token_to_id = {}
        self.id_to_token = {}
        if is_padded:
            self.add_token(PAD_TOKEN)
            self.add_token(UNK_TOKEN)
        if vocab_file_path:
            with open(vocab_file_path, mode='r') as f:
                for line in f:
                    token = line.strip()
                    if token not in self.token_to_id:
                        self.token_to_id[token] = len(self.token_to_id)
        
        for token in self.token_to_id:
            self.id_to_token[self.token_to_id[token]] = token
        
    def index_sentence(self, sent):
        words = sent.split()
        ids = []
        for word in words:
            if word in self.token_to_id:
                ids.append(self.token_to_id[word])
            else:
                ids.append(self.token_to_id[UNK_TOKEN])
        return ids

    def add_token(self, token):
        if token in self.token_to_id:
            return
        self.token_to_id[token] = len(self.token_to_id)
        self.id_to_token[len(self.id_to_token)] = token
