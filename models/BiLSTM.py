import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import CrossEntropyLoss
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
import numpy as np


class Net(torch.nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        self.batch_size = args.batch_size
        self.output_size = args.output_size
        self.hidden_size = args.hidden_size
        self.vocab_size = args.vocab_size
        self.embed_dim = args.embed_dim
        self.bidirectional = args.bidirectional
        self.dropout = args.dropout
        self.use_cuda = args.cuda
        self.sequence_length = args.sequence_length
        self.lookup_table = nn.Embedding(self.vocab_size, self.embed_dim, padding_idx=args.PAD)
        if args.vocab_embedding_weights:
            embedding_weights = self._load_word_embeddings(args.vocab_embedding_weights)
            self.lookup_table.weight.data.copy_(torch.from_numpy(embedding_weights))
        self.lookup_table.weight.data.uniform_(-1., 1.)
        self.layer_size = args.layer_size
        self.lstm = nn.LSTM(input_size=self.embed_dim,hidden_size=self.hidden_size,num_layers=self.layer_size,
                            dropout=self.dropout,bidirectional=self.bidirectional)
        if self.bidirectional:
            self.num_direction = 2
        else:
            self.num_direction = 1

        self.attention_size = args.attention_size
        if self.use_cuda:
            self.w_omega = Variable(torch.zeros(self.hidden_size * self.num_direction, self.attention_size).cuda())
            self.u_omega = Variable(torch.zeros(self.attention_size).cuda())
        else:
            self.w_omega = Variable(torch.zeros(self.hidden_size * self.num_direction, self.attention_size))
            self.u_omega = Variable(torch.zeros(self.attention_size))

        self.linear = nn.Linear(self.hidden_size * self.num_direction, self.hidden_size)

        self.classifier = nn.Linear(self.hidden_size, self.output_size)

    def attention_net(self, lstm_output):
        output_reshape = torch.Tensor.reshape(lstm_output, [-1, self.hidden_size*self.num_direction])
        # M = tanh(H)
        attn_tanh = torch.tanh(torch.mm(output_reshape, self.w_omega))
        # alpha = softmax(omega.T*M)
        attn_hidden_layer = torch.mm(attn_tanh, torch.Tensor.reshape(self.u_omega, [-1, 1]))
        exps = torch.Tensor.reshape(torch.exp(attn_hidden_layer), [-1, lstm_output.size()[0]])
        alphas = exps / torch.Tensor.reshape(torch.sum(exps, 1), [-1, 1])
        alphas_reshape = torch.Tensor.reshape(alphas, [-1, lstm_output.size()[0], 1])
        state = lstm_output.permute(1, 0, 2)
        # r = H*alpha.T
        attn_output = torch.sum(state * alphas_reshape, 1)
        return attn_output

    def forward_1(self, data):
        input_sentences, text_len, labels = data

        embedding = self.lookup_table(input_sentences)

        packed_input = pack_padded_sequence(embedding, text_len.squeeze(), batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)

        out_forward = output[range(len(output)), text_len.squeeze() - 1, :self.hidden_size]
        out_reverse = output[:, 0, self.hidden_size:]
        out_reduced = torch.cat((out_forward, out_reverse), 1)

        text_fea = self.linear(out_reduced)

        logits = self.classifier(F.relu(text_fea))

        outputs = (logits,)
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.output_size), labels.view(-1))
        outputs = (loss,) + outputs
        return outputs 

    def forward(self, data):
        input_sentences, text_len, labels = data
        input = self.lookup_table(input_sentences)
        input = input.permute(1, 0, 2)

        lstm_output, (final_hidden_state, final_cell_state) = self.lstm(input)
        # print(lstm_output.shape)
        # attn_output = self.attention_net(lstm_output)
        attn_output = lstm_output.sum(0)
        # print(attn_output.shape)
        logits = self.classifier(self.linear(attn_output))
        outputs = (logits,)
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.output_size), labels.view(-1))
        outputs = (loss,) + outputs

        return outputs 

    def _load_word_embeddings(self, embedding_files):
        embeddings = []
        embeddings.append(np.zeros(200))
        embeddings.append(np.zeros(200))
        with open(embedding_files, mode='r') as f:
            for line in f:
                vect = list(map(float, line.strip().split()[1:]))
                embeddings.append(np.array(vect))
        return np.stack(embeddings)  