import numpy as np
import pandas as pd
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from torch.utils.data import DataLoader
class ESIM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_classes, dropout = 0.3, pre_trained=None):
        
        super(ESIM, self).__init__()
        self.embedding_dim = embedding_dim
        if pre_trained is None:
            self.word_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        else:
            self.word_embedding = nn.Embedding.from_pretrained(pre_trained)
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.biLstm1 = nn.LSTM(self.embedding_dim, hidden_size, batch_first=True, bidirectional=True)
        self.projection = nn.Sequential(nn.Linear(4*2*hidden_size, hidden_size), nn.ReLU())
        self.biLstm2 = nn.LSTM(self.hidden_size, hidden_size, batch_first=True, bidirectional=True)
        self.classification = nn.Sequential(nn.Dropout(p=self.dropout),
                                            nn.Linear(4*2*hidden_size, hidden_size),
                                            nn.Tanh(),
                                            nn.Dropout(p=self.dropout),
                                            nn.Linear(hidden_size, num_classes))
        self.apply(_init_esim_weights)

    def forward(self, p, h, p_len, h_len):
        
        # embedding layer
        embeded_p = self.word_embedding(p)
        embeded_h = self.word_embedding(h)

        # encoding for premises
        sorted_p_len, sorted_p_index = p_len.sort(0, descending=True)
        re_p, re_p_index = sorted_p_index.sort(0, descending=False)
        sorted_p = embeded_p.index_select(0, sorted_p_index)
        packed_p = rnn_utils.pack_padded_sequence(sorted_p, sorted_p_len, batch_first=True)
        out_p, _ = self.biLstm1(packed_p, None)
        out_packed_p, _ = rnn_utils.pad_packed_sequence(out_p, batch_first=True)
        encoded_p = out_packed_p.index_select(0, re_p_index)

        # encoding for hypothesis
        sorted_h_len, sorted_h_index = h_len.sort(0, descending=True)
        re_h, re_h_index = sorted_h_index.sort(0, descending=False)
        sorted_h = embeded_h.index_select(0, sorted_h_index)
        packed_h = torch.nn.utils.rnn.pack_padded_sequence(sorted_h, sorted_h_len, batch_first=True)
        out_h, _ = self.biLstm1(packed_h, None)
        out_packed_h, _ = torch.nn.utils.rnn.pad_packed_sequence(out_h, batch_first=True)
        encoded_h = out_packed_h.index_select(0, re_h_index)

        # similarity matrix
        similarity = torch.bmm(encoded_p, encoded_h.transpose(1, 2))

        # masks for p and q and similarity
        p_mask = (1 - (p == 0).float()).unsqueeze(2).float()
        h_mask = (1 - (h == 0).float()).unsqueeze(1).float()
        mask = torch.bmm(p_mask, h_mask)
        mask = mask.float()

        # softmax attention for p
        exp_sim = mask * F.softmax(similarity, dim=-1)
        weight_p = exp_sim / (torch.sum(exp_sim, dim=-1, keepdim=True) + 1e-13)
        attended_p = torch.bmm(weight_p, encoded_h)

        # softmax attention for q
        tmp = exp_sim.transpose(1, 2)
        weight_h = tmp / (torch.sum(tmp, dim=-1, keepdim=True) + 1e-13)
        attended_h = torch.bmm(weight_h, encoded_p)

        # concat all features got
        all_p = torch.cat([encoded_p, encoded_p - attended_p, attended_p, encoded_p * attended_p], dim=-1)
        all_h = torch.cat([encoded_h, encoded_h - attended_h, attended_h, encoded_h * attended_h], dim=-1)

        # projection for p and h
        proj_p = self.projection(all_p)
        proj_h = self.projection(all_h)

        # composition for p
        sorted_p2 = proj_p.index_select(0, sorted_p_index)
        packed_p2 = rnn_utils.pack_padded_sequence(sorted_p2, sorted_p_len, batch_first=True)
        out_p2, _ = self.biLstm2(packed_p2, None)
        out_packed_p2, _ = rnn_utils.pad_packed_sequence(out_p2, batch_first=True)
        seq_p = out_packed_p2.index_select(0, re_p_index)

        # composition for h
        sorted_h2 = proj_h.index_select(0, sorted_h_index)
        packed_h2 = rnn_utils.pack_padded_sequence(sorted_h2, sorted_h_len, batch_first=True)
        out_h2, _ = self.biLstm2(packed_h2, None)
        out_packed_h2, _ = rnn_utils.pad_packed_sequence(out_h2, batch_first=True)
        seq_h = out_packed_h2.index_select(0, re_h_index)

        # avg pooling
        avg_p = seq_p.sum(1) / (p_mask.sum(1)).float()
        avg_h = seq_h.sum(1) / (h_mask.transpose(1,2).sum(1)).float()

        # max pooling
        max_p, _ = torch.max(seq_p + ((1-p_mask).float()*(-1e7)), 1)
        max_h, _ = torch.max(seq_h + ((1-h_mask.transpose(1,2)).float()*(-1e7)), 1)

        # classification_features
        features = torch.cat([max_p, avg_p, max_h, avg_h], -1)
        output = self.classification(features)

        return F.softmax(output, dim=-1)
def _init_esim_weights(module):
    """
    Initialise the weights of the ESIM model.
    """
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight.data)
        nn.init.constant_(module.bias.data, 0.0)

    elif isinstance(module, nn.LSTM):
        nn.init.xavier_uniform_(module.weight_ih_l0.data)
        nn.init.orthogonal_(module.weight_hh_l0.data)
        nn.init.constant_(module.bias_ih_l0.data, 0.0)
        nn.init.constant_(module.bias_hh_l0.data, 0.0)
        hidden_size = module.bias_hh_l0.data.shape[0] // 4
        module.bias_hh_l0.data[hidden_size:(2*hidden_size)] = 1.0

        if (module.bidirectional):
            nn.init.xavier_uniform_(module.weight_ih_l0_reverse.data)
            nn.init.orthogonal_(module.weight_hh_l0_reverse.data)
            nn.init.constant_(module.bias_ih_l0_reverse.data, 0.0)
            nn.init.constant_(module.bias_hh_l0_reverse.data, 0.0)
            module.bias_hh_l0_reverse.data[hidden_size:(2*hidden_size)] = 1.0




