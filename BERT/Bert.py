import json
import multiprocessing
import os
import torch
from torch import nn
from d2l import torch as d2l

# load pretrain_model
d2l.DATA_HUB['bert.base'] = (d2l.DATA_URL + 'bert.base.torch.zip',
                             '225d66f04cae318b841a13d32af3acc165f253ac')
d2l.DATA_HUB['bert.small'] = (d2l.DATA_URL + 'bert.small.torch.zip',
                              'c72329e68a732bef0452e4b96a1c341c8910f81f')


def load_pretrained_model(pretrained_mode, num_hiddens, ffn_num_hiddens, num_heads, num_blks, dropout, max_len,
                          devices):
    data_dir = d2l.download_extract(pretrained_mode)
    vocab = d2l.Vocab()
    vocab.idx_to_token = json.load(open(os.path.join(data_dir, 'vocab.json')))
    vocab.token_to_idx = {token: idx for idx, token in enumerate(
        vocab.idx_to_token
    )}
    bert = d2l.BERTModel(len(vocab), num_hiddens, ffn_num_hiddens, num_heads=4, dropout=0.2, num_blks=2,
                         max_len=max_len)
    bert.load_state_dict(torch.load(os.path.join(data_dir, 'pretrained.params')))

    return bert, vocab


class BERTClassifier(nn.Module):
    def __init__(self, bert):
        super(BERTClassifier, self).__init__()
        self.encoder = bert.encoder
        self.hidden = bert.hidden
        self.output = nn.LazyLinear(3)

    def forward(self, inputs):
        tokens_X, segments_X, valid_lens_x = inputs
        encoded_X = self.encoder(tokens_X, segments_X, valid_lens_x)
        return self.output(self.hidden(encoded_X[:, 0, :]))
