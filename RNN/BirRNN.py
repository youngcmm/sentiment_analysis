import torch
from torch import nn
from d2l import torch as d2l


class BiRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, num_hiddens,
                 num_layers, **kwargs):
        super(BiRNN, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # 将bidirectional设置为True以获取双向循环神经网络
        self.encoder = nn.LSTM(embed_size, num_hiddens, num_layers=num_layers,
                                bidirectional=True)
        self.decoder = nn.Linear(4 * num_hiddens, 2)
    def forward(self, inputs):
        # LSTM要求输入的维度第一维是时间步数
        # 输出shape为（时间步数，批量大小，词向量维度）
        embeddings = self.embedding(inputs.T)
        self.encoder.flatten_parameters()

        #outputs.shape = （时间步数，批量大小，隐藏单元数*2）
        outputs, _ = self.encoder(embeddings)

        #cat初始时间步和最后时间步的表征输入到全连接层 encoding.shape = (batchSize, 隐藏单元数*4)
        encoding = torch.cat((outputs[0], outputs[-1]), dim=1)

        outs = self.decoder(encoding)

        return outs
