import os
import torch
from torch import nn
from d2l import torch as d2l
import pickle

d2l.DATA_HUB['aclImdb'] = (
    'http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz',
    '01ada507287d82875905620988597833ad4e0903')
data_dir = d2l.download_extract('aclImdb', 'aclImdb')


def read_imdb(data_dir, is_train):
    data, labels = [], []
    for label in ('pos', 'neg'):
        folder_name = os.path.join(data_dir, 'train' if is_train else 'test', label)
        for file in os.listdir(folder_name):
            with open(os.path.join(folder_name, file), 'rb') as f:
                review = f.read().decode('utf-8').replace('\n', '')
                data.append(review)
                labels.append(1 if label == 'pos' else 0)

    return data, labels

# train_data = read_imdb(data_dir, is_train=True)
# #将数据集存在本地
# save_local_file = 'train_tokens.pkl'
#
# with open(save_local_file, 'wb') as f:
#     pickle.dump(train_data, f)
# print(f"train_tokens 已保存到文件: {save_local_file}")
#
# #句子分列
# train_tokens = d2l.tokenize(train_data[0], token='word')
# #创建词向量，只保留出现频率最低为5的词，使用特殊tokens,pad
# vocab = d2l.Vocab(train_tokens, min_freq=5, reserved_tokens=['<pad>'])
#
# #设置长度为500，截断或填充到500
# num_steps = 500
# train_features = torch.tensor([d2l.truncate_pad(
#     vocab[line], num_steps, vocab['<pad>']) for line in train_tokens])
#
# train_iter = d2l.load_array(train_features,
#                             torch.tensor(train_data[1]), 64) #64是一个batchSize的大小。

def load_data_imdb(batch_size, num_steps=500):
    """返回数据迭代器和IMDb评论数据集的词表"""
    if os.path.exists('train_tokens.pkl') and os.path.exists('test_tokens.pkl'):
        with open('train_tokens.pkl', 'rb') as f:
            train_data = pickle.load(f)
            f.close()
        with open('test_tokens.pkl', 'rb') as f:
            test_data = pickle.load(f)
            f.close()
    else:
        data_dir = d2l.download_extract('aclImdb', 'aclImdb')
        train_data = read_imdb(data_dir, True)
        test_data = read_imdb(data_dir, False)
        save_local_train_file = 'train_tokens.pkl'
        with open(save_local_train_file, 'wb') as f:
            pickle.dump(train_data, f)
        print(f"train_tokens 已保存到文件: {save_local_train_file}")

        save_local_test_file = 'test_tokens.pkl'
        with open(save_local_test_file, 'wb') as f:
            pickle.dump(test_data, f)
        print(f"test_tokens 已保存到文件: {save_local_test_file}")

    train_tokens = d2l.tokenize(train_data[0], token='word')
    test_tokens = d2l.tokenize(test_data[0], token='word')

    vocab = d2l.Vocab(train_tokens, min_freq=5)
    train_features = torch.tensor([d2l.truncate_pad(
        vocab[line], num_steps, vocab['<pad>']) for line in train_tokens])
    test_features = torch.tensor([d2l.truncate_pad(
        vocab[line], num_steps, vocab['<pad>']) for line in test_tokens])
    train_iter = d2l.load_array((train_features, torch.tensor(train_data[1])),
                                batch_size)
    test_iter = d2l.load_array((test_features, torch.tensor(test_data[1])),
                               batch_size,
                               is_train=False)
    return train_iter, test_iter, vocab


train_iter, test_iter, vocab = load_data_imdb(64)
print(train_iter)