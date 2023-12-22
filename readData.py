import os
import torch
from torch import nn
from d2l import torch as d2l
import pickle
import re
import torch.utils.data


def read_imdb(data_dir, is_train):
    d2l.DATA_HUB['aclImdb'] = (
        'http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz',
        '01ada507287d82875905620988597833ad4e0903')
    data_dir = d2l.download_extract('aclImdb', 'aclImdb')
    data, labels = [], []
    for label in ('pos', 'neg'):
        folder_name = os.path.join(data_dir, 'train' if is_train else 'test', label)
        for file in os.listdir(folder_name):
            with open(os.path.join(folder_name, file), 'rb') as f:
                review = f.read().decode('utf-8').replace('\n', '')
                data.append(review)
                labels.append(1 if label == 'pos' else 0)

    return data, labels


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


def read_snli(is_train):
    # d2l.DATA_HUB['SNLI'] = (
    #     'https://nlp.stanford.edu/projects/snli/snli_1.0.zip',
    #     '9fcde07509c7e87ec61c640c1b2753d9041758e4')

    data_dir = "F:/code/NLP/sentimen_analysis/snli_1.0"

    def extract_text(s):
        # 删除我们不会使用的信息
        s = re.sub('\\(', '', s)
        s = re.sub('\\)', '', s)
        # 用一个空格替换两个或多个连续的空格
        s = re.sub('\\s{2,}', ' ', s)
        return s.strip()
    label_set = {'entailment': 0, 'contradiction': 1, 'neutral': 2}
    file_name = os.path.join(data_dir, 'snli_1.0_train.txt'
                             if is_train else 'snli_1.0_test.txt')
    with open(file_name, 'r') as f:
        rows = [row.split('\t') for row in f.readlines()[1:]]
    premises = [extract_text(row[1]) for row in rows if row[0] in label_set]
    hypotheses = [extract_text(row[2]) for row in rows if row[0] in label_set]
    labels = [label_set[row[0]] for row in rows if row[0] in label_set]
    # print(labels)

    return premises, hypotheses, labels


class SNLIDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, num_steps, vocab=None):
        self.num_steps = num_steps
        all_premise_tokens = d2l.tokenize(dataset[0])
        all_hypothesis_tokens = d2l.tokenize(dataset[1])
        if vocab is None:
            self.vocab = d2l.Vocab(all_premise_tokens + all_hypothesis_tokens,
                                   min_freq=5, reserved_tokens=['<pad>'])
        else:
            self.vocab = vocab
        self.premises = self._pad(all_premise_tokens)
        self.hypotheses = self._pad(all_hypothesis_tokens)
        self.labels = torch.tensor(dataset[2])
        print('read ' + str(len(self.premises)) + ' examples')

    def _pad(self, lines):
        return torch.tensor([d2l.truncate_pad(
            self.vocab[line], self.num_steps, self.vocab['<pad>'])
            for line in lines])

    def __getitem__(self, idx):
        return (self.premises[idx], self.hypotheses[idx]), self.labels[idx]

    def __len__(self):
        return len(self.premises)


def load_data_snli(batch_size, num_step=50):
    # num_workers = d2l.get_dataloader_workers()
    if os.path.exists('train_data_snli.pkl') and os.path.exists('test_data_snli.pkl'):
        with open('train_data_snli.pkl', 'rb') as f:
            train_data = pickle.load(f)
            f.close()
        with open('test_data_snli.pkl', 'rb') as f:
            test_data = pickle.load(f)
            f.close()
    else:
        train_data = read_snli(is_train=True)
        test_data = read_snli(is_train=False)
        save_local_train_file = 'train_data_snil.pkl'

        with open(save_local_train_file, 'wb') as f:
            pickle.dump(train_data, f)
        print(f"train_data 已保存到文件: {save_local_train_file}")

        save_local_test_file = 'test_data_snil.pkl'

        with open(save_local_test_file, 'wb') as f:
            pickle.dump(test_data, f)
        print(f"test_data 已保存到文件: {save_local_test_file}")

    train_set = SNLIDataset(train_data, num_step)
    test_set = SNLIDataset(test_data, num_step, train_set.vocab)

    train_iter = torch.utils.data.DataLoader(train_set, batch_size, shuffle=True)
    test_iter = torch.utils.data.DataLoader(test_set, batch_size, shuffle=False)

    return train_iter, test_iter, test_set.vocab


# train_iter, test_iter, vocab = load_data_snli(128, 50)

# for i, (_, _) in range(train_iter):
#     pass
# len(vocab)


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
#                             torch.tensor(train_data[1]), 64) #64是一个batchSize的大小。# train_data = read_imdb(data_dir, is_train=True)
# # #将数据集存在本地
# # save_local_file = 'train_tokens.pkl'
# #
# # with open(save_local_file, 'wb') as f:
# #     pickle.dump(train_data, f)
# # print(f"train_tokens 已保存到文件: {save_local_file}")
# #
# # #句子分列
# # train_tokens = d2l.tokenize(train_data[0], token='word')
# # #创建词向量，只保留出现频率最低为5的词，使用特殊tokens,pad
# # vocab = d2l.Vocab(train_tokens, min_freq=5, reserved_tokens=['<pad>'])
# #
# # #设置长度为500，截断或填充到500
# # num_steps = 500
# # train_features = torch.tensor([d2l.truncate_pad(
# #     vocab[line], num_steps, vocab['<pad>']) for line in train_tokens])
# #
# # train_iter = d2l.load_array(train_features,
# #                             torch.tensor(train_data[1]), 64) #64是一个batchSize的大小。
# for i, (features, labels) in enumerate(train_iter):
