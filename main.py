import torch.nn
from tqdm import tqdm

from Attend.Attend import DecomposableAttention
from RNN.BirRNN import BiRNN
import torch
from torch import nn
from d2l import torch as d2l

from readData import load_data_snli


def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
    if type(m) == torch.nn.LSTM:
        for param in m._flat_weights_names:
            if "weight" in param:
                torch.nn.init.xavier_uniform_(m._parameters[param])


def train(net, train_iter, test_iter, loss, trainer, num_epochs, model_name,
          devices=d2l.try_all_gpus()):
    timer, num_batches = d2l.Timer(), len(train_iter)
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0, 1],
                            legend=['train loss', 'train acc', 'test acc'])
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])

    for epoch in tqdm(range(num_epochs)):
        # Sum of training loss, sum of training accuracy, no. of examples,
        # no. of predictions
        metric = d2l.Accumulator(4)

        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            l, acc = train_batch(
                net, features, labels, loss, trainer, devices)
            metric.add(l, acc, labels.shape[0], labels.numel())
            timer.stop()
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[2], metric[1] / metric[3],
                              None))
        test_acc = d2l.evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
    torch.save(net, '{}.pt'.format(model_name))
    print(f'loss {metric[0] / metric[2]:.3f}, train acc '
          f'{metric[1] / metric[3]:.3f}, test acc {test_acc:.3f}')

    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec on '
          f'{str(devices)}')


def train_batch(net, X, y, loss, trainer, devices):
    """Train for a minibatch with multiple GPUs (defined in Chapter 13).

    Defined in :numref:`sec_image_augmentation`"""
    if isinstance(X, list):
        # Required for BERT fine-tuning (to be covered later)
        X = [x.to(devices[0]) for x in X]
    else:
        X = X.to(devices[0])
    y = y.to(devices[0])
    net.train()
    trainer.zero_grad()
    pred = net(X)
    l = loss(pred, y)
    l.sum().backward()
    trainer.step()
    train_loss_sum = l.sum()
    train_acc_sum = d2l.accuracy(pred, y)
    return train_loss_sum, train_acc_sum


def predict_sentiment(net, vocab, sequence):
    sequence = torch.tensor(vocab[sequence.split()], device='cuda')
    label = torch.argmax(net(sequence.reshape(1, -1)), dim=1)
    return 'positive' if label == 1 else 'negative'

def predict_snli(net, vocab, premise, hypothesis):
    """Predict the logical relationship between the premise and hypothesis."""
    net.eval()
    premise = torch.tensor(vocab[premise], device=d2l.try_gpu())
    hypothesis = torch.tensor(vocab[hypothesis], device=d2l.try_gpu())
    label = torch.argmax(net([premise.reshape((1, -1)),
                           hypothesis.reshape((1, -1))]), dim=1)
    return 'entailment' if label == 0 else 'contradiction' if label == 1 \
            else 'neutral'

def main():
    use_RNN = False
    if use_RNN:
        batch_size = 64
        train_iter, test_iter, vocab = d2l.load_data_imdb(batch_size)
        embed_size, num_hiddens, num_layers = 100, 100, 2
        devices = d2l.try_all_gpus()
        net = BiRNN(len(vocab), embed_size, num_hiddens, num_layers)
        net.apply(init_weights)

        print(net)

        glove_embedding = d2l.TokenEmbedding('glove.6b.100d')
        embeds = glove_embedding[vocab.idx_to_token]
        net.embedding.weight.data.copy_(embeds)
        net.embedding.weight.requires_grad = False

        lr, num_hiddens = 0.01, 5
        trainer = torch.optim.Adam(net.parameters(), lr=lr)

        loss = torch.nn.CrossEntropyLoss(reduction='none')

        load = True
        if load:
            net = torch.load('RNN.pt')
        else:
            train(net=net, train_iter=train_iter, test_iter=test_iter,
                  loss=loss, trainer=trainer, num_epochs=num_hiddens, model_name='RNN',
                  devices=devices)

        predict_snli(net, vocab, ['he', 'is', 'good', '.'], ['he', 'is', 'bad', '.'])

    use_Attention = True
    if use_Attention:
        batch_size, num_steps = 128, 50
        train_iter, test_iter, vocab = load_data_snli(batch_size, num_steps)

        embed_size, num_hiddens, devices = 100, 200, d2l.try_all_gpus()
        net = DecomposableAttention(vocab, embed_size, num_hiddens)
        glove_embedding = d2l.TokenEmbedding('glove.6b.100d')
        embeds = glove_embedding[vocab.idx_to_token]
        net.embedding.weight.data.copy_(embeds)

        lr, num_epochs = 0.001, 4
        trainer = torch.optim.Adam(net.parameters(), lr=lr)
        loss = nn.CrossEntropyLoss(reduction="none")


        load = False
        if load:
            net = torch.load('Attend.pt')
        else:
            train(net, train_iter, test_iter, loss, trainer, num_epochs, devices=d2l.try_all_gpus(),
                  model_name='Attend')

        print(predict_sentiment(net, vocab, 'this move is good.'))


if __name__ == '__main__':
    main()
