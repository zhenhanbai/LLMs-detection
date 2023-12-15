import torch.nn as nn
import torch
from torch.nn import functional as F
import numpy as np

class FastText(nn.Module):
    def __init__(self, args):
        super(FastText, self).__init__()
        embedding_pretrained = torch.tensor(
            np.load(args.embedding_file)["embeddings"].astype('float32'))
        self.embedding = nn.Embedding.from_pretrained(embedding_pretrained, freeze=False)
        self.dropout = nn.Dropout(args.FastText_drop_rate)
        self.fc = nn.Linear(args.embedding_dim, args.num_labels)

    def forward(self, x):
        embedded = self.embedding(x).float()
        pooled = F.avg_pool2d(embedded, (embedded.shape[1], 1)).squeeze(1)
        pooled = self.dropout(pooled)
        logits = self.fc(pooled)
        return {"logits": logits}

class TextCNN(nn.Module):
    def __init__(self, args):
        super(TextCNN, self).__init__()
        embedding_pretrained = torch.tensor(
            np.load(args.embedding_file)["embeddings"].astype('float32'))
        self.embedding = nn.Embedding.from_pretrained(embedding_pretrained, freeze=False)
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, args.cnn_kernel_size, (k, args.embedding_dim)) for k in [2, 3, 4]])
        self.dropout = nn.Dropout(args.cnn_drop_rate)
        self.fc = nn.Linear(args.cnn_kernel_size * 3, args.num_labels)

    def conv_and_pool(self, x, conv):
        x = conv(x)
        x = F.relu(x).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        out = self.embedding(x)
        out = out.unsqueeze(1)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.dropout(out)
        out = self.fc(out)
        return {"logits": out}

class RNNModel(nn.Module):
    def __init__(self, args):
        super(RNNModel, self).__init__()
        embedding_pretrained = torch.tensor(
            np.load(args.embedding_file)["embeddings"].astype('float32'))
        self.rnn_type = args.rnn_type
        self.embedding = nn.Embedding.from_pretrained(embedding_pretrained, freeze=False)
        if self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(args.embedding_dim,
                               args.rnn_hidden_dim,
                               num_layers=args.rnn_nums_layer,
                               bidirectional=args.rnn_bidirectional,
                               batch_first=args.rnn_batch_first,
                               dropout=args.rnn_drop_rate)
        elif self.rnn_type == 'gru':
            self.rnn = nn.GRU(args.embedding_dim,
                               args.rnn_hidden_dim,
                               num_layers=args.rnn_nums_layer,
                               bidirectional=args.rnn_bidirectional,
                               batch_first=args.rnn_batch_first,
                               dropout=args.rnn_drop_rate)
        else:
            self.rnn = nn.RNN(args.embedding_size,
                               args.embedding_dim,
                               num_layers=args.rnn_nums_layer,
                               bidirectional=args.rnn_bidirectional,
                               batch_first=args.rnn_batch_first,
                               dropout=args.rnn_drop_rate)
        self.dropout = nn.Dropout(args.rnn_drop_rate)
        if args.rnn_bidirectional:
            self.fc = nn.Linear(args.rnn_hidden_dim * 2, args.num_labels)
        else:
            self.fc = nn.Linear(args.rnn_hidden_dim, args.num_labels)

    def forward(self, x):
        x = self.embedding(x)
        self.rnn.flatten_parameters()
        if self.rnn_type in ['rnn', 'gru']:
            output, hidden = self.rnn(x)
        else:
            output, (hidden, cell) = self.rnn(x)
        x = output[:, -1, :]
        x = self.dropout(x)
        logits = self.fc(x)

        return {"logits": logits}