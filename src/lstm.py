#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""The LSTM sample script."""

import csv
import random
import sys

import torch
import torch.nn as nn
import torch.autograd as autograd

import torch.optim as optim

from logging import (getLogger, Formatter, StreamHandler, INFO)

logger = getLogger(__name__)
logger.setLevel(INFO)
sh = StreamHandler()
logger.addHandler(sh)
fmtr = Formatter('%(asctime)s:%(lineno)d:%(levelname)s:%(message)s')
sh.setFormatter(fmtr)

MAX_LENGTH = 5
TRAINING_ITER = 4


class LSTM(nn.Module):
    """LSTM"""

    def __init__(self, input_dim, embed_dim, hidden_dim):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.embeds = nn.Embedding(input_dim, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim)
        self.linear = nn.Linear(hidden_dim, input_dim)
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax(dim=1)
        self.hidden = self.init_hidden()

    def forward(self, input_, hidden):
        embeds = self.embeds(input_)
        lstm_out, hidden = self.lstm(embeds.view(len(input_), 1, -1), hidden)
        output = self.linear(lstm_out.view(len(input_), -1))
        output = self.dropout(output)
        output = self.softmax(output)

        return output, hidden

    def init_hidden(self):
        return (autograd.Variable(torch.zeros(1, 1, self.hidden_dim)),
                autograd.Variable(torch.zeros(1, 1, self.hidden_dim)))


def train(model, criterion, input_, target):
    hidden = model.init_hidden()
    model.zero_grad()

    output, _ = model(input_, hidden)
    topv, topi = output.data.topk(1)
    loss = criterion(output, target)

    loss.backward()

    return loss.item() / input_.size()[0]


def input_tensor(input_idxs):
    tensor = torch.LongTensor(input_idxs)
    return autograd.Variable(tensor)


def target_tensor(input_idxs, eof_idx):
    input_idxs.append(eof_idx)
    tensor = torch.LongTensor(input_idxs)
    return autograd.Variable(tensor)


def main():
    """Main function."""
    def _generate(start_letter):
        sample_char_idx = [char2idx[start_letter]]
        logger.debug('sample_char_idx: ', sample_char_idx)
        input_ = input_tensor(sample_char_idx)
        hidden = model.init_hidden()
        output_name = start_letter

        for i in range(MAX_LENGTH):
            output, hidden = model(input_, hidden)
            _, topi = output.data.topk(1)
            logger.debug('topi before: ', topi)
            topi = topi.item()
            logger.debug('topi: ', topi)
            logger.debug('char2idx: ', char2idx['EOS'])
            if topi == char2idx['EOS']:
                break
            else:
                letter = idx2char[topi]
                output_name += letter
            input_ = input_tensor([topi])

        return output_name

    def generate(start_letters):
        for start_letter in start_letters:
            print(_generate(start_letter))

    with open('../input/pseudo-personal-infomation.csv') as f:
        reader = csv.reader(f)
        reader.__next__()  # skip header
        first_names = [row[2] for row in reader]  # list of first name
    all_char_set = set([char for first_name in first_names for char in first_name])
    char2idx = {char: i for i, char in enumerate(all_char_set)}
    char2idx['EOS'] = len(char2idx)

    idx2char = {v: k for k, v in char2idx.items()}

    names_idxs = [[char2idx[char] for char in name_str] for name_str in first_names]

    # build model
    model = LSTM(input_dim=len(char2idx), embed_dim=100, hidden_dim=128)

    criterion = nn.NLLLoss()
    optimizer = optim.RMSprop(model.parameters(), lr=0.001)

    for itr in range(TRAINING_ITER + 1):
        random.shuffle(names_idxs)
        total_loss = 0

        for i, name_idxs in enumerate(names_idxs):
            input_ = input_tensor(name_idxs)
            target = target_tensor(name_idxs[1:], char2idx['EOS'])

            loss = train(model, criterion, input_, target)
            total_loss += loss

            optimizer.step()

        print(itr, '/', TRAINING_ITER)
        print('loss {:.4f}'.format(float(total_loss / len(names_idxs))))

    generate(u'アイウエオカキクケコサシスセソタチツテト')


if __name__ == '__main__':
    sys.exit(main())
