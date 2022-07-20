#!/usr/bin/python
from io import open
import glob
import os
import unicodedata
import string
import torch
import random
import sys
from os import path
import torch.nn as nn
from os.path import exists

file_exists = exists('model.dat')
if not file_exists:
    print("You must train the dataset by \'train.py\' first...")
    exit()

all_letters_lang = {'fa':['اآبپتجچحخدذرزژسشصضطظعغفقکگلمنوهیءؤئيإأةك'],
                    'en':['abcdefghijklmnopqrstuvwxyz'],
                    'symbols':[':!@#$%^&*()_+-=\\/?,.{}[]\'\"<>`~']}

all_typing_mode = ['Farsi','English', 'InvFarsi', 'InvEnglish']

def getTensorLength():
    sum = 0
    for lang in all_letters_lang:
        sum += len(all_letters_lang[lang][0])
    return sum

def getLetterIndex(letter):
    sum = 0
    for lang in all_letters_lang:
        index = all_letters_lang[lang][0].find(letter)
        if index != -1:
            return (sum+index)
        sum += len(all_letters_lang[lang][0])
    return -1

def letterToTensor(letter):
    tensor_len = getTensorLength()
    tensor = torch.zeros(1, tensor_len)
    tensor[0][getLetterIndex(letter)] = 1
    return tensor

def wordToTensor(word):
    word_tensors = torch.zeros(len(word), 1, getTensorLength())

    for li, letter in enumerate(word):
        word_tensors[li][0][getLetterIndex(letter)] = 1
    return word_tensors

class LanguageModelRnn(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LanguageModelRnn, self).__init__()

        self.hidden_size = hidden_size

        self.i_2_h1 = nn.Linear(input_size + hidden_size, hidden_size)
        self.o1_2_h2 = nn.Linear(input_size + hidden_size, hidden_size)

        self.i_2_o1 = nn.Linear(input_size + hidden_size, input_size)
        self.o1_2_o2 = nn.Linear(input_size + hidden_size, output_size)

        self.act_func = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden1, hidden2):
        combined = torch.cat((input, hidden1), 1)
        hidden1 = self.i_2_h1(combined)
        output1 = self.i_2_o1(combined)

        combined = torch.cat((output1, hidden2), 1)
        hidden2 = self.o1_2_h2(combined)
        output2 = self.o1_2_o2(combined)

        output = self.act_func(output2)
        return output, hidden1, hidden2

    def initHidden(self):
        return torch.zeros(1, self.hidden_size), torch.zeros(1, self.hidden_size)

hidden_size = 256
rnn = LanguageModelRnn(getTensorLength(), hidden_size, 4)
rnn.load_state_dict(torch.load('model.dat'))

def evaluate(word_tensor):
    hidden1, hidden2 = rnn.initHidden()

    for i in range(word_tensor.size()[0]):
        output, hidden1, hidden2 = rnn(word_tensor[i], hidden1, hidden2)

    return output

def predict(word, n_predictions=3):
    print('\n> %s' % word)
    with torch.no_grad():
        output = evaluate(wordToTensor(word))

        # Get top N categories
        topv, topi = output.topk(n_predictions, 1, True)
        predictions = []

        for i in range(n_predictions):
            value = topv[0][i].item()
            category_index = topi[0][i].item()
            print('(%.2f) %s' % (value, all_typing_mode[category_index]))
            predictions.append([value, all_typing_mode[category_index]])

torch.save(rnn.state_dict(), 'model.dat')

iterargs = iter(sys.argv)
next(iterargs)
for word in iterargs:
    predict(str(word))
