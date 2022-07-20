#TextCheck
from io import open
import os
import unicodedata
import string
import torch
import random
import json
import torch.nn as nn
from os.path import exists

file_exists = exists('dataset.json')
if not file_exists:
    print("You must run \'dataset_generator.py\' first...")
    exit()

with open('dataset.json') as json_file:
    dataset = json.load(json_file)

all_letters_lang = {'fa':['اآبپتجچحخدذرزژسشصضطظعغفقکگلمنوهیءؤئيإأةك'],
                    'en':['abcdefghijklmnopqrstuvwxyz'],
                    'symbols':[':!@#$%^&*()_+-=\\/?,.{}[]\'\"<>`~']}

all_typing_mode = ['Farsi','English', 'InvFarsi', 'InvEnglish']
def getCategory(letter):
    for lang in all_letters_lang:
        if all_letters_lang[lang][0].find(letter) != -1:
            return lang
    return 'undef'

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

def categoryFromOutput(output):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return all_typing_mode[category_i], category_i

def getWordsOfLine(text):
    return text.split();

def getRandomWord():
    sample_text = ''
    while sample_text == '':
        sample_category_index = random.randint(0, len(dataset)-1)
        sample_category = all_typing_mode[sample_category_index]
        sample_index = random.randint(0, len(dataset[sample_category]) - 1)
        sample_text = dataset[sample_category][sample_index]

    if sample_category == 'English' or sample_category == 'InvFarsi':
        wrapped_text = ''
        for letter in sample_text:
            wrapped_text += letter.upper() if bool(random.getrandbits(1)) else letter.lower()
        sample_text = wrapped_text

    output_tensor = torch.tensor([sample_category_index], dtype=torch.long)
    word_tensor = wordToTensor(sample_text)
    return sample_category, sample_text, output_tensor, word_tensor


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

input = letterToTensor('ا')
hidden1 =torch.zeros(1, hidden_size)
hidden2 =torch.zeros(1, hidden_size)

output, next_hidden1, next_hidden2 = rnn(input, hidden1, hidden2)

input = wordToTensor('سلام')
hidden1 = torch.zeros(1, hidden_size)
hidden2 = torch.zeros(1, hidden_size)

output, next_hidden1, next_hidden2 = rnn(input[0], hidden1, hidden2)


learning_rate = 0.005
criterion = nn.NLLLoss()
def train(output_tensor, word_tensor):
    hidden1, hidden2 = rnn.initHidden()

    rnn.zero_grad()

    for i in range(word_tensor.size()[0]):
        output, hidden1, hidden2 = rnn(word_tensor[i], hidden1, hidden2)

    loss = criterion(output, output_tensor)
    loss.backward()

    # Add parameters' gradients to their values, multiplied by learning rate
    for p in rnn.parameters():
        p.data.add_(p.grad.data, alpha=-learning_rate)

    return output, loss.item()


import time
import math

n_iters = 100000
print_every = 5000
plot_every = 1000



# Keep track of losses for plotting
current_loss = 0
all_losses = []

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

start = time.time()

for iter in range(1, n_iters + 1):
    sample_category, sample_text, output_tensor, word_tensor = getRandomWord()
    output, loss = train(output_tensor, word_tensor)
    current_loss += loss

    # Print iter number, loss, name and guess
    if iter % print_every == 0:
        guess, guess_i = categoryFromOutput(output)
        correct = '✓' if guess == sample_category else '✗ (%s)' % sample_category
        print('%d %d%% (%s) %.4f %s / %s %s' % (iter, iter / n_iters * 100, timeSince(start), loss, sample_text, guess, correct))

    # Add current loss avg to list of losses
    if iter % plot_every == 0:
        all_losses.append(current_loss / plot_every)
        current_loss = 0

def evaluate(word_tensor):
    hidden1, hidden2 = rnn.initHidden()

    for i in range(word_tensor.size()[0]):
        output, hidden1, hidden2 = rnn(word_tensor[i], hidden1, hidden2)

    return output

torch.save(rnn.state_dict(), 'model.dat')