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
import json
import re
import zipfile
from os.path import exists

file_exists1 = exists('data/farsi.txt')
file_exists2 = exists('data/english.txt')

if not file_exists1 or not file_exists2:
    file_exists3 = exists('data/data.zip')
    if not file_exists3:
        print("*.txt files missed!");
        exit()
    with zipfile.ZipFile('data/data.zip', 'r') as zip_ref:
        zip_ref.extractall('data/')


letters_map_farsi = 'ضصثقفغعهخحجچشسیبلاتنمکگظطزرذدپوآءژك'
letters_map_english = 'qwertyuiop[]asdfghjkl;\'zxcvbnm,HMCZ'
input_file1 = 'data/farsi.txt'
input_file2 = 'data/english.txt'
dataset = {}
dataset['Farsi'] = []
dataset['English'] = []
dataset['InvFarsi'] = []
dataset['InvEnglish'] = []

string1 = open(path.relpath(input_file1), encoding='utf-8').read().strip()
words1 = re.split(' |,|\n',string1)
while("" in words1) :
    words1.remove("")

dataset['Farsi'].extend(words1)
inv_words1 = []

for word in words1:
    inv_word = ''
    for letter in word:
        letter_index = letters_map_farsi.find(letter)
        if letter_index != -1:
            inv_word+=(letters_map_english[letter_index])
    dataset['InvFarsi'].append(inv_word)

string2 = open(path.relpath(input_file2), encoding='utf-8').read().strip()
words2 = re.split(' |,|\n',string2)
while("" in words2) :
    words2.remove("")
dataset['English'].extend(words2)
inv_words2 = []

for word in words2:
    inv_word = ''
    for letter in word:
        letter_index = letters_map_english.find(letter)
        if letter_index != -1:
            inv_word+=(letters_map_farsi[letter_index])
    dataset['InvEnglish'].append(inv_word)

with open('dataset.json', 'w') as outfile:
    json.dump(dataset, outfile)
