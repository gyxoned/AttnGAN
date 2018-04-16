from __future__ import print_function

from miscc.config import cfg, cfg_from_file
from datasets import TextDataset
from trainer import condGANTrainer as trainer

import os
import sys
import time
import random
import pprint
import datetime
import dateutil.tz
import argparse
import numpy as np
import json

import torch
import torchvision.transforms as transforms

dir_path = (os.path.abspath(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(dir_path)

def gen_example(wordtoix, algo, sent):
    '''generate images from example sentences'''
    from nltk.tokenize import RegexpTokenizer
    captions = []
    cap_lens = []
    # for sent in sentences:
    if len(sent) == 0:
        return 0
    sent = sent.replace("\ufffd\ufffd", " ")
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(sent.lower())
    if len(tokens) == 0:
        print('sent', sent)
        return 0
    rev = []
    for t in tokens:
        t = t.encode('ascii', 'ignore').decode('ascii')
        if len(t) > 0 and t in wordtoix:
            rev.append(wordtoix[t])
    captions.append(rev)
    cap_lens.append(len(rev))
    max_len = np.max(cap_lens)

    sorted_indices = np.argsort(cap_lens)[::-1]
    cap_lens = np.asarray(cap_lens)
    cap_lens = cap_lens[sorted_indices]
    cap_array = np.zeros((len(captions), max_len), dtype='int64')
    for i in range(len(captions)):
        idx = sorted_indices[i]
        cap = captions[idx]
        c_len = len(cap)
        cap_array[i, :c_len] = cap
    data_dic = (cap_array, cap_lens, sorted_indices)
    algo.gen_example(data_dic)
    return 1


def generate_imgs(cfg_file):
    cfg_from_file(cfg_file)

    manualSeed = 85
    # random.seed(manualSeed)
    # np.random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    torch.cuda.manual_seed_all(manualSeed)

    algo = trainer(5450)
    with open('AttnGAN/data/birds/word2ix.json', 'rb') as fp:
        wordtoix = json.load(fp)
    # return gen_example(wordtoix, algo,sent)
    return wordtoix, algo
