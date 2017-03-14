# coding: utf-8
#---------------------------------------------------------------------
'''
Neural Machine Translation - Translation experiments
    Training, evaluation and prediction functions
'''
#---------------------------------------------------------------------

# In[ ]:

import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions
from tqdm import tqdm
import sys
import os
from collections import Counter
import math
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import csv
import time
import matplotlib.gridspec as gridspec
import importlib
import pdb
# %matplotlib inline

# In[ ]:

#---------------------------------------------------------------------
# Load configuration
#---------------------------------------------------------------------
from nmt_config import *

# In[ ]:
#---------------------------------------------------------------------
# Load encoder decoder model definition
#---------------------------------------------------------------------
from enc_dec import *

# In[ ]:

xp = cuda.cupy if gpuid >= 0 else np


# In[ ]:
#---------------------------------------------------------------------
# Load dataset
#---------------------------------------------------------------------
## Word to index and index to word
w2i = pickle.load(open(w2i_path, "rb"))
i2w = pickle.load(open(i2w_path, "rb"))
# frequency of word
vocab = pickle.load(open(vocab_path, "rb"))

vocab_size_en = min(len(i2w["en"]), max_vocab_size["en"])
vocab_size_fr = min(len(i2w["fr"]), max_vocab_size["fr"])
print("vocab size, en={0:d}, fr={1:d}".format(vocab_size_en, vocab_size_fr))
print("{0:s}".format("-"*50))
#---------------------------------------------------------------------

# In[ ]:
#---------------------------------------------------------------------
# Set up model
#---------------------------------------------------------------------
model = EncoderDecoder(vocab_size_fr, vocab_size_en,
                       num_layers_enc, num_layers_dec,
                       hidden_units, gpuid, attn=use_attn)


if gpuid >= 0:
    cuda.get_device(gpuid).use()
    model.to_gpu()

optimizer = optimizers.Adam()
optimizer.setup(model)

'''
___QUESTION-1-DESCRIBE-F-START___

- Describe what the following line of code does

add Hook function. hook function is executed after the gradient computation.
this hook function use gradient clipping to clip the exploding gradient 
by add L2 norm threshold after the gradient is computed.

'''
optimizer.add_hook(chainer.optimizer.GradientClipping(threshold=5))
'''___QUESTION-1-DESCRIBE-F-END___'''

# In[ ]:
def plot_dist(text_fname):
    # Set up log file for loss
    fr_len = []
    en_len = []
    fr_word_set = set()
    en_word_set = set()
    with open(text_fname["fr"], "rb") as fr_file, open(text_fname["en"], "rb") as en_file:
        for i, (line_fr, line_en) in enumerate(zip(fr_file, en_file), start=1):
            fr_sent = line_fr.strip().split()
            en_sent = line_en.strip().split()
            fr_len.append(len(fr_sent))
            en_len.append(len(en_sent))
            for word in fr_sent:
                fr_word_set.add(word)
            for word in en_sent:
                en_word_set.add(word)
    # pdb.set_trace()

    fr_UNK_counter=0
    for word in fr_word_set:
        if word not in START_VOCAB[:-1]:
            if ( not w2i["fr"].get(word) ) or ( w2i["fr"].get(word)==UNK_ID):
                # pdb.set_trace()
                fr_UNK_counter+=1

    en_UNK_counter=0
    for word in en_word_set:
        if word not in START_VOCAB[:-1]:
            if ( not w2i["en"].get(word) ) or ( w2i["en"].get(word)==UNK_ID):
                en_UNK_counter+=1



    # fig_1 = plt.figure()#figsize=(8,4))
    ax_1 = plt.subplot(211)
    ax_1.hist(fr_len,bins=30)
    ax_1.set_xlabel("length")
    ax_1.set_ylabel("frequency")
    ax_1.set_title("japanese length sentence distribution")
    plt.xticks(np.arange(0,70,5))
    # fig_2 = plt.figure()#figsize=(8,4))
    ax_2 = plt.subplot(212)
    ax_2.hist(en_len,bins=30)
    ax_2.set_xlabel("length")
    ax_2.set_ylabel("frequency")
    ax_2.set_title("english length sentence distribution")
    plt.xticks(np.arange(0,70,5))

    # ax_1.axis(np.arange(0,70,2))




    # ax_2.axis(np.arange(0,70,2))
    plt.tight_layout()
    plt.savefig("hasil.png")

    plt.clf()
    ax_3 = plt.subplot(111)
    ax_3.plot(fr_len,en_len,'bo')
    ax_3.set_xlabel("japanese")
    ax_3.set_ylabel("english")
    ax_3.set_title("correlation between japanese-english")
    # ax_1.axis(np.arange(0,70,2))
    plt.savefig("hasil_corr.png")

    print("word token in englsh data",sum(en_len))
    print("word token in japan data",sum(fr_len))
    print("word type in english data %d"% (len(en_word_set)))
    print("word type in japan data %d"% (len(fr_word_set)))
    print("number of UNK in english data %d"% (en_UNK_counter))
    print("number of UNK in japan data %d"% (fr_UNK_counter))
    
    # pdb.set_trace()
plot_dist(text_fname)
