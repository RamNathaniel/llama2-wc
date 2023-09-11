import os
import torch

import json
import math
from statistics import mean
from pathlib import Path
from llama import ModelArgs, Transformer, Tokenizer, LLaMA
from typing import Tuple, List
import time
from word_complete.batch_gen import BatchGen
from word_complete.textfile_gen import TextfileGen
from word_complete.wc_loss import wc_loss, get_suffix_mask
from word_complete.word_completer import WordCompleter
from word_complete.wc_utils import WcUtils

CORPUS = '/home/ram_nathaniel/lib/1984.txt'
SUFFIXES_FOLDER = '/home/ram_nathaniel/suffixes/1984.txt'
DEVICE = 'cuda'

BATCH_SIZE = 32

tokenizer = Tokenizer(WcUtils.TOKENIZER_PATH)

wc_model = WordCompleter()
wc_model.to(DEVICE)

# Optimizers specified in the torch.optim package
optimizer = torch.optim.SGD(wc_model.parameters(), lr=0.001, momentum=0.9)

batch: int = 0
epoch: int = 0

def on_batch_train(tokens: torch.Tensor, labels: torch.Tensor):
    start_time = time.time()

    optimizer.zero_grad()
    wc_model.zero_grad()
    logits, indicator = wc_model.forward(tokens.to(DEVICE), 0)

    # wc_probs = torch.nn.functional.softmax(logits, dim=1)
    loss = torch.nn.BCELoss(indicator, labels.to(DEVICE))
    
    loss.backward(retain_graph=True)
    optimizer.step()

    took = time.time() - start_time
    print(f'batch: {batch}, loss: {loss:.6f}, took: {took:.2f} sec')

    batch += 1
    pass

def on_epoch():
    # should run on the valuation set, to see how we are doing.
    epoch += 1
    batch = 0
    pass

batch_gen_train = BatchGen(CORPUS, SUFFIXES_FOLDER, BATCH_SIZE, on_batch_train, on_epoch)

batch_gen_train.run(epochs=-1)
