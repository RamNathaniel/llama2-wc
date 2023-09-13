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
optimizer = torch.optim.SGD(wc_model.parameters(), lr=0.1, momentum=0.9)


def on_batch_train(epoch: int, batch: int, tokens: torch.Tensor, labels: torch.Tensor):
    start_time = time.time()

    optimizer.zero_grad()
    wc_model.zero_grad()
    logits, indicator = wc_model.forward(tokens.to(DEVICE), 0)

    # wc_probs = torch.nn.functional.softmax(logits, dim=1)
    loss = torch.nn.L1Loss()(indicator, torch.unsqueeze(labels, 1).float().to(DEVICE))
    
    loss.backward(retain_graph=True)
    optimizer.step()

    took = time.time() - start_time
    print(f'batch: {batch}, loss: {loss:.6f}, took: {took:.2f} sec')

    batch += 1
    pass

def on_epoch(epoch: int):
    # should run on the valuation set, to see how we are doing.
    print(f'epoch: {epoch}')
    pass


if __name__ == '__main__':
    # start_time = time.time()
    # tokens = [t for t in TextfileGen(CORPUS, tokenizer).get_file_tokens()]    
    # print(f'Loading corpus took: {time.time() - start_time:.2f} sec')

    batch_gen_train = BatchGen(CORPUS, SUFFIXES_FOLDER, BATCH_SIZE, on_batch_train, on_epoch)
    batch_gen_train.run(epochs=-1)
