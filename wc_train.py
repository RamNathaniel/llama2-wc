import os
import torch

import json
import math
from statistics import mean
from pathlib import Path
from typing import Tuple, List
import time
from word_complete.batch_gen import BatchGen
from word_complete.textfile_gen import TextfileGen
from word_complete.wc_loss import wc_loss, get_suffix_mask
from word_complete.word_completer import WordCompleter
from word_complete.wc_utils import WcUtils

if not WcUtils.IS_MAC:
    from llama import ModelArgs, Transformer, Tokenizer, LLaMA

BOOK = '1984'

CORPUS = f'{WcUtils.DATA_ROOT}/{BOOK}.txt'
SUFFIXES_FOLDER = f'{WcUtils.DATA_ROOT}/{BOOK}.txt'

DEVICE = 'mps' if WcUtils.IS_MAC else 'cuda'

BATCH_SIZE = 32

if not WcUtils.IS_MAC:
    tokenizer = Tokenizer(WcUtils.TOKENIZER_PATH)
else:
    tokenizer = None

wc_model = WordCompleter()
wc_model.to(DEVICE)

# Optimizers specified in the torch.optim package
optimizer = torch.optim.SGD(wc_model.parameters(), lr=0.1, momentum=0.9)


def on_batch_train(
        epoch: int,
        batch: int,
        tokens: torch.Tensor,
        indicators: torch.Tensor,
        llama_probs: torch.Tensor,
        context):
    
    start_time = time.time()

    optimizer.zero_grad(set_to_none=True)
    wc_model.zero_grad(set_to_none=True)
    logits, indicator = wc_model.forward(tokens.to(DEVICE), 0)

    wc_probs = torch.nn.functional.softmax(logits, dim=1)
    loss = torch.nn.L1Loss().to(DEVICE)(indicator, torch.unsqueeze(indicators, 1).float().to(DEVICE)) + \
        torch.nn.CrossEntropyLoss().to(DEVICE)(wc_probs, llama_probs.to(DEVICE))
    
    loss.backward(retain_graph=True)
    optimizer.step()

    took = time.time() - start_time
    print(f'batch: {batch}, loss: {loss:.6f}, took: {took:.2f} sec')

    batch += 1
    pass


def on_epoch(epoch: int, context):
    # should run on the valuation set, to see how we are doing.
    print(f'epoch: {epoch}')
    pass


if __name__ == '__main__':
    # start_time = time.time()
    # tokens = [t for t in TextfileGen(CORPUS, tokenizer).get_file_tokens()]    
    # print(f'Loading corpus took: {time.time() - start_time:.2f} sec')

    batch_gen_train = BatchGen(CORPUS, SUFFIXES_FOLDER, BATCH_SIZE, on_batch_train, on_epoch)
    batch_gen_train.run(epochs=-1, batches=-1, context=None)
