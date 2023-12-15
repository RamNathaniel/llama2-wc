import gc
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
from word_complete.wc_loss import wc_loss
from word_complete.word_completer import WordCompleter
from word_complete.wc_utils import WcUtils


BOOK = '1984'

CORPUS = f'{WcUtils.DATA_ROOT}/lib/{BOOK}.txt'
SUFFIXES_FOLDER = f'{WcUtils.DATA_ROOT}/{BOOK}.txt'

DEVICE = 'mps' if WcUtils.IS_MAC else 'cuda'

BATCH_SIZE = 32

wc_model = WordCompleter()
wc_model.to(DEVICE)

# Optimizers specified in the torch.optim package
optimizer = torch.optim.SGD(wc_model.parameters(), lr=0.001, momentum=0.9)

def loss_function(gt_indicators: torch.Tensor, llama_probs: torch.Tensor, indicator: torch.Tensor, logits: torch.Tensor):
    wc_probs = torch.nn.functional.softmax(logits, dim=1)
    # sig = torch.nn.functional.sigmoid(indicator)
    return torch.nn.L1Loss().to(DEVICE)(indicator, gt_indicators), \
        torch.sum(gt_indicators * torch.nn.CrossEntropyLoss(reduction='none').to(DEVICE)(wc_probs, llama_probs)) / (torch.sum(gt_indicators) + 1)

def on_batch_train(
        epoch: int,
        batch: int,
        tokens: torch.Tensor,
        indicators: torch.Tensor,
        llama_probs: torch.Tensor,
        context):
    
    data_pipeline_took = time.time() - context.last_batch_time
    
    start_time = time.time()

    context.optimizer.zero_grad(set_to_none=True)
    context.model.zero_grad(set_to_none=True)
    logits, indicator = context.model.forward(tokens.to(DEVICE), 0)

    loss_indicators, loss_probs = loss_function(indicators.to(DEVICE), llama_probs.to(DEVICE), indicator, logits)

    # wc_probs = torch.nn.functional.softmax(logits, dim=1)
    # loss = torch.nn.L1Loss().to(DEVICE)(indicator, torch.unsqueeze(indicators, 1).float().to(DEVICE)) + \
    #     0.1 * torch.nn.CrossEntropyLoss().to(DEVICE)(wc_probs, llama_probs.to(DEVICE))
    
    loss = loss_indicators # + 0.00001 * loss_probs

    loss.backward(retain_graph=True)
    context.optimizer.step()

    took = time.time() - start_time
    print(f'batch: {batch}, loss_ind: {loss_indicators:.6f}, loss_probs: {loss_probs:.6f}, valid: {torch.sum(indicators):.1f}, took: {took:.2f} sec, etl: {data_pipeline_took:.2f} sec')

    batch += 1

    if batch % 200 == 0:
        model_fn = f'{WcUtils.MODELS_FOLDER}/{BOOK}-wc-model_{batch}_{loss}.pth'
        torch.save(context.model.state_dict(), model_fn)

        optimizer_fn = f'{WcUtils.MODELS_FOLDER}/{BOOK}-optimizer_{batch}.pth'
        torch.save(context.optimizer.state_dict(), optimizer_fn)

        # reload the model and optimizer from disk - hoping to remove slowness
        context.model = WordCompleter()
        context.model.load_state_dict(torch.load(model_fn))
        context.model.to(DEVICE)
        
        context.optimizer = torch.optim.SGD(context.model.parameters(), lr=0.1, momentum=0.9)
        context.optimizer.load_state_dict(torch.load(optimizer_fn))

        gc.collect()

    del loss, logits, indicator, loss_indicators, loss_probs

    context.last_batch_time = time.time()

    pass


def on_epoch(epoch: int, context):
    # should run on the valuation set, to see how we are doing.
    print(f'epoch: {epoch}')
    pass


if __name__ == '__main__':
    token_gen = TextfileGen(CORPUS)
    batch_gen_train = BatchGen(token_gen, SUFFIXES_FOLDER, BATCH_SIZE, DEVICE, on_batch_train, on_epoch)

    class Context:
        pass
    context = Context()
    
    context.model = wc_model
    context.optimizer = optimizer
    context.last_batch_time = time.time()

    batch_gen_train.run(epochs=-1, batches=-1, context=context)
