import torch

import os
import json
import math
from statistics import mean
from pathlib import Path
from llama import ModelArgs, Transformer, Tokenizer, LLaMA
from typing import Tuple, List
import time
from word_complete.textfile_gen import TextfileGen
from word_complete.wc_loss import wc_loss, get_suffix_mask
from word_complete.word_completer import WordCompleter
from word_complete.wc_utils import WcUtils
from models_files_manager import ModelsFilesManager
from iterutils import IterUtil

CORPUS = '/home/ram_nathaniel/lib/1984.txt'
DEVICE = 'cuda'

output_folder = os.path.join('/home/ram_nathaniel/suffixes', os.path.basename(CORPUS))
os.makedirs(output_folder, exist_ok=True)

manager = ModelsFilesManager(output_folder, ['pos', 'suffix_percent', 'probs'], ext='txt')
last_pos_fn = IterUtil(manager.get_model_files()).max_item(lambda fn: int(manager.get_fields_values(fn)['pos']))
last_pos = max([int(manager.get_fields_values(fn)['pos']) for fn in manager.get_model_files()])

if last_pos is not None:
    print(f'Last pos: {last_pos} - we will continue from there')

BATCH_SIZE = 1

tokenizer = Tokenizer(WcUtils.TOKENIZER_PATH)
suffix_mask = get_suffix_mask(tokenizer, torch.device(DEVICE)).detach()
suffix_count = torch.sum(suffix_mask.float()).item()

tokens_gen = TextfileGen(CORPUS, tokenizer).get_file_tokens()

WINDOW_SIZE = 1024

model, tokenizer = WcUtils.load(max_batch_size=BATCH_SIZE)

start_time = time.time()

window = []
pos = 0
samples = 0

for token in tokens_gen:
    pos += 1

    window.append(token)

    if len(window) > WINDOW_SIZE:
        window.pop(0)
    
    if last_pos is not None and pos <= last_pos:
        continue

    if len(window) == WINDOW_SIZE:
        # we filled up the winodw, so we can start filling up the batch
        start_time = time.time()
        probs = WcUtils.run_llama_on_tokens(model, window)
        took = time.time() - start_time
        
        suffix_percent = torch.sum(probs * suffix_mask.float()).item()
        
        print(f'{pos}) took {took:.2f} seconds, suffix_percent: {suffix_percent:.6f}')

        if suffix_percent >= 0.99:
            # save this point + probs
            fn = f'{output_folder}/{pos}_{suffix_percent}_probs.txt'
            torch.save(probs, fn)
