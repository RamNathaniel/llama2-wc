import torch

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

CORPUS = '/home/ram_nathaniel/lib/1984.txt'
DEVICE = 'cuda'

tokenizer = Tokenizer(WcUtils.TOKENIZER_PATH)
suffix_mask = get_suffix_mask(tokenizer, torch.device(DEVICE)).detach()
suffix_count = torch.sum(suffix_mask.float()).item()

tokens_gen = TextfileGen(CORPUS, tokenizer).get_file_tokens()

WINDOW_SIZE = 1024
WC_WINDOW_SIZE = 64

model, tokenizer = WcUtils.load_model()
wc_model = WordCompleter()
wc_model.to(DEVICE)

# Optimizers specified in the torch.optim package
optimizer = torch.optim.SGD(wc_model.parameters(), lr=0.001, momentum=0.9)

start_time = time.time()

total_loss = []
window = []
next_token: int = None
pos = 0
samples = 0
for token in tokens_gen:
    pos += 1
    if next_token is None:
        next_token = token
        continue

    window.append(next_token)
    next_token = token

    if len(window) > WINDOW_SIZE:
        window.pop(0)
    if len(window) == WINDOW_SIZE:
        probs = WcUtils.run_llama_on_tokens(model, window).detach()

        suffix_percent = torch.sum(suffix_mask.float() * probs)

        if suffix_percent < 0.1:
            continue

        if suffix_percent > 0.9:
            # save this point + probs
            fn = f'/home/ram_nathaniel/suffixes/{pos}_{suffix_percent}_probs.txt'
            with open(fn, 'wb') as f:
                torch.save(probs, f)

        samples += 1
        wc_tokens_tensor = torch.unsqueeze(torch.tensor(window[-WC_WINDOW_SIZE:]).long(), 0).cuda().detach()

        optimizer.zero_grad()
        wc_model.zero_grad()
        logits = wc_model.forward(wc_tokens_tensor, 0)
        wc_probs = torch.nn.functional.softmax(logits, dim=1)[0, :]
        loss = wc_loss(
            wc_probs,
            probs.detach(),
            suffix_mask.detach(),
            wc_probs.device)

        loss.backward(retain_graph=True)
        optimizer.step()

        total_loss.append(math.sqrt(loss.item()) * suffix_count)
        if len(total_loss) > 100:
            total_loss.pop(0)

        if samples % 10 == 0:
            txt = tokenizer.decode(window[-6:])
            sec = (time.time() - start_time)
            next_token_str = tokenizer.id_to_piece(next_token)

            print(f'pos: {pos}, '
                + f'samples: {samples}, '
                + f'loss: {torch.sqrt(loss) * suffix_count:.6f}, '
                + f'%suffix: {suffix_percent:.4f}, '
                + f'txt: ...{txt} -> {next_token_str} (+{int(sec)} sec)')

        if samples % 1000 == 0:
            l = mean(total_loss)
            fn = f'/home/ram_nathaniel/checkpoints/wc-model_{pos}_{l}.pth'

            torch.save(wc_model.state_dict(), fn)
            print(f'Saved model to {fn}')
