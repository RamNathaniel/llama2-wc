import torch

import json
from pathlib import Path
from llama import ModelArgs, Transformer, Tokenizer, LLaMA
from typing import Tuple, List
import time
from word_complete.textfile_gen import TextfileGen
from word_complete.wc_loss import wc_loss, get_suffix_mask
from word_complete.word_completer import WordCompleter

CHECKPOINT_DIR = '/home/ram_nathaniel/llama/llama-2-7b'
TOKENIZER_PATH = '/home/ram_nathaniel/llama/tokenizer.model'
MAX_SEQ_LEN = 1024
MAX_BATCH_SIZE = 1

DEVICE = 'cuda'

CORPUS = '/home/ram_nathaniel/lib/1984.txt'

tokenizer = Tokenizer(TOKENIZER_PATH)
suffix_mask = get_suffix_mask(tokenizer, torch.device(DEVICE))

tokens_gen = TextfileGen(CORPUS, tokenizer).get_file_tokens()

def load_model() -> Tuple[Transformer, Tokenizer]:

    start_time = time.time()
    
    model, tokenizer = load()

    print(f'Loading took {time.time()-start_time:.2f} seconds')

    return model, tokenizer

def load(
    local_rank: int = 0,
    world_size: int = 1,
    max_seq_len: int = MAX_SEQ_LEN,
    max_batch_size: int = MAX_BATCH_SIZE,
    verbose: bool = False,
) -> Tuple[Transformer, Tokenizer]:
    checkpoints = sorted(Path(CHECKPOINT_DIR).glob("*.pth"))
    assert world_size == len(
        checkpoints
    ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {world_size}"
    ckpt_path = checkpoints[local_rank]

    if verbose:
        print(f'starting loading {ckpt_path}')
    
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    if verbose:
        print('done')

    with open(Path(CHECKPOINT_DIR) / "params.json", "r") as f:
        params = json.loads(f.read())

    model_args: ModelArgs = ModelArgs(
        max_seq_len=max_seq_len, max_batch_size=max_batch_size, **params
    )

    tokenizer = Tokenizer(model_path=TOKENIZER_PATH)
    if verbose:
        print('Tokenizer loaded')

    model_args.vocab_size = tokenizer.n_words
    torch.set_default_tensor_type(torch.cuda.HalfTensor)

    if verbose:
        print('Creating Transformer')

    model = Transformer(model_args)
    torch.set_default_tensor_type(torch.FloatTensor)
    model.load_state_dict(checkpoint, strict=False)

    return model, tokenizer

def run_llama_on_tokens(model: Transformer, tokens: List[int]) -> torch.Tensor:
    tokens_tensor = torch.unsqueeze(torch.tensor(tokens).long(), 0).cuda()
    logits = model(tokens_tensor, 0)
    probs = torch.nn.functional.softmax(logits)[0, :]

    return probs

WINDOW_SIZE = 1024
WC_WINDOW_SIZE = 64

model, tokenizer = load_model()
wc_model = WordCompleter()
wc_model.to(DEVICE)

# Optimizers specified in the torch.optim package
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

window = []
pos = 0
for token in tokens_gen:
    pos += 1
    window.append(token)
    if len(window) > WINDOW_SIZE:
        window.pop(0)
    if len(window) == WINDOW_SIZE:
        optimizer.zero_grad()

        probs = run_llama_on_tokens(model, window)

        logits = wc_model.forward(torch.unsqueeze(torch.tensor(window).long(), 0).cuda(), -WC_WINDOW_SIZE)
        wc_probs = torch.nn.functional.softmax(logits)[0, :]
        loss = wc_loss(wc_probs, probs, suffix_mask, wc_probs.device)

        loss.backward()
        optimizer.step()

        if pos % 1 == 0:
            print(f'pos: {pos}, loss: {loss:.4f}, %suffix: {torch.sum(suffix_mask.float() * probs):.4f}')
