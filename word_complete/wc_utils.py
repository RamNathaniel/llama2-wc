import platform
import torch

import json
import math
from statistics import mean
from pathlib import Path
from llama import ModelArgs, Transformer, Tokenizer, LLaMA
from typing import Tuple, List
import time


class WcUtils:
    IS_WINDOWS = platform.system() == 'Windows'
    IS_MAC = platform.system() == 'Darwin'
    HAS_GPU = torch.cuda.is_available()

    HOME = '/Users/ramnathaniel' if IS_MAC else '/home/ram_nathaniel'
    DATA_ROOT = f'{HOME}/llama2-wc/data' if IS_MAC else f'{HOME}/suffixes'

    MODELS_FOLDER = f'{HOME}/wc-models'

    CHECKPOINT_DIR = '/home/ram_nathaniel/llama/llama-2-7b'
    TOKENIZER_PATH = '/home/ram_nathaniel/llama/tokenizer.model'
    MAX_SEQ_LEN = 1024
    MAX_BATCH_SIZE = 1

    VOCAB_SIZE = 32000

    @staticmethod
    def load_model() -> Tuple[Transformer, Tokenizer]:

        start_time = time.time()
        
        model, tokenizer = WcUtils.load()

        print(f'Loading took {time.time()-start_time:.2f} seconds')

        return model, tokenizer

    @staticmethod
    def load(
        local_rank: int = 0,
        world_size: int = 1,
        max_seq_len: int = MAX_SEQ_LEN,
        max_batch_size: int = MAX_BATCH_SIZE,
        verbose: bool = False,
    ) -> Tuple[Transformer, Tokenizer]:
        checkpoints = sorted(Path(WcUtils.CHECKPOINT_DIR).glob("*.pth"))
        assert world_size == len(
            checkpoints
        ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {world_size}"
        ckpt_path = checkpoints[local_rank]

        if verbose:
            print(f'starting loading {ckpt_path}')
        
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        if verbose:
            print('done')

        with open(Path(WcUtils.CHECKPOINT_DIR) / "params.json", "r") as f:
            params = json.loads(f.read())

        model_args: ModelArgs = ModelArgs(
            max_seq_len=max_seq_len, max_batch_size=max_batch_size, **params
        )

        tokenizer = Tokenizer(model_path=WcUtils.TOKENIZER_PATH)
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

    @staticmethod
    def run_llama_on_tokens(model: Transformer, tokens: List[int]) -> torch.Tensor:
        with torch.no_grad():
            tokens_tensor = torch.unsqueeze(torch.tensor(tokens).long(), 0).cuda()

            model.training = False
            logits = model(tokens_tensor, 0)
            probs = torch.nn.functional.softmax(logits, dim=1)[0, :]

        return probs

    @staticmethod
    def run_llama_on_batch(model: Transformer, tokens: List[List[int]]) -> torch.Tensor:
        with torch.no_grad():
            tokens_tensor = torch.tensor(tokens).long().cuda()

            model.training = False
            logits = model(tokens_tensor, 0)
            probs = torch.nn.functional.softmax(logits, dim=1)
        
        return probs
