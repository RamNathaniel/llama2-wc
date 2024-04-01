import torch

from statistics import mean
from pathlib import Path
from llama import ModelArgs, Transformer, Tokenizer
from typing import Set, Tuple, List

from utils.llama_utils import LlamaUtils


class PhraserUtils:
    DATA_ROOT = f'{LlamaUtils.HOME}/llama2-wc/data' if LlamaUtils.IS_MAC else f'{LlamaUtils.HOME}/suffixes'
    MODELS_FOLDER = f'{LlamaUtils.HOME}/wc-models'

    LAYER_TO_USE = 30

    @staticmethod
    def run_llama_on_tokens(model: Transformer, tokens: List[int]) -> torch.Tensor:
        with torch.no_grad():
            tokens_tensor = torch.unsqueeze(torch.tensor(tokens).long(), 0).cuda()

            model.training = False
            model.layer_output_ind = PhraserUtils.LAYER_TO_USE
            logits = model(tokens_tensor, 0)
            probs = torch.nn.functional.softmax(logits, dim=1)[0, :]
            idea = model.layer_output.no_grad()

        return probs, idea

    @staticmethod
    def run_llama_on_batch(model: Transformer, tokens: List[List[int]]) -> torch.Tensor:
        with torch.no_grad():
            tokens_tensor = torch.tensor(tokens).long().cuda()

            model.training = False
            logits = model(tokens_tensor, 0)
            probs = torch.nn.functional.softmax(logits, dim=1)
        
        return probs
