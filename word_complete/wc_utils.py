import torch

from statistics import mean
from pathlib import Path
from llama import ModelArgs, Transformer, Tokenizer, LLaMA
from typing import Set, Tuple, List
from utils.llama_utils import LlamaUtils


class WcUtils:
    HOME = '/Users/ramnathaniel' if LlamaUtils.IS_MAC else '/home/ram_nathaniel'
    DATA_ROOT = f'{HOME}/llama2-wc/data' if LlamaUtils.IS_MAC else f'{HOME}/suffixes'

    MODELS_FOLDER = f'{HOME}/wc-models'

    PUNCTUATIONS: List[str] = ['.', ',', '!', '?', ';', ':', '(', ')', '[', ']', '{', '}', '<', '>']
    PUNCTUATIONS_IDS: Set[int] = [313, 426, 500, 518, 529, 584, 869, 1405, 1577, 1723, 1738, 1919, 2056, 4514]


    @staticmethod
    def get_suffix_mask(t: Tokenizer, device: torch.device) -> torch.Tensor:
        suffix_mask = torch.zeros(t.n_words, dtype=torch.bool, device=device)
        for id in range(t.n_words):
            suffix_mask[id] = t.is_suffix(id)
        
        return suffix_mask

    @staticmethod
    def get_puctuations_mask(t: Tokenizer, device: torch.device) -> torch.Tensor:
        punctuations_mask = torch.zeros(t.n_words, dtype=torch.bool, device=device)
        for id in range(t.n_words):
            punctuations_mask[id] = t.id_to_piece(id) in WcUtils.PUNCTUATIONS
        
        return punctuations_mask
    
    @staticmethod
    def get_puctuations_ids(t: Tokenizer) -> Set[int]:
        return set(t.piece_to_id(p) for p in WcUtils.PUNCTUATIONS)
