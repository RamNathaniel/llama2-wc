from typing import List
import torch

from utils.llama_utils import LlamaUtils
from ..word_complete.wc_utils import WcUtils
from ..word_complete.word_completer import WordCompleter
from ..word_complete.batch_gen import BatchGen

def test_wc():
    model = WordCompleter()
    input = torch.rand((32, BatchGen.WC_WINDOW_SIZE), dtype=torch.float).long()
    output, indicator = model(input, 0)

    assert output.shape == (32, LlamaUtils.VOCAB_SIZE), 'output shape is wrong'
    assert indicator.shape == (32, 1), 'indicator shape is wrong'
    pass

def test_wc_utils():
    if not LlamaUtils.IS_MAC:
        from llama.tokenizer import Tokenizer
        tokenizer = Tokenizer(LlamaUtils.TOKENIZER_PATH)

        ids: List[int] = []
        for t in WcUtils.PUNCTUATIONS:
            token_ids = tokenizer.encode(t, False, False)
            assert len(token_ids) == 1, 'Punctuation token is not a single token'
            ids.append(token_ids[0])
        
        ids.sort()
        as_list = [t for t in WcUtils.PUNCTUATIONS_IDS]
        as_list.sort()
        assert ids == as_list, 'Punctuations ids are wrong'
