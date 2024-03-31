from typing import List
import torch
from utils.llama_utils import LlamaUtils
from phraser.phraser_utils import PhraserUtils
from phraser.phraser import Phraser, Loopback
from word_complete.batch_gen import BatchGen

def test_loopback():
    loopback = Loopback(16)
    idea: torch.Tensor = torch.rand([1, 1024, 16])

    new_idea = loopback(idea, 5)
    assert new_idea.shape == (1, 1025, 16)

    idea = torch.run([1, LlamaUtils.CONTEXT_WINDOW, 16])
    new_idea = loopback(idea, 45)
    assert new_idea.shape == (1, LlamaUtils.CONTEXT_WINDOW, 16)
    pass

def test_phaser_tokens():
    model, tokenizer = LlamaUtils.load_model()
    tokens = tokenizer.encode('The quick brown fox jumps over the lazy dog', False, False)
    probs = PhraserUtils.run_llama_on_tokens(model, tokens)
    assert probs.shape == (tokenizer.n_words, LlamaUtils.VOCAB_SIZE,)

def test_wc_utils():
    if not LlamaUtils.IS_MAC:
        from llama.tokenizer import Tokenizer
        tokenizer = Tokenizer(LlamaUtils.TOKENIZER_PATH)

        ids: List[int] = []
        for t in LlamaUtils.PUNCTUATIONS:
            token_ids = tokenizer.encode(t, False, False)
            assert len(token_ids) == 1, 'Punctuation token is not a single token'
            ids.append(token_ids[0])
        
        ids.sort()
        as_list = [t for t in LlamaUtils.PUNCTUATIONS_IDS]
        as_list.sort()
        assert ids == as_list, 'Punctuations ids are wrong'
