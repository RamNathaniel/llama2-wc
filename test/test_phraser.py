from typing import List
import torch
from utils.llama_utils import LlamaUtils
from phraser.phraser_utils import PhraserUtils
from phraser.phraser import Phraser, Loopback
from word_complete.batch_gen import BatchGen

def test_loopback():
    loopback = Loopback(16)
    idea: torch.Tensor = torch.rand([1, 1024, 16], dtype=torch.float32)

    new_idea = loopback(idea, 5)
    assert new_idea.shape == (1, 1025, 16)

    idea = torch.rand([1, LlamaUtils.CONTEXT_WINDOW, 16], dtype=torch.float32)
    new_idea = loopback(idea, 45)
    assert new_idea.shape == (1, LlamaUtils.CONTEXT_WINDOW, 16)
    pass

def test_phaser_tokens():
    model, tokenizer = LlamaUtils.load_model()
    tokens = tokenizer.encode('The quick brown fox jumps over the lazy dog', False, False)
    probs, idea = PhraserUtils.run_llama_on_tokens(model, tokens)
    assert probs.shape == (LlamaUtils.VOCAB_SIZE,)
    print(idea.shape)
    assert idea.shape == (1, len(tokens), LlamaUtils.EMB_DIM)
