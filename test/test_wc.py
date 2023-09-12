import torch
from ..word_complete.wc_utils import WcUtils
from ..word_complete.word_completer import WordCompleter
from ..word_complete.batch_gen import BatchGen

def test_wc():
    model = WordCompleter()
    input = torch.rand((32, BatchGen.WC_WINDOW_SIZE), dtype=torch.float).long()
    output = model(input, 0)

    assert output.shape == (32, WcUtils.VOCAB_SIZE), 'output shape is wrong'
    pass