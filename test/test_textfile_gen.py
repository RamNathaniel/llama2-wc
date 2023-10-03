import torch
from ..word_complete.textfile_gen import TextfileGen
from ..word_complete.batch_gen import BatchGen
from ..word_complete.wc_utils import WcUtils


def test_textfile_gen():
    tg = TextfileGen('test/text_file_to_tokenize.txt')

    tokens = [token for token in tg.get_file_tokens()]
    
    assert len(tokens) > 0, 'No tokens found in text file'
    assert tokens[0] != 2, 'First token is <s>'
    assert tokens[-1] != 3, 'Last token is </s>'

    pass

def test_batch_gen():
    CORPUS = f'{WcUtils.DATA_ROOT}/lib/1984.txt'
    SUFFIXES_FOLDER = f'{WcUtils.DATA_ROOT}/1984.txt'

    BATCH_SIZE = 32    

    textfile_gen = TextfileGen(CORPUS)

    def on_batch(
        epoch: int,
        batch: int,
        tokens: torch.Tensor,
        indicators: torch.Tensor,
        llama_probs: torch.Tensor,
        context):
    
        assert tokens.shape == (BATCH_SIZE, BatchGen.WC_WINDOW_SIZE), 'tokens shape is wrong'
        assert indicators.shape == (BATCH_SIZE, 1), 'indicators shape is wrong'
        assert llama_probs.shape == (BATCH_SIZE, WcUtils.VOCAB_SIZE), 'llama_probs shape is wrong'

        bc = context['batch_counter']
        ec = context['epoch_counter']

        assert ec == epoch, f'epoch number is wrong, expected: {ec}, actual: {epoch}'
        assert bc == batch, f'batch number is wrong, expected: {bc}, actual: {batch}'

        context['batch_counter'] += 1
        pass

    def on_epoch(epoch: int, context):
        ec = context['epoch_counter']

        assert ec == epoch, f'epoch number is wrong, expected: {ec}, actual: {epoch}'
        context['epoch_counter'] += 1
        context['batch_counter'] = 0
        pass

    batch_gen_train = BatchGen(textfile_gen, SUFFIXES_FOLDER, BATCH_SIZE, 'cpu', on_batch, on_epoch)

    llama_probs = batch_gen_train.get_llama_probs(54280)
    assert llama_probs.shape == torch.Size((WcUtils.VOCAB_SIZE,)), 'llama_probs shape is wrong'

    context = { 'batch_counter': 0, 'epoch_counter': 0 }
    batch_gen_train.run(epochs=2, batches=5, context=context)

    pass