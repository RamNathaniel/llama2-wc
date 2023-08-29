from llama.tokenizer import Tokenizer
from pyllama.word_complete.textfile_gen import TextfileGen

TOKENIZER_PATH = '/home/ram_nathaniel/llama/tokenizer.model'

def test_textfile_gen():
    t = Tokenizer(TOKENIZER_PATH)
    tg = TextfileGen('test/text_file_to_tokenize.txt', t)

    tokens = [token for token in tg.get_file_tokens()]
    assert len(tokens) > 0, 'No tokens found in text file'
    assert tokens[0] != 2, 'First token is <s>'
    assert tokens[-1] != 3, 'Last token is </s>'
