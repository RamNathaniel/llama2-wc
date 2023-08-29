from llama.tokenizer import Tokenizer

class TextfileGen:
    """
    Read a text file one token at a time
    """
    def __init__(self, filename: str, tokenizer: Tokenizer):
        self.filename = filename
        self.tokenizer = tokenizer
    
    def get_file_tokens(self):
        with open(self.filename, 'r') as f:
            for line in f:
                for t in self.tokenizer.encode(line, bos=False, eos=False):
                    yield t
