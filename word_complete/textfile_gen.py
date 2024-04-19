import os
from llama.tokenizer import Tokenizer
from utils.llama_utils import LlamaUtils


class TextfileGen:
    """
    Read a text file one token at a time
    """
    def __init__(self, filename: str):
        self.filename = filename

        if not LlamaUtils.IS_MAC:
            from llama.tokenizer import Tokenizer
            self.tokenizer = Tokenizer(LlamaUtils.TOKENIZER_PATH)
        else:
            self.tokenizer = None

        if self.tokenizer is None:
            # This file must have been pre-tokenized
            file_name, old_extension = os.path.splitext(self.filename)

            if old_extension == '.txt':
                self.filename = file_name + '.tokens'
        
        assert os.path.exists(self.filename), 'filename must end with .tokens'
    
    def get_file_tokens(self):
        with open(self.filename, 'r') as f:
            for line in f:
                if self.tokenizer is not None:
                    for t in self.tokenizer.encode(line, bos=False, eos=False):
                        yield t
                else:
                    # This file must have been pre-tokenized
                    yield int(line.strip())


if __name__ == '__main__':
    import sys

    if LlamaUtils.IS_MAC:
        print('This script is not supported on Mac')
        sys.exit(1)

    if len(sys.argv) != 3:
        print(f'Usage: {sys.argv[0]} <text file> <output file>')
        sys.exit(1)

    input_fn = sys.argv[1]
    output_fn = sys.argv[2]

    t = Tokenizer('/home/ram_nathaniel/pyllama_data/tokenizer.model')
    gen = TextfileGen(input_fn, t)

    print(f'Reading tokens from {input_fn} and writing to {output_fn}')

    tokens = [tok for tok in gen.get_file_tokens()]

    with open(output_fn, 'wt') as f:
        for token in tokens:
            f.write(f'{token}\n')

    print('Done')
    pass
