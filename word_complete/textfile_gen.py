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


if __name__ == '__main__':
    import sys

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
