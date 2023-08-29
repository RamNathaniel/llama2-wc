from llama.tokenizer import Tokenizer

t = Tokenizer('/home/ram_nathaniel/pyllama_data/tokenizer.model')

suffixes = [t.id_to_piece(id) for id in range(32000) if t.is_suffix(id)]
print(suffixes)

while True:
    text = input('Enter text: ')
    ids = t.encode(text, bos=True, eos=True)
    words = [t.id_to_piece(id) for id in ids]

    print(ids)
    print(words)
