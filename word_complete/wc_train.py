import torch

import json
from pathlib import Path
from llama import ModelArgs, Transformer, Tokenizer, LLaMA
from typing import Tuple
import time
from textfile_gen import TextfileGen

CHECKPOINT_DIR = '/home/ram_nathaniel/llama/llama-2-7b'
TOKENIZER_PATH = '/home/ram_nathaniel/llama/tokenizer.model'
MAX_SEQ_LEN = 1024
MAX_BATCH_SIZE = 1

DEVICE = 'cuda'

CORPUS = '/home/ram_nathaniel/lib/1984.txt'

tokenizer = Tokenizer(TOKENIZER_PATH)
suffix_mask = wc_loss.get_suffix_mask(tokenizer, torch.device(DEVICE))

token_gen = TextfileGen(CORPUS, tokenizer).get_file_tokens()


def load(
    ckpt_dir: str,
    tokenizer_path: str,
    local_rank: int,
    world_size: int,
    max_seq_len: int,
    max_batch_size: int,
    verbose: bool = False,
) -> Tuple[LLaMA, Transformer, Tokenizer]:
    checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
    assert world_size == len(
        checkpoints
    ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {world_size}"
    ckpt_path = checkpoints[local_rank]

    if verbose:
        print(f'starting loading {ckpt_path}')
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    if verbose:
        print('done')

    with open(Path(ckpt_dir) / "params.json", "r") as f:
        params = json.loads(f.read())

    model_args: ModelArgs = ModelArgs(
        max_seq_len=max_seq_len, max_batch_size=max_batch_size, **params
    )

    tokenizer = Tokenizer(model_path=tokenizer_path)
    if verbose:
        print('Tokenizer loaded')

    model_args.vocab_size = tokenizer.n_words
    torch.set_default_tensor_type(torch.cuda.HalfTensor)

    if verbose:
        print('Creating Transformer')

    model = Transformer(model_args)
    torch.set_default_tensor_type(torch.FloatTensor)
    model.load_state_dict(checkpoint, strict=False)

    if verbose:
        print('Creating LLaMA model')

    generator = LLaMA(model, tokenizer)

    if verbose:
        print('done')

    return generator, model, tokenizer

def run_base() -> None:
    local_rank = 0
    world_size = 1

    start_time = time.time()
    
    generator, model, tokenizer = load(
        CHECKPOINT_DIR,
        TOKENIZER_PATH,
        local_rank,
        world_size,
        MAX_SEQ_LEN,
        MAX_BATCH_SIZE,
        True)

    print(f'Loading took {time.time()-start_time:.2f} seconds')

    # for debugging
    words = [tokenizer.id_to_piece(id) for id in range(32000)]

    MAX_TO_PRINT = 20

    while True:
        text = input('Text to complete:')
        start_time = time.time()
        tokens = tokenizer.encode(text, bos=True, eos=False)
        tokens_tensor = torch.unsqueeze(torch.tensor(tokens).long(), 0).cuda()
        logits = model(tokens_tensor, 0)
        stop_time = time.time()

        probs = torch.nn.functional.softmax(logits)[0, :].cpu()
        sorted_probs, sorted_ind = torch.sort(probs, descending=True)

        to_print = MAX_TO_PRINT
        
        for ind in range(MAX_TO_PRINT):
            print(f'{sorted_probs[ind]:.2f} - {words[sorted_ind[ind]]}')

        # print(logits)

        print(f'Took: {stop_time - start_time:.2f} seconds')

    return

def run(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.8,
    top_p: float = 0.95,
    max_seq_len: int = 1024,
    max_batch_size: int = 1,
):
    local_rank = 0
    world_size = 1

    print('Loading')

    generator, model, tokenizer = load(
        ckpt_dir, tokenizer_path, local_rank, world_size, max_seq_len, max_batch_size, True
    )
    prompts = [
        # For these prompts, the expected answer is the natural continuation of the prompt
        "I believe the meaning of life is",  # removed: keep only one prompt
    ]
    while True:
        print("Prompt:", prompts)
        results = generator.generate(
            prompts, max_gen_len=256, temperature=temperature, top_p=top_p
        )
        for result in results:
            print("🦙LLaMA:", result.strip())

        user_input = input("please enter your prompts (Ctrl+C to exit): ")
        prompts = [user_input]


def get_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", type=str, default="/llama_data/7B")
    parser.add_argument(
        "--tokenizer_path", type=str, default="/llama_data/tokenizer.model"
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_base()
    # args = get_args()
    # run(
    #     ckpt_dir=args.ckpt_dir,
    #     tokenizer_path=args.tokenizer_path,
    #     temperature=0.8,
    #     top_p=0.95,
    #     max_seq_len=1024,
    #     max_batch_size=1,
    # )
