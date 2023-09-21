import io
import os
import torch

from typing import Callable
from llama import Tokenizer
from .textfile_gen import TextfileGen
from .wc_utils import WcUtils


class BatchGen:
    WINDOW_SIZE = 1024
    WC_WINDOW_SIZE = 64


    def __init__(
            self,
            corpus: str,
            suffix_folder: str,
            batch_size: int,
            on_batch: Callable[[int, int, torch.Tensor, torch.Tensor], None],
            on_epoch: Callable[[int], None]):
        
        self.corpus = corpus
        self.suffix_folder = suffix_folder
        self.batch_size = batch_size

        self.tokenizer = Tokenizer(WcUtils.TOKENIZER_PATH)        

        self.on_epoch = on_epoch
        self.on_batch = on_batch

        self.files = os.listdir(self.suffix_folder)
        pass


    def has_file(self, pos: int) -> bool:
        # return any([fn.startswith(f'{self.suffix_folder}/{pos}_') for fn in self.files])
        return self.get_file(pos) is not None


    def get_file(self, pos: int) -> str:
        return next((fn for fn in self.files if fn.startswith(f'{pos}_')), None)


    def get_llama_probs(self, pos: int) -> torch.Tensor:
        fn = self.get_file(pos)
        if fn is None:
            return None
        
        full_path = os.path.join(self.suffix_folder, fn)
        probs = torch.load(full_path)
        return probs

    def run(self, epochs: int, batches: int, context):
        if epochs < 0:
            epoch = 0
            while True:
                self.run_epoch(epoch, batches, context)
                epoch += 1
        else:
            for epoch in range(epochs):
                self.run_epoch(epoch, batches, context)

        pass

    def run_epoch(self, epoch: int, batches: int = -1, context = None):
        window = []  # window for llama (len=WINDOW_SIZE)
        inputs = []  # windows for wc BATCH_SIZE x WC_WINDOW_SIZE
        indicators = []  # labels for wc (len=BATCH_SIZE)
        probs_batch = []  # probs for llama (len=BATCH_SIZE x VOCAB_SIZE)

        self.tokens = [t for t in TextfileGen(self.corpus, self.tokenizer).get_file_tokens()]

        pos = 0
        batch = 0
        next_token = None
        for token in self.tokens:
            pos += 1
            if next_token is None:
                next_token = token
                continue

            window.append(next_token)
            next_token = token

            if len(window) > BatchGen.WINDOW_SIZE:
                window.pop(0)
            
            if len(window) == BatchGen.WINDOW_SIZE:
                indicator = self.has_file(pos)
                try:
                    prs = self.get_llama_probs(pos)
                except Exception as e:
                    print(f'Error getting probs for pos: {pos} - skip it!')
                    window.pop(0)
                    continue
                
                input = window[-BatchGen.WC_WINDOW_SIZE:]

                indicators.append(1 if indicator else 0)
                inputs.append(input)
                probs_batch.append(prs if prs is not None else torch.zeros((WcUtils.VOCAB_SIZE,), dtype=torch.float, device=torch.device('cuda:0')))

                if len(inputs) == self.batch_size:
                    # we filled up the winodw, so we can start filling up the batch
                    wc_tokens_tensor = torch.tensor(inputs).long().detach()
                    wc_indicators_tensor = torch.tensor(indicators).unsqueeze(dim=1).long().detach()
                    wc_probs_tensor = torch.stack([pb.to('cpu') for pb in probs_batch], dim=0).float().detach()

                    if self.on_batch is not None:
                        self.on_batch(epoch, batch, wc_tokens_tensor, wc_indicators_tensor, wc_probs_tensor, context)

                    inputs = []
                    indicators = []
                    probs_batch = []

                    batch += 1
                    if batches > 0 and batch >= batches:
                        break

        if self.on_epoch is not None:
            self.on_epoch(epoch, context)
        
        pass
