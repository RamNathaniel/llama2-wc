import os
import torch

from llama import Tokenizer
from word_complete.textfile_gen import TextfileGen

from word_complete.wc_utils import WcUtils

class BatchGen:
    WINDOW_SIZE = 1024
    WC_WINDOW_SIZE = 64

    def __init__(
            self,
            corpus: str,
            suffix_folder: str,
            batch_size: int,
            on_batch: callable[torch.Tensor, torch.Tensor],
            on_epoch: callable):
        
        self.corpus = corpus
        self.suffix_folder = suffix_folder
        self.batch_size = batch_size

        self.tokenizer = Tokenizer(WcUtils.TOKENIZER_PATH)        

        self.on_epoch = on_epoch
        self.on_batch = on_batch

        self.files = os.listdir(self.suffix_folder)
        pass

    def has_file(self, pos: int) -> bool:
        return any([fn.startswith(f'{self.suffix_folder}/{pos}_') for fn in self.files])

    def run(self, epochs: int):
        if epochs < 0:
            while True:
                self.run_epoch()
        else:
            for epoch in range(epochs):
                self.run_epoch()

        pass

    def run_epoch(self):
        window = []  # window for llama (len=WINDOW_SIZE)
        inputs = []  # windows for wc BATCH_SIZE x WC_WINDOW_SIZE
        labels = []  # labels for wc (len=BATCH_SIZE)

        self.tokens_gen = TextfileGen(self.corpus, self.tokenizer).get_file_tokens()

        pos = 0
        batch = 0
        for token in self.tokens_gen:
            pos += 1
            if next_token is None:
                next_token = token
                continue

            window.append(next_token)
            next_token = token

            if len(window) > BatchGen.WINDOW_SIZE:
                window.pop(0)
            if len(window) == BatchGen.WINDOW_SIZE:
                label = self.has_file(pos)
                input = window[-BatchGen.WC_WINDOW_SIZE:]

                labels += label
                inputs += input

                if len(inputs) == self.batch_size:
                    # we filled up the winodw, so we can start filling up the batch
                    wc_tokens_tensor = torch.tensor(inputs).long().detach()
                    wc_labels_tensor = torch.tensor(labels).long().detach()

                    if self.on_batch is not None:
                        self.on_batch(wc_tokens_tensor, wc_labels_tensor)

                    inputs = []
                    labels = []

                    batch += 1

        if self.on_epoch is not None:
            self.on_epoch()
        
        pass
