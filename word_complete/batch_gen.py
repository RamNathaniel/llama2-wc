import io
import os
import torch
import numpy as np

from typing import Callable
from .textfile_gen import TextfileGen
from .wc_utils import WcUtils


class BatchGen:
    WINDOW_SIZE = 1024
    WC_WINDOW_SIZE = 64

    def __init__(
            self,
            token_gen: TextfileGen,
            suffix_folder: str,
            batch_size: int,
            device,
            on_batch: Callable[[int, int, torch.Tensor, torch.Tensor], None],
            on_epoch: Callable[[int], None]):
        
        self.suffix_folder = suffix_folder
        self.token_gen = token_gen
        self.batch_size = batch_size

        self.on_epoch = on_epoch
        self.on_batch = on_batch
        self.device = device

        self.tokens = [t for t in self.token_gen.get_file_tokens()]

        # Only positions with files have a token that we want to predict.
        # So we keep a set of all positions with files, and then we can check
        # if a position has a file in O(1) instead of O(n).

        # Note: we assume that the files are named <pos>_<suffix>.txt
        # Note2: If we want to address a more limited mask of suffixes, we need
        # to change the way we check if a position has a file, i.e., look at the
        # next token and decide. We can only limit the targetted suffixes if we
        # want to make use of the database.
        self.files: list[str] = os.listdir(self.suffix_folder)
        self.pos_with_files = set([int(fn.split('_')[0]) for fn in self.files])

        self.inputs_np: np.ndarray = None  # Inputs for the batch for wc model (/classifier) 
        self.tokens_np: np.array = None  # All tokens in a numpy array
        self.indicators_np: np.ndarray = None  # Indicators for the batch for wc model (/classifier)

        pass


    def has_file(self, pos: int) -> bool:
        # return any([fn.startswith(f'{self.suffix_folder}/{pos}_') for fn in self.files])
        # return self.get_file(pos) is not None
        return pos in self.pos_with_files

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

    def _empty_batch(self) -> torch.Tensor:
        return torch.zeros((WcUtils.VOCAB_SIZE,), dtype=torch.float, device=self.device)

    def _set_inputs_for_pos(self, pos: int):
        for p in range(pos-self.batch_size, pos):
            self.inputs_np[p - pos + self.batch_size, :] = self.tokens_np[p - BatchGen.WC_WINDOW_SIZE:p]
            self.indicators_np[p - pos + self.batch_size] = 1 if self.has_file(p + 1) else 0

    def run_epoch(self, epoch: int, batches: int = -1, context = None):
        window = []  # window for llama (len=WINDOW_SIZE)
        inputs = []  # windows for wc BATCH_SIZE x WC_WINDOW_SIZE
        indicators = []  # labels for wc (len=BATCH_SIZE)
        probs_batch = []  # probs for llama (len=BATCH_SIZE x VOCAB_SIZE)

        self.inputs_np = np.zeros((self.batch_size, BatchGen.WC_WINDOW_SIZE), dtype=np.int32)
        self.tokens_np = np.array(self.tokens, dtype=np.int32)
        self.indicators_np = np.zeros((self.batch_size,), dtype=np.int32)

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

                # try:
                #     prs = self.get_llama_probs(pos)
                # except Exception as e:
                #     print(f'Error getting probs for pos: {pos} - skip it!')
                #     window.pop(0)
                #     continue
                prs = None
                
                input = window[-BatchGen.WC_WINDOW_SIZE:]

                indicators.append(1 if indicator else 0)
                inputs.append(input)
                probs_batch.append(prs if prs is not None else self._empty_batch())

                if len(inputs) == self.batch_size:
                    # we filled up the winodw, so we can start filling up the batch
                    # wc_tokens_tensor = torch.tensor(inputs).long().detach()
                    # wc_indicators_tensor = torch.tensor(indicators).unsqueeze(dim=1).long().detach()
                    # wc_probs_tensor = torch.stack([pb.to('cpu') for pb in probs_batch], dim=0).float().detach()

                    self._set_inputs_for_pos(pos)

                    wc_tokens_tensor = torch.tensor(self.inputs_np).long().detach()
                    wc_indicators_tensor = torch.tensor(self.indicators_np).unsqueeze(dim=1).long().detach()
                    wc_probs_tensor = torch.zeros((self.batch_size, WcUtils.VOCAB_SIZE), dtype=torch.float, device=self.device).detach()

                    # verify that inputs_np == inputs and indicators_np == indicators
                    # for i in range(self.batch_size):
                    #     assert np.array_equal(self.inputs_np[i, :], inputs[i])
                    #     assert self.indicators_np[i] == indicators[i]

                    if self.on_batch is not None:
                        self.on_batch(epoch, batch, wc_tokens_tensor, wc_indicators_tensor, wc_probs_tensor, context)

                    del wc_tokens_tensor, wc_indicators_tensor, wc_probs_tensor
                    
                    inputs = []
                    indicators = []
                    probs_batch = []

                    batch += 1
                    if batches > 0 and batch >= batches:
                        break

        if self.on_epoch is not None:
            self.on_epoch(epoch, context)
        
        pass
