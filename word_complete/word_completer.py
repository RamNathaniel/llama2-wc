import torch
from torch import nn

from llama.model_single import ModelArgs, RMSNorm, TransformerBlock, precompute_freqs_cis
from utils.llama_utils import LlamaUtils

class WordCompleter(torch.nn.Module):
    @staticmethod
    def should_use_word_complete(tokens_tensor: torch.Tensor) -> torch.Tensor:
        """
        Should we run the word completion model?

        Args:
            tokens_tensor (torch.Tensor): B x T tensor of token IDs

        Returns:
            torch.Tensor: B tensor of 1s and 0s
        """
        # The plan:
        # 1. Get the last token in each sequence
        # 2. Check if it appeared in the list of prefixes
        # 3. If it did, return 1, else return 0
        last_token = tokens_tensor[:, -1]
        return (torch.count_nonzero(tokens_tensor == last_token, dim=1) > 1).int()

    @staticmethod
    def get_model_args() -> ModelArgs:
        return ModelArgs(
            dim=64,  # much smaller context.
            n_layers=1,
            n_heads=8,
            vocab_size=LlamaUtils.VOCAB_SIZE,
            multiple_of=64,
            norm_eps=1e-5,
            max_batch_size=128,
            max_seq_len=1024,
        )

    def __init__(self):
        super().__init__()

        params = WordCompleter.get_model_args()

        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = nn.Linear(params.dim, params.vocab_size, bias=False)

        self.indicator = nn.Linear(params.dim, 1, bias=False)

        self.freqs_cis = precompute_freqs_cis(
            self.params.dim // self.params.n_heads, self.params.max_seq_len * 2
        )

    def forward(self, tokens: torch.Tensor, start_pos: int):
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        self.freqs_cis = self.freqs_cis.to(h.device)
        if start_pos + seqlen == 0:
            freqs_cis = self.freqs_cis[start_pos:]
        else:
            freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

        mask = None
        if seqlen > 1:
            mask = torch.full(
                (1, 1, seqlen, seqlen), float("-inf"), device=tokens.device
            )
            mask = torch.triu(mask, diagonal=start_pos + 1).type_as(h)

        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)
        h = self.norm(h)
        output = self.output(h[:, -1, :])  # only compute last logits
        indicator = self.indicator(h[:, -1, :])
        return output.float(), indicator.float()
