import torch
from torch import nn

from llama.model_single import ModelArgs, RMSNorm, TransformerBlock, precompute_freqs_cis
from utils.llama_utils import LlamaUtils

"""
Phraser takes an internal (near end) representation of a sentence and predicts
the next word in the sentence. It then takes the next word and predicts the next
word after that, and so on. It is used to generate text.

It contains 3 main components:
- An extractor to get the prompt understanding and idea forming from the LLM
- A generator (called Phraser) to generate the next word
- A loopback to feed the generated word back in with the extractor ouptut
"""

class Phraser(torch.nn.Module):
    """
    Phraser takes an internal (near end) representation of a sentence and predicts
    the next word in the sentence. It then takes the next word and predicts the next
    word after that, and so on. It is used to generate text.

    Args:
        torch (_type_): _description_
    """
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = nn.Linear(params.dim, params.vocab_size, bias=False)

        self.freqs_cis = precompute_freqs_cis(
            self.params.dim // self.params.n_heads, self.params.max_seq_len * 2
        )

        pass


    def forward(self, idea: torch.Tensor, start_pos: int) -> torch.Tensor:
        h = idea  # back to LlaMA naming
        seqlen = h.shape[1]

        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

        mask = None # we can use any word at any position
        
        # if seqlen > 1:
        #     mask = torch.full(
        #         (1, 1, seqlen, seqlen), float("-inf"), device=tokens.device
        #     )
        #     mask = torch.triu(mask, diagonal=start_pos + 1).type_as(h)

        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)

        h = self.norm(h)
        output = self.output(h[:, -1, :])  # only compute last logits
        return output.float()


class Extractor(torch.nn.Module):
    """
    Extractor takes the internal representation of a sentence and creates the
    idea tensor to feed into the phraser.

    Nomrally it should be the identity operator.
    One day we will also try to improve the internal representation.

    Args:
        torch (_type_): _description_
    """
    def __init__(self):
        super(Extractor, self).__init__()

    def forward(self, idea: torch.Tensor) -> torch.Tensor:
        return idea
    
class Generator(torch.nn.Module):
    """
    Generator takes the internal representation of a sentence and predicts
    the next word in the sentence.

    For simplicity we use the last X layers of the LLM as the initial architecture
    and weights, and then fine-tune it on the generator task.

    Args:
        torch (_type_): _description_
    """
    def __init__(self):
        super(Generator, self).__init__()

    def forward(self, idea: torch.Tensor) -> torch.Tensor:
        return idea
    
class Loopback(torch.nn.Module):
    """
    Loopback takes the chosen token and adds it to the "idea" fed into the generator.
    We then use the generator to generate the next token.
    
    We use an embedding to convert the token to the same dimension as the idea.
    We then "push left" the idea and concatenate the token to the right.

    Args:
        torch (_type_): _description_
    """
    def __init__(self, idea_dim: int):
        super(Loopback, self).__init__()
        self.idea_dim = idea_dim
        self.emb = torch.nn.Embedding(LlamaUtils.VOCAB_SIZE, idea_dim)

    def forward(self, idea: torch.Tensor, token: int) -> torch.Tensor:
        token = torch.tensor([token])
        embedded_token = self.emb(token)
        b, l, d = idea.shape
        embedded_token = embedded_token.unsqueeze(0).expand(b, 1, d)
        max_len = min(LlamaUtils.CONTEXT_WINDOW, l+1)
        return torch.cat([idea, embedded_token], dim=1)[:, -max_len:, :]
