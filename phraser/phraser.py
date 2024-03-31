from typing import List
import torch

from utils.llama_utils import LlamaUtils


class Phraser(torch.nn.Module):
    """
    Phraser takes an internal (near end) representation of a sentence and predicts
    the next word in the sentence. It then takes the next word and predicts the next
    word after that, and so on. It is used to generate text.

    It contains 3 main components:
    - An extractor to get the prompt understanding and idea forming from the LLM
    - A generator to generate the next word
    - A loopback to feed the generated word back in with the extractor ouptut

    Args:
        torch (_type_): _description_
    """
    def __init__(self):
        super(Phraser, self).__init__()

    def forward(self, tokens: List[int]) -> torch.Tensor:
        with torch.no_grad():
            tokens_tensor = torch.unsqueeze(torch.tensor(tokens).long(), 0).cuda()

            self.model.training = False
            logits = self.model(tokens_tensor, 0)
            probs = torch.nn.functional.softmax(logits, dim=1)[0, :]

        return probs

class Extractor(torch.nn.Module):
    """
    Extractor takes the internal representation of a sentence and predicts
    the next word in the sentence.

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
