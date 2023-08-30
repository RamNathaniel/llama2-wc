import torch
from llama.tokenizer import Tokenizer

def get_suffix_mask(t: Tokenizer, device: torch.device) -> torch.Tensor:
    suffix_mask = torch.zeros(t.n_words, dtype=torch.bool, device=device)
    for id in range(t.n_words):
        suffix_mask[id] = t.is_suffix(id)
    
    return suffix_mask

def wc_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    suffix_mask: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """
    Computes the loss for the word completer model.
    """
    loss = torch.mean((logits[suffix_mask] - labels[suffix_mask])**2)
    return loss
