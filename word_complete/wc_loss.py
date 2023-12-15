import torch
from llama.tokenizer import Tokenizer


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
