from collections.abc import Iterable

import torch
from jaxtyping import Float


def cross_entropy_loss(
    logits: Float[torch.Tensor, " batch_size vocab_size"], targets: Float[torch.Tensor, " batch_size"]
) -> Float[torch.Tensor, ""]:
    max_logits = logits.max(dim=-1, keepdim=True).values
    log_softmax = (logits - max_logits) - torch.log(torch.sum(torch.exp(logits - max_logits), dim=-1, keepdim=True))

    target_log_prob = log_softmax.gather(-1, targets.unsqueeze(-1)).squeeze(-1)

    return -target_log_prob.mean()


def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float, eps: float = 1e-6):
    max_norm = 0.0
    for p in parameters:
        if p.grad is None:
            continue

        grad = p.grad.data
        max_norm += (grad**2).sum()

    max_norm **= 0.5
    if max_l2_norm < max_norm:
        for p in parameters:
            if p.grad is None:
                continue

            p.grad.data.mul_(max_l2_norm).div_(max_norm + eps)
