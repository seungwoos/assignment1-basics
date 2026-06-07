import math
from collections.abc import Callable

import torch


def cosine_lr_scheduler(
    it: int, max_learning_rate: float, min_learning_rate: float, warmup_iters: int, cosine_cycle_iters: int
) -> float:
    if it < warmup_iters:
        return it / warmup_iters * max_learning_rate
    elif warmup_iters <= it <= cosine_cycle_iters:
        return min_learning_rate + 0.5 * (
            1 + math.cos((it - warmup_iters) / (cosine_cycle_iters - warmup_iters) * math.pi)
        ) * (max_learning_rate - min_learning_rate)
    else:
        return min_learning_rate


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, weight_decay=0.01, betas=(0.9, 0.95), eps=1e-8):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1 parameter: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2 parameter: {betas[1]}")
        if eps < 0:
            raise ValueError(f"Invalid epsilon value: {eps}")

        defaults = {"lr": lr, "weight_decay": weight_decay, "betas": betas, "eps": eps}
        super().__init__(params, defaults)

    def step(self, closure: Callable | None = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta_1, beta_2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                grad = p.grad.data

                if len(state) == 0:
                    state["t"] = 0
                    state["m"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["v"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                m, v = state["m"], state["v"]
                state["t"] += 1
                t = state["t"]

                lr_t = lr * math.sqrt(1 - beta_2**t) / (1 - beta_1**t)
                p.data -= lr * weight_decay * p.data

                m.mul_(beta_1).add_(grad, alpha=1 - beta_1)
                v.mul_(beta_2).add_(grad**2, alpha=1 - beta_2)

                p.data -= lr_t * m / (v**0.5 + eps)

        return loss
