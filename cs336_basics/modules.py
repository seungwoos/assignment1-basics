import math

import torch
import torch.nn as nn
from jaxtyping import Bool, Float, Int


def silu(x: Float[torch.Tensor, " ..."]) -> Float[torch.Tensor, " ..."]:
    return x * torch.sigmoid(x)


def softmax(x: Float[torch.Tensor, " ..."], dim: int) -> Float[torch.Tensor, " ..."]:
    x_max = x.max(dim=dim, keepdim=True).values
    x_exp = torch.exp(x - x_max)
    return x_exp / x_exp.sum(dim=dim, keepdim=True)


def scaled_dot_product_attention(
    q: Float[torch.Tensor, " ... queries d_k"],
    k: Float[torch.Tensor, " ... keys d_k"],
    v: Float[torch.Tensor, " ... keys d_v"],
    mask: Bool[torch.Tensor, " ... queries keys"],
) -> Float[torch.Tensor, " ... queries d_v"]:
    dim_k = k.shape[-1]

    qk = torch.einsum("...ik, ...jk -> ...ij", q, k) / dim_k**0.5
    qk = torch.where(mask, qk, float("-inf"))

    attention_weights = softmax(qk, dim=-1)
    output = torch.einsum("...ij, ...jk -> ...ik", attention_weights, v)

    return output


class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()

        self.weight = nn.Parameter(torch.empty((out_features, in_features), device=device, dtype=dtype))
        self.device = device
        self.dtype = dtype

        self.reset_parameter()

    def reset_parameter(self):
        fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(self.weight)
        std = math.sqrt(2 / (fan_in + fan_out))
        bound = 3 * std
        nn.init.trunc_normal_(self.weight, mean=0, std=std, a=-bound, b=bound)

    def forward(self, x: Float[torch.Tensor, " d_out d_in"]) -> Float[torch.Tensor, " ... d_out"]:
        return x @ self.weight.T


class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()

        self.weight = nn.Parameter(torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype))
        self.device = device
        self.dtype = dtype

        self.reset_parameter()

    def reset_parameter(self):
        nn.init.trunc_normal_(self.weight, a=-3, b=3)

    def forward(self, token_ids: Int[torch.Tensor, "..."]) -> Float[torch.Tensor, "... embedding_dim"]:
        embedded = self.weight[token_ids]

        return embedded


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()

        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype

    def forward(self, x: Float[torch.Tensor, " ... d_model"]) -> Float[torch.Tensor, " ... d_model"]:
        in_dtype = x.dtype
        x = x.to(torch.float32)

        var = x.pow(2).mean(dim=-1, keepdim=True) + self.eps
        input_norm = x * torch.rsqrt(var)
        result = self.weight * input_norm

        return result.to(in_dtype)


class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()

        self.w1 = Linear(d_model, d_ff, device, dtype)
        self.w2 = Linear(d_ff, d_model, device, dtype)
        self.w3 = Linear(d_model, d_ff, device, dtype)

        self.device = device
        self.dtype = dtype

    def forward(self, x: Float[torch.Tensor, " ... d_model"]) -> Float[torch.Tensor, " ... d_model"]:
        return self.w2(silu(self.w1(x)) * self.w3(x))


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device

        self.init_rope()

    def init_rope(self):
        freq = 1.0 / (self.theta ** (torch.arange(0, self.d_k, 2, device=self.device)[: self.d_k // 2] / self.d_k))

        seq_idx = torch.arange(self.max_seq_len, dtype=freq.dtype, device=freq.device)
        idx_theta = torch.einsum("i, j -> ij", seq_idx, freq).float()  # shape: [bs, d_k//2, max_seq_len]

        cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)
        self.register_buffer("cache", cache, persistent=False)  # shape: [bs, d_k//2, max_seq_len, 2]

    def forward(
        self,
        x: Float[torch.Tensor, " ... sequence_length d_k"],
        token_positions: Int[torch.Tensor, " ... sequence_length"],
    ) -> Float[torch.Tensor, " ... sequence_length d_k"]:
        rope_cache = self.cache[token_positions]

        x_shaped = x.float().reshape(*x.shape[:-1], -1, 2)
        x_out = torch.stack(
            [
                x_shaped[..., 0] * rope_cache[..., 0] - x_shaped[..., 1] * rope_cache[..., 1],
                x_shaped[..., 1] * rope_cache[..., 0] + x_shaped[..., 0] * rope_cache[..., 1],
            ],
            dim=-1,
        )
        x_out = x_out.reshape(*x_out.shape[:-2], -1)
        return x_out.type_as(x)


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int | None = None,
        theta: int | None = None,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.max_seq_len = max_seq_len
        self.theta = theta

        if self.theta is not None:
            self.rope = RotaryPositionalEmbedding(theta, self.head_dim, max_seq_len, device)

        self.q_proj = Linear(d_model, d_model, device, dtype)
        self.k_proj = Linear(d_model, d_model, device, dtype)
        self.v_proj = Linear(d_model, d_model, device, dtype)
        self.output_proj = Linear(d_model, d_model, device, dtype)

    def forward(
        self,
        x: Float[torch.Tensor, " ... sequence_length d_model"],
        token_positions: Int[torch.Tensor, " ... sequence_length"] | None = None,
        mask: Bool[torch.Tensor, " ... queries keys"] | None = None,
    ) -> Float[torch.Tensor, " ... sequence_length d_model"]:
        bs, seq_len, _ = x.size()

        q = self.q_proj(x).reshape(bs, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(x).reshape(bs, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(x).reshape(bs, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        if token_positions is not None:
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        if mask is None:
            mask = torch.tril(torch.ones((seq_len, seq_len), device=x.device)).bool()

        attn_output = scaled_dot_product_attention(q, k, v, mask)
        attn_output = attn_output.permute(0, 2, 1, 3)  # shape: [bs, seq_len, num_heads, self.head_dim]
        attn_output = attn_output.reshape(bs, seq_len, self.d_model)

        out = self.output_proj(attn_output)
        return out


class TransformerBlock(nn.Module):
    def __init__(
        self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, theta: float, device=None, dtype=None
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff

        self.attn = MultiHeadSelfAttention(
            d_model=d_model, num_heads=num_heads, max_seq_len=max_seq_len, theta=theta, device=device, dtype=dtype
        )
        self.ffn = FeedForwardNetwork(d_model=d_model, d_ff=d_ff, device=device, dtype=dtype)
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)

    def forward(
        self,
        x: Float[torch.Tensor, " batch sequence_length d_model"],
    ) -> Float[torch.Tensor, " batch sequence_length d_model"]:
        token_positions = torch.arange(0, x.shape[1], device=x.device)

        y = x + self.attn(self.ln1(x), token_positions)
        out = y + self.ffn(self.ln2(y))

        return out


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        theta: float,
        device=None,
        dtype=None,
    ):
        super().__init__()

        self.context_length = context_length

        self.token_embeddings = Embedding(num_embeddings=vocab_size, embedding_dim=d_model, device=device, dtype=dtype)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    max_seq_len=context_length,
                    theta=theta,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        )
        self.ln_final = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self.lm_head = Linear(in_features=d_model, out_features=vocab_size, device=device, dtype=dtype)

    def forward(
        self, x: Int[torch.Tensor, " batch_size sequence_length"]
    ) -> Float[torch.Tensor, " batch_size sequence_length vocab_size"]:
        seq_len = x.shape[1]

        if seq_len > self.context_length:
            raise ValueError(f"Input sequence length ({seq_len}) exceeds model context length ({self.context_length})")

        x = self.token_embeddings(x)

        for layer in self.layers:
            x = layer(x)

        x = self.ln_final(x)
        x = self.lm_head(x)

        return x
