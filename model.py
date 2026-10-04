"""DeepSeek-V3's architecture, made trainable.

Adapted from ``inference/model.py`` of https://github.com/deepseek-ai/DeepSeek-V3
(MIT, Copyright (c) 2023 DeepSeek; see LICENSE). The architecture is theirs and
kept whole: Multi-head Latent Attention, DeepSeekMoE with shared and routed
experts, sigmoid routing with a bias for balancing, RoPE with YaRN. What the
original leaves out because it only generates, and this file adds or changes
so that it trains:

* **No ``inference_mode`` and no KV cache.** The original's forward runs under
  ``torch.inference_mode()`` and writes every key and value into buffers sized
  for generation. Here attention is the "naive" form - keys and values
  expanded per head - through ``scaled_dot_product_attention`` with a causal
  mask, which has a backward and uses a fused kernel where there is one.
* **Logits for every position**, not only the last, for the loss.
* **bf16 only.** The FP8 path (``kernel.py``, block-quantised weights) is for
  serving DeepSeek's released weights; training starts from scratch.
* **No tensor or expert parallelism.** Every process holds the whole model;
  several machines are Ravex's outer loop's business, not this file's.
* **Balancing without an auxiliary loss**, as V3 trains: the gate's bias is
  used to *choose* experts, never to weight them, and after every step it is
  moved toward the experts that got fewer tokens than average
  (:meth:`Transformer.balance`). The original has the bias only in the 671B
  configuration and never updates it.
* **Experts summed out of place** (``index_add``), not written into ``y``
  in place: the in-place form is the generation path's, and out of place
  autograd never has to reason about a tensor it saved being overwritten.
* **Initialisation**: every weight from a normal with std 0.006, as the V3
  report says; the original leaves them empty for a checkpoint to fill.

Not here (yet): Multi-Token Prediction, and V3's tiny sequence-wise balance
loss.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, fields
from typing import Dict, List, Literal, Optional

import torch
import torch.nn.functional as F
from torch import nn

#: The V3 report's initialisation: every learnable weight from N(0, 0.006).
INIT_STD = 0.006


@dataclass
class ModelArgs:
    """The model's shape. Field names follow DeepSeek's configs, so theirs load
    unchanged (``configs/*.json``)."""

    max_seq_len: int = 2048
    vocab_size: int = 129280
    dim: int = 1024
    inter_dim: int = 2816
    moe_inter_dim: int = 512
    n_layers: int = 12
    n_dense_layers: int = 1
    n_heads: int = 16
    # moe
    n_routed_experts: int = 32
    n_shared_experts: int = 1
    n_activated_experts: int = 6
    n_expert_groups: int = 1
    n_limited_groups: int = 1
    score_func: Literal["softmax", "sigmoid"] = "sigmoid"
    route_scale: float = 1.0
    # mla
    q_lora_rank: int = 0
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    # yarn
    original_seq_len: int = 4096
    rope_theta: float = 10000.0
    rope_factor: float = 40
    beta_fast: int = 32
    beta_slow: int = 1
    mscale: float = 1.0

    @classmethod
    def load(cls, path: str, **overrides) -> "ModelArgs":
        """A config file, ignoring keys this class does not have (DeepSeek's
        carry ``dtype`` and others that only serving uses)."""
        with open(path, encoding="utf-8") as handle:
            raw = json.load(handle)
        known = {f.name for f in fields(cls)}
        values = {key: value for key, value in raw.items() if key in known}
        values.update({key: value for key, value in overrides.items() if value is not None})
        return cls(**values)

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def precompute_freqs_cis(args: ModelArgs) -> torch.Tensor:
    """Rotary frequencies as complex numbers, with YaRN's correction when the
    model is longer than ``original_seq_len`` (unchanged from DeepSeek's)."""
    dim = args.qk_rope_head_dim
    seqlen = args.max_seq_len
    base = args.rope_theta
    factor = args.rope_factor

    def find_correction_dim(num_rotations, dim, base, max_seq_len):
        return dim * math.log(max_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(base))

    def find_correction_range(low_rot, high_rot, dim, base, max_seq_len):
        low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
        high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
        return max(low, 0), min(high, dim - 1)

    def linear_ramp_factor(min_, max_, dim):
        if min_ == max_:
            max_ += 0.001
        linear_func = (torch.arange(dim, dtype=torch.float32) - min_) / (max_ - min_)
        return torch.clamp(linear_func, 0, 1)

    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if seqlen > args.original_seq_len:
        low, high = find_correction_range(args.beta_fast, args.beta_slow, dim, base, args.original_seq_len)
        smooth = 1 - linear_ramp_factor(low, high, dim // 2)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth
    t = torch.arange(seqlen)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    dtype = x.dtype
    x = torch.view_as_complex(x.float().view(*x.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.view(1, x.size(1), 1, x.size(-1))
    y = torch.view_as_real(x * freqs_cis).flatten(3)
    return y.to(dtype)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, (self.dim,), self.weight, self.eps)


class MLA(nn.Module):
    """Multi-head Latent Attention: keys and values pass through a
    ``kv_lora_rank`` bottleneck, and a separate ``qk_rope_head_dim`` slice
    carries position, shared by every head on the key side."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.dim = args.dim
        self.n_heads = args.n_heads
        self.q_lora_rank = args.q_lora_rank
        self.kv_lora_rank = args.kv_lora_rank
        self.qk_nope_head_dim = args.qk_nope_head_dim
        self.qk_rope_head_dim = args.qk_rope_head_dim
        self.qk_head_dim = args.qk_nope_head_dim + args.qk_rope_head_dim
        self.v_head_dim = args.v_head_dim

        if self.q_lora_rank == 0:
            self.wq = nn.Linear(self.dim, self.n_heads * self.qk_head_dim, bias=False)
        else:
            self.wq_a = nn.Linear(self.dim, self.q_lora_rank, bias=False)
            self.q_norm = RMSNorm(self.q_lora_rank)
            self.wq_b = nn.Linear(self.q_lora_rank, self.n_heads * self.qk_head_dim, bias=False)
        self.wkv_a = nn.Linear(self.dim, self.kv_lora_rank + self.qk_rope_head_dim, bias=False)
        self.kv_norm = RMSNorm(self.kv_lora_rank)
        self.wkv_b = nn.Linear(self.kv_lora_rank, self.n_heads * (self.qk_nope_head_dim + self.v_head_dim), bias=False)
        self.wo = nn.Linear(self.n_heads * self.v_head_dim, self.dim, bias=False)
        self.softmax_scale = self.qk_head_dim ** -0.5
        if args.max_seq_len > args.original_seq_len:
            mscale = 0.1 * args.mscale * math.log(args.rope_factor) + 1.0
            self.softmax_scale = self.softmax_scale * mscale * mscale

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        bsz, seqlen, _ = x.size()
        q = self.wq(x) if self.q_lora_rank == 0 else self.wq_b(self.q_norm(self.wq_a(x)))
        q = q.view(bsz, seqlen, self.n_heads, self.qk_head_dim)
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        q_pe = apply_rotary_emb(q_pe, freqs_cis)
        kv = self.wkv_a(x)
        kv, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis)
        kv = self.wkv_b(self.kv_norm(kv)).view(bsz, seqlen, self.n_heads, self.qk_nope_head_dim + self.v_head_dim)
        k_nope, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        q = torch.cat([q_nope, q_pe], dim=-1)
        k = torch.cat([k_nope, k_pe.expand(-1, -1, self.n_heads, -1)], dim=-1)
        out = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True, scale=self.softmax_scale
        )
        return self.wo(out.transpose(1, 2).flatten(2))


class MLP(nn.Module):
    """SwiGLU feed-forward: the dense layers, and the shared experts."""

    def __init__(self, dim: int, inter_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, inter_dim, bias=False)
        self.w2 = nn.Linear(inter_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, inter_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


Expert = MLP


class Gate(nn.Module):
    """Which experts a token goes to, and with what weight.

    The bias is a buffer, not a parameter: no gradient reaches it. It shifts
    which experts are *chosen* and nothing else - the weights come from the
    scores without it - and :meth:`Transformer.balance` moves it after each
    step. ``load`` is how many tokens each expert got in the last forward.
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.dim = args.dim
        self.topk = args.n_activated_experts
        self.n_groups = args.n_expert_groups
        self.topk_groups = args.n_limited_groups
        self.score_func = args.score_func
        self.route_scale = args.route_scale
        self.weight = nn.Parameter(torch.empty(args.n_routed_experts, args.dim))
        self.register_buffer("bias", torch.zeros(args.n_routed_experts, dtype=torch.float32))
        self.register_buffer("load", torch.zeros(args.n_routed_experts, dtype=torch.float32), persistent=False)

    def forward(self, x: torch.Tensor):
        scores = F.linear(x, self.weight)
        if self.score_func == "softmax":
            scores = scores.softmax(dim=-1, dtype=torch.float32)
        else:
            scores = scores.float().sigmoid()
        original_scores = scores
        choice = scores.detach() + self.bias
        if self.n_groups > 1:
            choice = choice.view(x.size(0), self.n_groups, -1)
            group_scores = choice.topk(2, dim=-1)[0].sum(dim=-1)
            indices = group_scores.topk(self.topk_groups, dim=-1)[1]
            mask = choice.new_ones(x.size(0), self.n_groups, dtype=torch.bool).scatter_(1, indices, False)
            choice = choice.masked_fill(mask.unsqueeze(-1), float("-inf")).flatten(1)
        indices = torch.topk(choice, self.topk, dim=-1)[1]
        weights = original_scores.gather(1, indices)
        if self.score_func == "sigmoid":
            weights = weights / weights.sum(dim=-1, keepdim=True)
        weights = weights * self.route_scale
        with torch.no_grad():
            self.load.copy_(torch.bincount(indices.flatten(), minlength=self.load.numel()).float())
        return weights.type_as(x), indices


class MoE(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.dim = args.dim
        self.n_routed_experts = args.n_routed_experts
        self.gate = Gate(args)
        self.experts = nn.ModuleList([Expert(args.dim, args.moe_inter_dim) for _ in range(args.n_routed_experts)])
        self.shared_experts = MLP(args.dim, args.n_shared_experts * args.moe_inter_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.size()
        x = x.view(-1, self.dim)
        weights, indices = self.gate(x)
        y = torch.zeros_like(x)
        counts = self.gate.load.tolist()
        for i, expert in enumerate(self.experts):
            if counts[i] == 0:
                continue
            idx, top = torch.where(indices == i)
            y = y.index_add(0, idx, expert(x[idx]) * weights[idx, top, None])
        return (y + self.shared_experts(x)).view(shape)


class Block(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.attn = MLA(args)
        self.ffn = MLP(args.dim, args.inter_dim) if layer_id < args.n_dense_layers else MoE(args)
        self.attn_norm = RMSNorm(args.dim)
        self.ffn_norm = RMSNorm(args.dim)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), freqs_cis)
        return x + self.ffn(self.ffn_norm(x))


class Transformer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.embed = nn.Embedding(args.vocab_size, args.dim)
        self.layers = nn.ModuleList(Block(i, args) for i in range(args.n_layers))
        self.norm = RMSNorm(args.dim)
        self.head = nn.Linear(args.dim, args.vocab_size, bias=False)
        self.register_buffer("freqs_cis", precompute_freqs_cis(args), persistent=False)
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, std=INIT_STD)
            elif isinstance(module, Gate):
                nn.init.normal_(module.weight, std=INIT_STD)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Logits for every position, ``(batch, seq, vocab)``."""
        h = self.embed(tokens)
        freqs_cis = self.freqs_cis[: tokens.size(1)]
        for layer in self.layers:
            h = layer(h, freqs_cis)
        return self.head(self.norm(h))

    def gates(self) -> List[Gate]:
        return [module for module in self.modules() if isinstance(module, Gate)]

    @torch.no_grad()
    def balance(self, rate: float) -> float:
        """V3's auxiliary-loss-free balancing: after a step, every gate's bias
        moves by ``rate`` toward its under-loaded experts and away from its
        over-loaded ones. Answers how uneven the load was - the most loaded
        expert over the mean, averaged over the MoE layers; 1.0 is even."""
        imbalance = []
        for gate in self.gates():
            load = gate.load
            mean = load.mean()
            if mean <= 0:
                continue
            gate.bias.add_(rate * torch.sign(mean - load))
            imbalance.append((load.max() / mean).item())
        return sum(imbalance) / len(imbalance) if imbalance else 1.0


def count_parameters(args: ModelArgs) -> Dict[str, int]:
    """Total parameters, and those one token passes through (embeddings and
    the head included), without building the model."""
    with torch.device("meta"):
        model = Transformer(args)
    total = sum(p.numel() for p in model.parameters())
    expert = sum(p.numel() for p in model.layers[-1].ffn.experts[0].parameters()) if args.n_layers > args.n_dense_layers else 0
    moe_layers = args.n_layers - args.n_dense_layers
    idle = moe_layers * expert * (args.n_routed_experts - args.n_activated_experts)
    return {"total": total, "active": total - idle}


if __name__ == "__main__":
    import sys

    args = ModelArgs.load(sys.argv[1]) if len(sys.argv) > 1 else ModelArgs()
    counted = count_parameters(args)
    print("%.3fB parameters, %.3fB active per token" % (counted["total"] / 1e9, counted["active"] / 1e9))
