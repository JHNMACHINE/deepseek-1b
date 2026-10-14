"""How long ``torch.compile`` takes on a node, with and without the cache (GPU-181).

smith fills Triton's and Inductor's cache directories from the workspace's
cache before the job starts, and sends back what the job compiled after it
ends. Two runs of this script on the same kind of GPU should then differ in
one number: the first compiles everything, the second finds the kernels and
Inductor's graphs already there. This prints that number - the seconds of the
first step, which is where the compiling happens - and the seconds of a step
once compiled, for scale, as one ``COMPILE`` line; and how many files the
cache directories held when it started, which is what smith pulled.

The model is a few transformer blocks, enough for Inductor to generate a few
dozen kernels and take tens of seconds over it, small enough for any GPU.
"""

from __future__ import annotations

import argparse
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        self.heads = heads
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.up = nn.Linear(dim, 4 * dim)
        self.down = nn.Linear(4 * dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        q, k, v = self.qkv(self.norm1(x)).view(b, t, 3, self.heads, d // self.heads).unbind(2)
        attn = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True)
        x = x + self.out(attn.transpose(1, 2).reshape(b, t, d))
        return x + self.down(F.gelu(self.up(self.norm2(x))))


def files_in(variable: str) -> int:
    directory = os.environ.get(variable, "")
    return sum(len(files) for _root, _dirs, files in os.walk(directory)) if directory else 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dim", type=int, default=1024)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()

    # Before anything compiles: what the pull left in the directories.
    print("CACHE at start: triton %d files, inductor %d files" % (
        files_in("TRITON_CACHE_DIR"), files_in("TORCHINDUCTOR_CACHE_DIR")), flush=True)
    print("GPU %s" % torch.cuda.get_device_name(0), flush=True)

    torch.manual_seed(0)
    model = nn.Sequential(*[Block(args.dim, 16) for _ in range(args.layers)]).cuda().to(torch.bfloat16)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    compiled = torch.compile(model)
    x = torch.randn(8, 512, args.dim, device="cuda", dtype=torch.bfloat16)

    def step() -> None:
        compiled(x).float().pow(2).mean().backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    started = time.perf_counter()
    step()
    torch.cuda.synchronize()
    first = time.perf_counter() - started

    started = time.perf_counter()
    for _ in range(args.steps):
        step()
    torch.cuda.synchronize()
    steady = (time.perf_counter() - started) / args.steps

    print("COMPILE first step %.1f s, then %.3f s a step" % (first, steady), flush=True)
    print("CACHE at end: triton %d files, inductor %d files" % (
        files_in("TRITON_CACHE_DIR"), files_in("TORCHINDUCTOR_CACHE_DIR")), flush=True)


if __name__ == "__main__":
    main()
