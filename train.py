"""DeepSeek-V3's architecture at about a billion parameters, trained under Ravex.

    python train.py --config configs/1b.json --steps 2000

The data is ``data/train.bin`` and ``data/val.bin`` from ``prepare.py``; when
they are missing it is run first, for ``--data-tokens`` tokens. The run's
store, name and metrics endpoint come from the ``RAVEX_*`` environment, so the
same script runs on a laptop or on a node an agent started it on, and its
checkpoints make it resumable and forkable from any of them.

**Batches are drawn by step, not by position.** Step ``s`` takes its windows
from a generator seeded with ``seed + s``, so a run resumed at step ``s`` sees
exactly the batches it would have seen, with nothing about the data to save
in the checkpoint.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

import ravex
from model import ModelArgs, MoE, Transformer, count_parameters

HERE = os.path.dirname(os.path.abspath(__file__))


def tokens(data: str, split: str) -> np.ndarray:
    return np.memmap(os.path.join(data, split + ".bin"), dtype="<u4", mode="r")


def compile_static_parts(model: Transformer) -> None:
    """``torch.compile`` what has the same shapes at every step.

    Attention, the norms, the dense layer's MLP and each MoE layer's shared
    experts see ``(batch, seq, dim)`` every time, so Inductor compiles them
    once and generates Triton kernels for them on a GPU. The routed experts do
    not: how many tokens each one gets changes at every step, and the loop
    over them reads the counts back (``tolist``), which would break the graph
    and recompile until torch gave up. They stay eager.

    ``Module.compile`` works in place: the state dict keeps its keys, so
    checkpoints from a run without ``--compile`` resume with it and back.
    """
    for block in model.layers:
        block.attn.compile()
        block.attn_norm.compile()
        block.ffn_norm.compile()
        if isinstance(block.ffn, MoE):
            block.ffn.shared_experts.compile()
        else:
            block.ffn.compile()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/1b.json")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch", type=int, default=8, help="sequences per micro-batch")
    parser.add_argument("--accum", type=int, default=4, help="micro-batches per step")
    parser.add_argument("--seq", type=int, default=None, help="sequence length; the config's by default")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--balance-rate", type=float, default=0.001, help="how far a gate's bias moves per step")
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--data", default=os.path.join(HERE, "data"))
    parser.add_argument("--data-tokens", type=int, default=20_000_000, help="what prepare.py writes when data is missing")
    parser.add_argument("--compile", action="store_true", help="torch.compile the model")
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    config = args.config if os.path.isabs(args.config) else os.path.join(HERE, args.config)
    model_args = ModelArgs.load(config, max_seq_len=args.seq)
    seq = model_args.max_seq_len
    if not os.path.isfile(os.path.join(args.data, "train.bin")):
        print("no data in %s; preparing %d tokens" % (args.data, args.data_tokens), flush=True)
        subprocess.run(
            [sys.executable, os.path.join(HERE, "prepare.py"), "--tokens", str(args.data_tokens), "--out", args.data],
            check=True,
        )
    train, val = tokens(args.data, "train"), tokens(args.data, "val")

    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    # bf16 on a GPU, where it is what the model was designed for; full
    # precision on a CPU, where bf16 matmuls are slow and only tests run.
    autocast = torch.autocast("cuda", dtype=torch.bfloat16) if cuda else torch.autocast("cpu", enabled=False)
    if cuda:
        torch.backends.cuda.matmul.allow_tf32 = True

    def batch_of(source: np.ndarray, generator: torch.Generator):
        starts = torch.randint(len(source) - seq - 1, (args.batch,), generator=generator).tolist()
        window = np.stack([source[s : s + seq + 1] for s in starts]).astype(np.int64)
        window = torch.from_numpy(window).to(device, non_blocking=True)
        return window[:, :-1], window[:, 1:]

    def lr_at(step: int) -> float:
        if step < args.warmup:
            return args.lr * (step + 1) / args.warmup
        progress = min(1.0, (step - args.warmup) / max(1, args.steps - args.warmup))
        floor = args.lr * args.min_lr_ratio
        return floor + (args.lr - floor) * 0.5 * (1 + math.cos(math.pi * progress))

    @ravex.train_loop(backend="moonclip", checkpoint_every=250, keep_last=3, metrics_every=10)
    def run():
        torch.manual_seed(args.seed)
        model = Transformer(model_args).to(device)
        decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
        plain = [p for n, p in model.named_parameters() if p.dim() < 2]
        optimizer = torch.optim.AdamW(
            [{"params": decay, "weight_decay": args.weight_decay}, {"params": plain, "weight_decay": 0.0}],
            lr=args.lr, betas=(0.9, 0.95), eps=1e-8, fused=cuda,
        )
        schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: lr_at(s) / args.lr)
        ravex.track(model=model, optimizer=optimizer, scheduler=schedule)
        if args.compile:
            compile_static_parts(model)
        forward = model

        counted = count_parameters(model_args)
        print(
            "%.3fB parameters, %.3fB active; %d tokens a step; %d training tokens on %s"
            % (counted["total"] / 1e9, counted["active"] / 1e9, args.batch * args.accum * seq, len(train), device),
            flush=True,
        )

        def loss_of(x, y):
            with autocast:
                logits = forward(x)
            return F.cross_entropy(logits.float().view(-1, logits.size(-1)), y.reshape(-1))

        last = time.monotonic()
        while ravex.step() < args.steps:
            ravex.batch_boundary()
            step = ravex.step()
            model.train()
            generator = torch.Generator().manual_seed(args.seed + step)
            total = 0.0
            for _ in range(args.accum):
                x, y = batch_of(train, generator)
                loss = loss_of(x, y) / args.accum
                loss.backward()
                total += loss.item()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            schedule.step()
            imbalance = model.balance(args.balance_rate)

            now = time.monotonic()
            ravex.log_metrics(
                {
                    "train/loss": total,
                    "train/grad_norm": grad_norm,
                    "train/lr": schedule.get_last_lr()[0],
                    "moe/imbalance": imbalance,
                    "perf/tokens_per_s": args.batch * args.accum * seq / max(now - last, 1e-9),
                }
            )
            last = now
            done = ravex.step()
            if done % args.eval_every == 0 or done == args.steps:
                # Tensors are logged as histograms (Ravex bins any value with
                # more than one element). Taken before the eval, whose own
                # forward would overwrite the gates' loads with its batches':
                # how each expert's share compares with the mean, the bias the
                # balancing has pushed each one by, and the two largest
                # matrices, the embedding and the head.
                gates = list(model.gates())
                ravex.log_metrics(
                    {
                        "moe/expert_load": torch.cat([g.load / g.load.mean().clamp_min(1e-9) for g in gates]),
                        "moe/gate_bias": torch.cat([g.bias.detach().flatten() for g in gates]),
                        "weights/embed": model.embed.weight.detach(),
                        "weights/head": model.head.weight.detach(),
                    }
                )
                model.eval()
                with torch.no_grad():
                    losses = []
                    for i in range(args.eval_batches):
                        vx, vy = batch_of(val, torch.Generator().manual_seed(10_000_000 + i))
                        losses.append(loss_of(vx, vy).item())
                vloss = sum(losses) / len(losses)
                ravex.log_metrics({"eval/loss": vloss, "eval/perplexity": math.exp(min(vloss, 20))})
                print("step %d  train %.3f  eval %.3f  imbalance %.2f" % (done, total, vloss, imbalance), flush=True)
                last = time.monotonic()

    run()


if __name__ == "__main__":
    main()
