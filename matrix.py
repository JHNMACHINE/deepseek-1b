"""Checkpoint compatibility of plain PyTorch, small enough to run anywhere (GPU-211).

    torchrun --nproc_per_node 4 matrix.py --case layouts

This repository has become the dummy one the platform is tried with: the model
here is a toy, so a run costs minutes. The question each case asks is about
``torch.distributed.checkpoint`` (DCP) on a training state sharded in
different ways, and every answer is a number on a ``RESULT`` line, never "it
seems to work". Needs a world size that is a multiple of 4.

Three layouts of the same model, all with FSDP2 (``fully_shard``):

* ``fsdp``: one mesh axis, every parameter sharded over all ranks.
* ``hsdp``: a (replicate, shard) mesh, sharded inside a group and replicated
  between the groups.
* ``tp``: a (dp, tp) mesh, the two linears of each block split column- and
  row-wise over ``tp`` and then sharded over ``dp``: DTensors on a 2D mesh.

Every rank draws the same batch for a step, from a generator seeded by the
step, so the losses of two layouts are comparable and a resumed run needs
nothing about the data saved.

Cases:

* ``layouts``: train, save with layout A, keep training; load into a fresh
  model of layout B, train the same steps; ``max_diff`` is how far the two
  continuations' losses are. A to A is a resume: ``exact`` says whether it is
  bit-identical. A to B is a reshard onto another mesh.
* ``single``: save with each layout, then read the checkpoint in one process,
  into a plain model with no process group (what fine-tuning or inference
  would do), and compare every weight with the saved model's.
* ``async``: the same resume, saved with ``dcp.async_save``: how long the
  call blocks the loop, how long the write takes in the background, and
  whether the resumed losses are still exact.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel, parallelize_module

LAYOUTS = ("fsdp", "hsdp", "tp")


class Block(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, 4 * dim, bias=False)
        self.w2 = nn.Linear(4 * dim, dim, bias=False)

    def forward(self, x):
        return x + self.w2(F.gelu(self.w1(x)))


class Toy(nn.Module):
    def __init__(self, dim: int, layers: int):
        super().__init__()
        self.blocks = nn.ModuleList(Block(dim) for _ in range(layers))

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


def say(**fields) -> None:
    """One line a run's log can be searched for, from rank 0 only."""
    if dist.get_rank() == 0:
        print("RESULT " + json.dumps(fields), flush=True)


def synced() -> float:
    """The time, after every rank and the device have caught up."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    dist.barrier()
    return time.perf_counter()


def build(layout: str, args, device: str) -> tuple[Toy, torch.optim.Optimizer]:
    world = dist.get_world_size()
    torch.manual_seed(0)
    model = Toy(args.dim, args.layers).to(device)
    if layout == "fsdp":
        mesh = init_device_mesh(device, (world,))
        for block in model.blocks:
            fully_shard(block, mesh=mesh)
        fully_shard(model, mesh=mesh)
    elif layout == "hsdp":
        mesh = init_device_mesh(device, (world // 2, 2), mesh_dim_names=("replicate", "shard"))
        for block in model.blocks:
            fully_shard(block, mesh=mesh)
        fully_shard(model, mesh=mesh)
    else:
        mesh = init_device_mesh(device, (world // 2, 2), mesh_dim_names=("dp", "tp"))
        for block in model.blocks:
            parallelize_module(block, mesh["tp"], {"w1": ColwiseParallel(), "w2": RowwiseParallel()})
            fully_shard(block, mesh=mesh["dp"])
        fully_shard(model, mesh=mesh["dp"])
    return model, torch.optim.AdamW(model.parameters(), lr=1e-3)


def train(model, optimizer, first: int, steps: int, args, device: str) -> list[float]:
    losses = []
    for step in range(first, first + steps):
        generator = torch.Generator().manual_seed(1000 + step)
        x = torch.randn(args.batch, args.dim, generator=generator).to(device)
        loss = F.mse_loss(model(x), x.roll(1, dims=-1))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        losses.append(loss.item())
    return losses


def state_of(model, optimizer, step: int) -> dict:
    model_state, optim_state = get_state_dict(model, optimizer)
    return {"model": model_state, "optim": optim_state, "step": step}


def load_into(model, optimizer, path: str) -> int:
    """Fill a fresh model and optimizer from ``path``; the step it was saved at.

    An optimizer has no moments before its first step, so there is nothing for
    DCP to load into: a step on zero gradients creates them (and the load then
    overwrites everything that step touched).
    """
    for p in model.parameters():
        p.grad = torch.zeros_like(p)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    state = state_of(model, optimizer, -1)
    dcp.load(state, checkpoint_id=path)
    set_state_dict(model, optimizer, model_state_dict=state["model"], optim_state_dict=state["optim"])
    return int(state["step"])


def full_weights(model) -> dict[str, torch.Tensor]:
    """Every weight whole and on the CPU, on every rank (a collective)."""
    weights = {}
    for name, p in model.named_parameters():
        whole = p.full_tensor() if hasattr(p, "full_tensor") else p
        weights[name] = whole.detach().cpu()
    return weights


def layouts(args, device: str) -> None:
    for a in LAYOUTS:
        model, optimizer = build(a, args, device)
        head = train(model, optimizer, 0, args.steps, args, device)
        path = os.path.join(args.out, "layouts-" + a)
        t0 = synced()
        dcp.save(state_of(model, optimizer, args.steps), checkpoint_id=path)
        save_s = synced() - t0
        size = sum(os.path.getsize(os.path.join(path, f)) for f in os.listdir(path))
        reference = train(model, optimizer, args.steps, args.steps, args, device)
        for b in LAYOUTS:
            fresh, fresh_optimizer = build(b, args, device)
            t0 = synced()
            step = load_into(fresh, fresh_optimizer, path)
            load_s = synced() - t0
            tail = train(fresh, fresh_optimizer, step, args.steps, args, device)
            diff = max(abs(u - v) for u, v in zip(reference, tail))
            say(case="layouts", saved_with=a, loaded_into=b, step=step, head_loss=head[-1],
                max_diff=diff, exact=tail == reference, save_s=round(save_s, 3),
                load_s=round(load_s, 3), bytes=size)
        shutil.rmtree(path, ignore_errors=True) if dist.get_rank() == 0 else None
        dist.barrier()


def single(args, device: str) -> None:
    for a in LAYOUTS:
        model, optimizer = build(a, args, device)
        train(model, optimizer, 0, args.steps, args, device)
        path = os.path.join(args.out, "single-" + a)
        dcp.save(state_of(model, optimizer, args.steps), checkpoint_id=path)
        saved = full_weights(model)
        if dist.get_rank() == 0:
            plain = Toy(args.dim, args.layers)
            state = {"model": dict(plain.state_dict())}
            t0 = time.perf_counter()
            dcp.load(state, checkpoint_id=path, no_dist=True)
            load_s = time.perf_counter() - t0
            diff = max((state["model"][k].float() - v.float()).abs().max().item() for k, v in saved.items())
            say(case="single", saved_with=a, process_group=False, max_weight_diff=diff,
                load_s=round(load_s, 3), tensors=len(saved))
            shutil.rmtree(path, ignore_errors=True)
        dist.barrier()


def asynchronous(args, device: str) -> None:
    model, optimizer = build("fsdp", args, device)
    train(model, optimizer, 0, args.steps, args, device)
    path = os.path.join(args.out, "async")
    # A second process group for the background write, as DCP's docs ask: its
    # collectives must not interleave with the training ones.
    group = dist.new_group(backend="gloo")
    t0 = synced()
    future = dcp.async_save(state_of(model, optimizer, args.steps), checkpoint_id=path, process_group=group)
    call_s = synced() - t0
    started = time.perf_counter()
    reference = train(model, optimizer, args.steps, args.steps, args, device)
    during_s = time.perf_counter() - started
    future.result()
    total_s = synced() - t0
    fresh, fresh_optimizer = build("fsdp", args, device)
    step = load_into(fresh, fresh_optimizer, path)
    tail = train(fresh, fresh_optimizer, step, args.steps, args, device)
    say(case="async", step=step, exact=tail == reference, call_blocks_s=round(call_s, 3),
        steps_during_write_s=round(during_s, 3), write_done_after_s=round(total_s, 3),
        max_diff=max(abs(u - v) for u, v in zip(reference, tail)))
    shutil.rmtree(path, ignore_errors=True) if dist.get_rank() == 0 else None
    dist.barrier()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--case", choices=("layouts", "single", "async"), default="layouts")
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--steps", type=int, default=5, help="steps before the save, and again after it")
    parser.add_argument("--out", default="runs/matrix")
    args = parser.parse_args()

    cuda = torch.cuda.is_available()
    if cuda:
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    dist.init_process_group("nccl" if cuda else "gloo")
    device = "cuda" if cuda else "cpu"
    if dist.get_world_size() % 4:
        raise SystemExit("needs a world size that is a multiple of 4, got %d" % dist.get_world_size())
    if dist.get_rank() == 0:
        os.makedirs(args.out, exist_ok=True)
    dist.barrier()
    say(case="environment", torch=torch.__version__, device=device, world=dist.get_world_size(),
        gpu=torch.cuda.get_device_name(0) if cuda else None, params=sum(p.numel() for p in Toy(args.dim, args.layers).parameters()))
    {"layouts": layouts, "single": single, "async": asynchronous}[args.case](args, device)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
