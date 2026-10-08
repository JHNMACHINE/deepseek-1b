"""Several machines, one run, with a toy model: the outer loop's measurements (GPU-211).

Started as a cluster of N nodes with one GPU each, this is an ordinary Ravex
script: nothing in it knows about the other nodes. What makes it a cluster is
the launcher, which sets ``RAVEX_*`` so the nodes find each other and the
outer loop turns itself on; the checkpoints, the resume and the replication
between machines are Ravex's, and this script only has to leave evidence.

**Each node draws its own batches** (the generator is seeded by the host name
and the step), so the nodes' weights drift apart between rounds and a round
that averages them is something that can be seen. Every ``--report-every``
steps each node prints one ``RESULT`` line: the step, the loss and a checksum
of the weights. Between rounds the checksums differ.

**A round is seen where it happens.** Ravex closes it inside
``ravex.batch_boundary()``, at the top of the next step, so a ``step`` line
is always the weights before it. The script compares the checksum across that
call, and when the weights changed under it - nothing else replaces them - it
prints a ``round`` line with the checksum after: every node's must be the
same number, which is what shows the round averaged them. The first run on
two machines printed only ``step`` lines, before the round and ten steps
after, and so could not show that (GPU-211). A node that was killed and came back shows up as a line
with a lower step than the one before it, and a checksum that has rejoined
the others' after the next round.

``--pace`` slows each step down, so a run lasts long enough to take a node
away by hand, which is the scenario this exists for.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import time

import torch
import torch.nn.functional as F

import ravex
from matrix import Toy


def checksum(model: torch.nn.Module) -> float:
    """One number for the whole parameter vector: equal weights, equal number."""
    return round(sum(p.detach().double().sum().item() for p in model.parameters()), 9)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--report-every", type=int, default=10)
    parser.add_argument("--pace", type=float, default=0.05, help="seconds to sleep per step")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    host = socket.gethostname()
    salt = int(hashlib.sha256(host.encode()).hexdigest()[:8], 16)

    @ravex.train_loop(backend="moonclip", checkpoint_every=100, keep_last=3, metrics_every=10)
    def run():
        torch.manual_seed(0)
        model = Toy(args.dim, args.layers).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        ravex.track(model=model, optimizer=optimizer)
        print("RESULT " + json.dumps({
            "case": "cluster", "event": "start", "host": host, "device": device,
            "resumed_at": ravex.step(), "torch": torch.__version__,
            "ravex_env": sorted(k for k in os.environ if k.startswith("RAVEX_")),
        }), flush=True)

        last = None
        while ravex.step() < args.steps:
            ravex.batch_boundary()
            step = ravex.step()
            now = checksum(model)
            if last is not None and now != last:
                print("RESULT " + json.dumps({
                    "case": "cluster", "event": "round", "host": host, "step": step, "checksum": now,
                }), flush=True)
            generator = torch.Generator().manual_seed(salt + step)
            x = torch.randn(args.batch, args.dim, generator=generator).to(device)
            loss = F.mse_loss(model(x), x.roll(1, dims=-1))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            last = checksum(model)
            ravex.log_metrics({"train/loss": loss})
            done = ravex.step()
            if done % args.report_every == 0:
                print("RESULT " + json.dumps({
                    "case": "cluster", "event": "step", "host": host, "step": done,
                    "loss": round(loss.item(), 6), "checksum": checksum(model),
                }), flush=True)
            if args.pace:
                time.sleep(args.pace)
        print("RESULT " + json.dumps({"case": "cluster", "event": "done", "host": host,
                                      "step": ravex.step(), "checksum": checksum(model)}), flush=True)

    run()


if __name__ == "__main__":
    main()
