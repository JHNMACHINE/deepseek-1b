# deepseek-1b

[DeepSeek-V3](https://github.com/deepseek-ai/DeepSeek-V3)'s architecture at
about a billion parameters, made trainable and trained under
[Ravex](https://github.com/JHNMACHINE/ravex), so a run checkpoints, resumes
and forks.

| | |
|---|---|
| Parameters | 0.94B in all, 0.49B active per token (`configs/1b.json`) |
| Attention | Multi-head Latent Attention, `kv_lora_rank` 512, 16 heads |
| Feed-forward | 1 dense layer, then DeepSeekMoE: 32 routed experts (6 per token) + 1 shared |
| Routing | sigmoid scores, balanced by V3's bias without an auxiliary loss |
| Tokenizer | DeepSeek-V3's, 129,280 entries |
| Data | [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) `sample/10BT` |

`model.py` is DeepSeek's `inference/model.py` with what training needs and
generation does not: no `inference_mode` or KV cache, logits for every
position, bf16 only, the gate's balancing bias updated every step, and an
initialisation. Its docstring lists every change. Multi-Token Prediction is
not there yet.

```bash
pip install torch ravex -r requirements.txt
python prepare.py --tokens 20000000 --out data      # FineWeb-Edu, tokenized
python model.py configs/1b.json                     # count the parameters
RAVEX_STORAGE_PATH=runs/mine python train.py --config configs/1b.json --steps 2000
```

`configs/tiny.json` is the same architecture small enough for a CPU, to try the
whole loop - checkpoint, resume, fork - before renting anything:

```bash
RAVEX_STORAGE_PATH=runs/tiny python train.py --config configs/tiny.json \
  --batch 4 --accum 1 --seq 128 --lr 3e-3 --warmup 20 --steps 300
```

`--compile` runs `torch.compile` on what has the same shapes at every step -
attention, norms, the dense and shared MLPs - and leaves the routed experts
eager, since how many tokens each gets changes every step. Losses are the same
as without it, and a checkpoint resumes either way.

`gpuzero.toml` declares the scripts an agent may start, with their default
parameters; `compile` is on there.

## License

MIT. The model code is adapted from DeepSeek-V3's, Copyright (c) 2023
DeepSeek, under the MIT license in `LICENSE`.
