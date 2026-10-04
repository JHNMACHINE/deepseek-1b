"""FineWeb-Edu, tokenized with DeepSeek-V3's tokenizer, into flat files of tokens.

    python prepare.py --tokens 20000000 --out data

Writes ``data/train.bin`` and ``data/val.bin``: every document's tokens
followed by the end-of-sentence token, as little-endian uint32 (the vocabulary
is 129,280, past what uint16 holds), plus ``data/meta.json`` saying what they
are. One document in a hundred goes to ``val``.

The dataset is read from the Hub a row group at a time, so a small sample
downloads only what it uses rather than the 2 GB files whole. Documents are
taken in the dataset's own order, so the same ``--tokens`` gives the same
files on any machine.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

DATASET = "HuggingFaceFW/fineweb-edu"
SUBSET = "sample/10BT"
TOKENIZER = "deepseek-ai/DeepSeek-V3"
EOS = "<｜end▁of▁sentence｜>"
VAL_EVERY = 100


def documents(limit_files: int | None = None):
    """The dataset's ``text`` column, document by document, row group by row group."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()
    files = sorted(fs.glob("datasets/%s/%s/*.parquet" % (DATASET, SUBSET)))
    for path in files[:limit_files]:
        with fs.open(path, "rb", block_size=8 << 20) as handle:
            parquet = pq.ParquetFile(handle)
            for group in range(parquet.num_row_groups):
                for text in parquet.read_row_group(group, columns=["text"]).column("text").to_pylist():
                    yield text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tokens", type=int, default=20_000_000, help="training tokens to write")
    parser.add_argument("--out", default="data")
    parser.add_argument("--batch", type=int, default=256, help="documents tokenized at once")
    args = parser.parse_args()

    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(hf_hub_download(TOKENIZER, "tokenizer.json"))
    eos = tokenizer.token_to_id(EOS)
    if eos is None:
        raise SystemExit("the tokenizer has no %r" % EOS)
    os.makedirs(args.out, exist_ok=True)
    paths = {name: os.path.join(args.out, name + ".bin") for name in ("train", "val")}
    written = {"train": 0, "val": 0}
    started = time.monotonic()
    count = 0
    with open(paths["train"] + ".part", "wb") as train, open(paths["val"] + ".part", "wb") as val:
        outs = {"train": train, "val": val}
        batch: list[str] = []
        for text in documents():
            batch.append(text)
            if len(batch) < args.batch:
                continue
            for encoding in tokenizer.encode_batch(batch, add_special_tokens=False):
                split = "val" if count % VAL_EVERY == 0 else "train"
                ids = np.asarray(encoding.ids + [eos], dtype="<u4")
                outs[split].write(ids.tobytes())
                written[split] += ids.size
                count += 1
            batch = []
            if written["train"] >= args.tokens:
                break
            print("%d documents, %d tokens, %.0fs" % (count, written["train"], time.monotonic() - started), flush=True)
    for name, path in paths.items():
        os.replace(path + ".part", path)
    meta = {
        "dataset": "%s/%s" % (DATASET, SUBSET),
        "tokenizer": TOKENIZER,
        "dtype": "uint32",
        "eos": eos,
        "vocab_size": tokenizer.get_vocab_size(),
        "documents": count,
        "tokens": written,
    }
    with open(os.path.join(args.out, "meta.json"), "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)
    print("wrote %s" % json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
