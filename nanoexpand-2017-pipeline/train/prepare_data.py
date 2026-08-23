#!/usr/bin/env python3
"""Tokenize train_2017_mix.jsonl into packed uint16 .bin shards for training."""

import argparse
import struct
from pathlib import Path

import numpy as np
import orjson
import yaml
from tqdm import tqdm
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent


def load_train_config(path: Path | None = None) -> dict:
    cfg_path = path or ROOT / "train" / "config_train.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=None)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--output", default=None)
    ap.add_argument("--seq-len", type=int, default=None)
    ap.add_argument("--shard-tokens", type=int, default=50_000_000,
                    help="Max tokens per .bin shard (~50M tokens per file)")
    args = ap.parse_args()

    cfg = load_train_config()
    input_path = Path(args.input or cfg["mix_jsonl"])
    if not input_path.is_absolute():
        input_path = (ROOT / input_path).resolve()
    tok_path = args.tokenizer or cfg["base_checkpoint"]
    if not Path(tok_path).is_absolute():
        tok_path = str((ROOT / tok_path).resolve())
    out_dir = Path(args.output or cfg["token_bins"])
    if not out_dir.is_absolute():
        out_dir = (ROOT / out_dir).resolve()
    seq_len = args.seq_len or cfg["seq_len"]
    shard_tokens = args.shard_tokens

    out_dir.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(tok_path)

    eos = tok.eos_token_id
    if eos is None:
        eos = tok.pad_token_id or 0

    buffer: list[int] = []
    shard_idx = 0
    total_tokens = 0
    total_docs = 0

    def flush_buffer():
        nonlocal shard_idx, buffer
        if not buffer:
            return
        arr = np.array(buffer, dtype=np.uint16)
        out = out_dir / f"train_{shard_idx:03d}.bin"
        arr.tofile(out)
        meta = {
            "path": out.name,
            "num_tokens": len(buffer),
            "seq_len": seq_len,
        }
        with open(out_dir / f"train_{shard_idx:03d}.meta.json", "wb") as mf:
            mf.write(orjson.dumps(meta))
        shard_idx += 1
        buffer.clear()

    with open(input_path, "rb") as f:
        for line in tqdm(f, desc="tokenize"):
            if not line.strip():
                continue
            row = orjson.loads(line)
            text = (row.get("text") or "").strip()
            if not text:
                continue
            ids = tok.encode(text, add_special_tokens=False)
            if not ids:
                continue
            buffer.extend(ids)
            buffer.append(eos)
            total_docs += 1
            total_tokens += len(ids) + 1
            if len(buffer) >= shard_tokens:
                flush_buffer()

    flush_buffer()

    manifest = {
        "num_shards": shard_idx,
        "total_docs": total_docs,
        "total_tokens": total_tokens,
        "seq_len": seq_len,
        "tokenizer": tok_path,
    }
    with open(out_dir / "manifest.json", "wb") as mf:
        mf.write(orjson.dumps(manifest, option=orjson.OPT_INDENT_2))

    print(f"docs={total_docs} tokens={total_tokens} shards={shard_idx} -> {out_dir}")


if __name__ == "__main__":
    main()
