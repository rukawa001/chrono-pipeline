#!/usr/bin/env python3
"""Stage 4: split long documents into chunks (no LLM)."""

import argparse
import re

import orjson

from common import load_config


def split_paragraphs(text: str, max_chars: int) -> list[str]:
    parts = re.split(r"\n\s*\n", text)
    chunks: list[str] = []
    cur = ""
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if len(cur) + len(p) + 2 <= max_chars:
            cur = (cur + "\n\n" + p).strip()
        else:
            if cur:
                chunks.append(cur)
            cur = p if len(p) <= max_chars else p[:max_chars]
    if cur:
        chunks.append(cur)
    return chunks or [text[:max_chars]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--max-chars", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config()
    max_chars = args.max_chars or cfg["chunk_max_chars"]

    with open(args.input, "rb") as fin, open(args.output, "wb") as fout:
        for line in fin:
            row = orjson.loads(line)
            text = row.get("text") or ""
            if len(text) <= max_chars:
                fout.write(orjson.dumps(row) + b"\n")
                continue
            for i, ch in enumerate(split_paragraphs(text, max_chars)):
                out = dict(row)
                out["text"] = ch
                out["chunk_idx"] = i
                out["parent_url"] = row.get("url")
                fout.write(orjson.dumps(out) + b"\n")


if __name__ == "__main__":
    main()
