#!/usr/bin/env python3
"""Stage 2: URL dedup + MinHash near-dup removal."""

import argparse
from pathlib import Path

import orjson
from datasketch import MinHash, MinHashLSH

from common import load_config


def shingles(text: str, n: int = 5) -> list[str]:
    words = text.lower().split()
    if len(words) < n:
        return [text.lower()]
    return [" ".join(words[i:i + n]) for i in range(len(words) - n + 1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--threshold", type=float, default=None)
    args = ap.parse_args()

    cfg = load_config()
    threshold = args.threshold or cfg["dedup_minhash_threshold"]

    seen_url: set[str] = set()
    lsh = MinHashLSH(threshold=threshold, num_perm=128)
    kept = 0
    dropped = 0
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(args.input, "rb") as fin, open(out_path, "wb") as fout:
        for line in fin:
            row = orjson.loads(line)
            url = row.get("url")
            if url and url in seen_url:
                dropped += 1
                continue

            text = row.get("text") or ""
            mh = MinHash(num_perm=128)
            for s in shingles(text):
                mh.update(s.encode("utf-8"))

            key = f"doc-{kept}"
            if lsh.query(mh):
                dropped += 1
                continue

            lsh.insert(key, mh)
            if url:
                seen_url.add(url)
            fout.write(orjson.dumps(row) + b"\n")
            kept += 1

    print(f"dedup kept={kept} dropped={dropped} -> {out_path}")


if __name__ == "__main__":
    main()
