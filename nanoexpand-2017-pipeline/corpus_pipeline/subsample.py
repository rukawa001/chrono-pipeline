#!/usr/bin/env python3
"""Random subsample with fixed seed. probability=0.50 keeps ~50% of input rows."""

import argparse
import random

from common import load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--probability", type=float, required=True)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config()
    seed = args.seed if args.seed is not None else cfg["subsample_seed"]
    rng = random.Random(seed)

    kept = 0
    total = 0
    with open(args.input, "rb") as fin, open(args.output, "wb") as fout:
        for line in fin:
            if not line.strip():
                continue
            total += 1
            if rng.random() < args.probability:
                fout.write(line)
                kept += 1

    print(f"subsample kept={kept}/{total} p={args.probability} seed={seed}")


if __name__ == "__main__":
    main()
