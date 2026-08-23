#!/usr/bin/env python3
"""Merge part_A + part_B finals into train_2017_mix.jsonl."""

import argparse
import random
from pathlib import Path

import orjson

from common import load_config


def load_pool(dir_path: Path, year: str) -> list[dict]:
    p = dir_path / f"{year}_train.jsonl"
    if not p.exists():
        return []
    rows = []
    with open(p, "rb") as f:
        for line in f:
            if line.strip():
                rows.append(orjson.loads(line))
    return rows


def sample_rows(pool: list[dict], n: int, rng: random.Random) -> list[dict]:
    if n <= 0 or not pool:
        return []
    if n >= len(pool):
        return list(pool)
    return rng.sample(pool, n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part-a", required=True, help="part_A/final directory")
    ap.add_argument("--part-b", required=True, help="part_B/final directory")
    ap.add_argument("--output", required=True)
    ap.add_argument("--total", type=int, default=None)
    ap.add_argument("--replay-fraction", type=float, default=None)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config()
    n_total = args.total or cfg["mix_total_rows"]
    replay_frac = args.replay_fraction if args.replay_fraction is not None else cfg["mix_replay_fraction"]
    seed = args.seed if args.seed is not None else cfg["subsample_seed"]
    weights = cfg["mix_replay_weights"]
    primary_year = str(cfg["mix_primary_year"])

    rng = random.Random(seed)
    part_a = Path(args.part_a)
    part_b = Path(args.part_b)

    # Primary 2017 lives on part B; fallback part A if missing
    p2017 = load_pool(part_b, primary_year)
    if not p2017:
        p2017 = load_pool(part_a, primary_year)
    if not p2017:
        raise SystemExit(f"No {primary_year}_train.jsonl in part_a or part_b")

    replay_pools = {
        "2013": load_pool(part_a, "2013"),
        "2014": load_pool(part_a, "2014"),
        "2015": load_pool(part_a, "2015"),
        "2016": load_pool(part_b, "2016"),
    }

    n_replay = int(n_total * replay_frac)
    n_primary = n_total - n_replay

    primary = sample_rows(p2017, n_primary, rng)
    if len(primary) < n_primary:
        print(f"WARNING: primary pool {len(p2017)} < requested {n_primary}, using {len(primary)}")

    replay: list[dict] = []
    for year, w in weights.items():
        n = int(n_replay * w)
        picked = sample_rows(replay_pools[year], n, rng)
        if len(picked) < n:
            print(f"WARNING: replay {year} pool {len(replay_pools[year])} < requested {n}")
        replay.extend(picked)

    rng.shuffle(primary)
    rng.shuffle(replay)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as fout:
        for r in primary:
            r["mix_role"] = "primary"
            fout.write(orjson.dumps(r) + b"\n")
        for r in replay:
            r["mix_role"] = "replay"
            fout.write(orjson.dumps(r) + b"\n")

    print(
        f"mix primary={len(primary)} replay={len(replay)} "
        f"total={len(primary) + len(replay)} -> {out_path}"
    )


if __name__ == "__main__":
    main()
