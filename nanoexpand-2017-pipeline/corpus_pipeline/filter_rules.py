#!/usr/bin/env python3
"""Stage 1: rule filter and route rows into buckets."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import orjson

from common import load_config, make_text, resolve_path, year_from_date


def classify_row(row: dict, expect_year: int, cfg: dict) -> tuple[str, list[str], str]:
    text = make_text(row)
    reasons: list[str] = []
    blocklist = cfg["blocklist"]
    min_chars = cfg["min_chars"]
    ready_min = cfg["train_ready_min_chars"]
    chunk_thr = cfg["chunk_threshold_chars"]

    y = year_from_date(row.get("published_date"))
    if y is None:
        reasons.append("bad_date")
    elif y != expect_year:
        reasons.append("wrong_year")

    content = (row.get("content") or "").strip()
    if not content:
        reasons.append("empty_content")

    for phrase in blocklist:
        if phrase in text or phrase in content:
            reasons.append("boilerplate")
            break

    if reasons:
        return "dropped", reasons, text

    n = len(text)
    if n < min_chars:
        return "dropped", ["too_short"], text
    if n < ready_min:
        return "needs_llm", [], text
    if n > chunk_thr:
        return "needs_chunk", [], text
    return "train_ready", [], text


def process_chunk(lines: list[bytes], expect_year: int, cfg: dict, is_upto2012: bool):
    out: list[tuple[str, dict]] = []
    for line in lines:
        if not line.strip():
            continue
        row = orjson.loads(line)
        if is_upto2012:
            y = year_from_date(row.get("published_date"))
            if y is None or y > 2012:
                rec = {
                    "text": make_text(row),
                    "published_date": row.get("published_date"),
                    "year": y,
                    "sitename": row.get("sitename"),
                    "url": row.get("url"),
                    "route": "dropped",
                    "drop_reasons": ["wrong_year_upto2012"],
                }
                out.append(("dropped", rec))
                continue

        route, reasons, text = classify_row(row, expect_year, cfg)
        y = year_from_date(row.get("published_date"))
        rec = {
            "text": text,
            "published_date": row.get("published_date"),
            "year": y,
            "sitename": row.get("sitename"),
            "url": row.get("url"),
            "route": route,
            "drop_reasons": reasons,
        }
        out.append((route if route != "dropped" else "dropped", rec))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year-file", required=True, help="e.g. 2017 or upto2012")
    ap.add_argument("--out-root", required=True, help="e.g. corpus/processed/part_A")
    ap.add_argument("--config", default=None)
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(Path(args.config) if args.config else None)
    raw_dir = resolve_path(cfg["raw_dir"])
    out_dir = Path(args.out_root) / "stage1_rules"
    out_dir.mkdir(parents=True, exist_ok=True)

    year_file = args.year_file
    in_path = raw_dir / f"{year_file}.jsonl"
    if not in_path.exists():
        raise SystemExit(f"Missing {in_path}")

    if year_file == "upto2012":
        expect_year = 2012
        is_upto = True
    else:
        expect_year = int(year_file)
        is_upto = False

    workers = args.workers or cfg["cpu_workers"]
    handles = {
        k: open(out_dir / f"{year_file}_{k}.jsonl", "wb")
        for k in ("train_ready", "needs_chunk", "needs_llm", "dropped")
    }

    with open(in_path, "rb") as f:
        lines = f.readlines()

    chunk_size = max(1, (len(lines) + workers - 1) // workers)
    chunks = [lines[i:i + chunk_size] for i in range(0, len(lines), chunk_size)]

    counts = {k: 0 for k in handles}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [
            ex.submit(process_chunk, ch, expect_year, cfg, is_upto)
            for ch in chunks
        ]
        for fut in as_completed(futures):
            for route, rec in fut.result():
                bucket = route if route in handles else "dropped"
                handles[bucket].write(orjson.dumps(rec) + b"\n")
                counts[bucket] += 1

    for h in handles.values():
        h.close()

    print(
        f"{year_file}: train_ready={counts['train_ready']} "
        f"needs_chunk={counts['needs_chunk']} needs_llm={counts['needs_llm']} "
        f"dropped={counts['dropped']}"
    )


if __name__ == "__main__":
    main()
