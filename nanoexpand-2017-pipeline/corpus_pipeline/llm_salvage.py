#!/usr/bin/env python3
"""Stage 5: LLM salvage for borderline rows (extract/clean, no summarization)."""

import argparse
import json
from pathlib import Path

import orjson

from common import load_config

SYSTEM_PROMPT = """You clean news corpus rows for chronological LM pretraining.

Cutoff year: {year}. The model must only learn facts knowable on or before December 31 of that year.

Input is JSON with published_date, sitename, text, url.

Rules:
1. Remove UI boilerplate, video templates, navigation, social widgets.
2. Do NOT add new facts, names, dates, or events not present in the input.
3. Do NOT rewrite from a modern informed perspective.
4. If fewer than 2 informative sentences remain, return action=drop.
5. If content clearly requires post-cutoff knowledge, return action=drop.
6. Return strict JSON only: {{"action":"keep"|"drop","text":"...","reason":"..."}}
"""


def parse_verdict(raw: str) -> dict | None:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```", 2)[1]
        if raw.startswith("json"):
            raw = raw[4:]
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(raw[start:end + 1])
            except json.JSONDecodeError:
                return None
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--max-model-len", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config()
    model_name = args.model or cfg["llm_model"]
    max_model_len = args.max_model_len or cfg["llm_max_model_len"]

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=model_name,
        tensor_parallel_size=1,
        max_model_len=max_model_len,
        gpu_memory_utilization=0.90,
    )
    sp = SamplingParams(temperature=0, max_tokens=1024)

    rows: list[dict] = []
    with open(args.input, "rb") as f:
        for line in f:
            if line.strip():
                rows.append(orjson.loads(line))

    prompts = []
    for r in rows:
        user = json.dumps({
            "published_date": r.get("published_date"),
            "sitename": r.get("sitename"),
            "text": (r.get("text") or "")[:6000],
            "url": r.get("url"),
        }, ensure_ascii=False)
        prompts.append(
            f"<|im_start|>system\n{SYSTEM_PROMPT.format(year=args.year)}\n"
            f"<|im_start|>user\n{user}\n"
            f"<|im_start|>assistant\n"
        )

    outputs = llm.generate(prompts, sp)
    kept = 0
    blocklist = ["Now Playing:", "{{", "Facebook Messenger"]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "wb") as fout:
        for row, out in zip(rows, outputs):
            verdict = parse_verdict(out.outputs[0].text)
            if not verdict or verdict.get("action") != "keep":
                continue
            clean = (verdict.get("text") or "").strip()
            if len(clean) < 200:
                continue
            if any(b in clean for b in blocklist):
                continue
            row["text"] = clean
            row["llm_salvaged"] = True
            fout.write(orjson.dumps(row) + b"\n")
            kept += 1

    print(f"llm_salvage kept={kept}/{len(rows)} year={args.year} -> {out_path}")


if __name__ == "__main__":
    main()
