#!/usr/bin/env python3
"""Stage 3: FineWeb-Edu quality scoring on GPU (streaming)."""

import argparse
from pathlib import Path

import orjson
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from common import load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--min-score", type=float, default=None)
    ap.add_argument("--max-length", type=int, default=None)
    ap.add_argument("--model", default=None)
    args = ap.parse_args()

    cfg = load_config()
    model_id = args.model or cfg["classifier_model"]
    batch_size = args.batch_size or cfg["classifier_batch_size"]
    min_score = args.min_score if args.min_score is not None else cfg["classifier_min_score"]
    max_length = args.max_length or cfg["classifier_max_length"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: running classifier on CPU — use CUDA_VISIBLE_DEVICES for GPU scoring")

    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id).to(device).eval()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    with open(args.input, "rb") as f:
        for line in f:
            if line.strip():
                rows.append(orjson.loads(line))

    kept = 0
    with open(out_path, "wb") as fout:
        for i in tqdm(range(0, len(rows), batch_size), desc="classify"):
            batch = rows[i:i + batch_size]
            texts = [(r.get("text") or "")[:8000] for r in batch]
            enc = tok(
                texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            with torch.inference_mode():
                logits = model(**enc).logits
                if logits.shape[-1] == 1:
                    scores = logits.squeeze(-1).float().cpu().tolist()
                else:
                    scores = logits[:, 0].float().cpu().tolist()

            for row, score in zip(batch, scores):
                row["quality_score"] = float(score)
                if score >= min_score:
                    fout.write(orjson.dumps(row) + b"\n")
                    kept += 1

    print(f"scored kept={kept}/{len(rows)} min_score={min_score} -> {out_path}")


if __name__ == "__main__":
    main()
