#!/usr/bin/env bash
# Merge part_A + part_B into 900k training mix (720k 2017 + 180k replay @ 20%).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ -f "$ROOT/.venv-corpus/bin/activate" ]]; then
  source "$ROOT/.venv-corpus/bin/activate"
fi

export PYTHONPATH="$ROOT/corpus_pipeline:${PYTHONPATH:-}"

PART_A="${PART_A:-$ROOT/../corpus/processed/part_A/final}"
PART_B="${PART_B:-$ROOT/../corpus/processed/part_B/final}"
OUTPUT="${OUTPUT:-$ROOT/../corpus/processed/train_2017_mix.jsonl}"

python "$ROOT/corpus_pipeline/merge_mix.py" \
  --part-a "$PART_A" \
  --part-b "$PART_B" \
  --output "$OUTPUT"

wc -l "$OUTPUT"
