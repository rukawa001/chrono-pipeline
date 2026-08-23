#!/usr/bin/env bash
# Run corpus pipeline for one machine part (8x A100).
# Usage:
#   export PART=A   # Machine A: 2016 only (early 40% subsample)
#   export PART=B   # Machine B: 2017, upto2012, 2013, 2014, 2015
#   ./corpus_pipeline/run_part.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PART="${PART:?Set PART=A or PART=B}"
NUM_GPUS="${NUM_GPUS:-8}"
CPU_WORKERS="${CPU_WORKERS:-48}"
BATCH="${CLASSIFIER_BATCH_SIZE:-256}"
SUBSAMPLE_DEFAULT="${SUBSAMPLE_DEFAULT:-0.70}"
SUBSAMPLE_2016="${SUBSAMPLE_2016:-0.40}"
SUBSAMPLE_SEED="${SUBSAMPLE_SEED:-42}"

if [[ -f "$ROOT/.venv-corpus/bin/activate" ]]; then
  source "$ROOT/.venv-corpus/bin/activate"
fi

export PYTHONPATH="$ROOT/corpus_pipeline:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-$ROOT/../corpus/.hf_cache}"
mkdir -p "$HF_HOME" "$ROOT/logs"

OUT="$(cd "$ROOT/../corpus/processed/part_${PART}" && pwd)"
mkdir -p "$OUT/stage1_rules" "$OUT/stage2_dedup" "$OUT/shards" \
         "$OUT/stage3_scored" "$OUT/stage4_chunked" "$OUT/stage5_llm" "$OUT/final"

if [[ "$PART" == "A" ]]; then
  YEARS=(2016)
elif [[ "$PART" == "B" ]]; then
  YEARS=(2017 upto2012 2013 2014 2015)
else
  echo "PART must be A or B"
  exit 1
fi

log() { echo "[$(date -Iseconds)] $*"; }

subsample_buckets() {
  local year="$1"
  local prob="$2"
  for bucket in train_ready needs_chunk needs_llm; do
    in="$OUT/stage1_rules/${year}_${bucket}.jsonl"
    if [[ ! -f "$in" ]]; then
      continue
    fi
    tmp="$OUT/stage1_rules/${year}_${bucket}_sub.jsonl"
    python "$ROOT/corpus_pipeline/subsample.py" \
      --input "$in" --output "$tmp" --probability "$prob" --seed "$SUBSAMPLE_SEED"
    mv "$tmp" "$in"
  done
}

log "=== PART $PART years: ${YEARS[*]} ==="

# Stage 1: rules
log "Stage 1: rules"
for y in "${YEARS[@]}"; do
  python "$ROOT/corpus_pipeline/filter_rules.py" \
    --year-file "$y" \
    --out-root "$OUT" \
    --workers "$CPU_WORKERS" &
done
wait

# Stage 1b: early subsample 2016 (before dedup / classifier / LLM)
if [[ " ${YEARS[*]} " == *" 2016 "* ]]; then
  log "Stage 1b: early subsample 2016 at ${SUBSAMPLE_2016} (before heavy stages)"
  subsample_buckets 2016 "$SUBSAMPLE_2016"
fi

# Stage 2: dedup
log "Stage 2: dedup"
for y in "${YEARS[@]}"; do
  in="$OUT/stage1_rules/${y}_train_ready.jsonl"
  if [[ -f "$in" ]]; then
    python "$ROOT/corpus_pipeline/dedup.py" \
      --input "$in" \
      --output "$OUT/stage2_dedup/${y}_deduped.jsonl" &
  fi
done
wait

# Stage 3: quality score — 8 GPU data parallel per year
log "Stage 3: classifier (8 GPUs)"
for y in "${YEARS[@]}"; do
  in="$OUT/stage2_dedup/${y}_deduped.jsonl"
  if [[ ! -f "$in" ]]; then
    continue
  fi
  rm -f "$OUT/shards/${y}_part_"* "$OUT/stage3_scored/${y}_scored_"*.jsonl
  split -n "l/$NUM_GPUS" -d -a 1 "$in" "$OUT/shards/${y}_part_"
  for gpu in $(seq 0 $((NUM_GPUS - 1))); do
    shard="$OUT/shards/${y}_part_${gpu}"
    if [[ ! -f "$shard" ]]; then
      continue
    fi
    CUDA_VISIBLE_DEVICES=$gpu python "$ROOT/corpus_pipeline/score_quality.py" \
      --input "$shard" \
      --output "$OUT/stage3_scored/${y}_scored_${gpu}.jsonl" \
      --batch-size "$BATCH" &
  done
  wait
  cat "$OUT/stage3_scored/${y}_scored_"*.jsonl > "$OUT/stage3_scored/${y}_scored.jsonl"
done

# Stage 4: assemble + chunk
log "Stage 4: chunk + assemble"
for y in "${YEARS[@]}"; do
  final="$OUT/final/${y}_train.jsonl"
  : > "$final"
  scored="$OUT/stage3_scored/${y}_scored.jsonl"
  if [[ -f "$scored" ]]; then
    cat "$scored" >> "$final"
  fi
  chunk_in="$OUT/stage1_rules/${y}_needs_chunk.jsonl"
  if [[ -f "$chunk_in" ]]; then
    python "$ROOT/corpus_pipeline/chunk_long.py" \
      --input "$chunk_in" \
      --output "$OUT/stage4_chunked/${y}_chunked.jsonl"
    rm -f "$OUT/shards/${y}_chunk_part_"* "$OUT/stage3_scored/${y}_chunk_scored_"*.jsonl
    split -n "l/$NUM_GPUS" -d -a 1 "$OUT/stage4_chunked/${y}_chunked.jsonl" \
      "$OUT/shards/${y}_chunk_part_"
    for gpu in $(seq 0 $((NUM_GPUS - 1))); do
      shard="$OUT/shards/${y}_chunk_part_${gpu}"
      if [[ ! -f "$shard" ]]; then
        continue
      fi
      CUDA_VISIBLE_DEVICES=$gpu python "$ROOT/corpus_pipeline/score_quality.py" \
        --input "$shard" \
        --output "$OUT/stage3_scored/${y}_chunk_scored_${gpu}.jsonl" \
        --batch-size "$BATCH" &
    done
    wait
    cat "$OUT/stage3_scored/${y}_chunk_scored_"*.jsonl >> "$final"
  fi
done

# Stage 5: LLM salvage — 8 GPU data parallel
log "Stage 5: LLM salvage (8 GPUs)"
for y in "${YEARS[@]}"; do
  llm_in="$OUT/stage1_rules/${y}_needs_llm.jsonl"
  if [[ ! -f "$llm_in" ]]; then
    continue
  fi
  if [[ "$y" == "upto2012" ]]; then
    cutoff_year=2012
  else
    cutoff_year="$y"
  fi
  rm -f "$OUT/shards/${y}_llm_part_"* "$OUT/stage5_llm/${y}_salvaged_"*.jsonl
  split -n "l/$NUM_GPUS" -d -a 1 "$llm_in" "$OUT/shards/${y}_llm_part_"
  for gpu in $(seq 0 $((NUM_GPUS - 1))); do
    shard="$OUT/shards/${y}_llm_part_${gpu}"
    if [[ ! -f "$shard" ]]; then
      continue
    fi
    CUDA_VISIBLE_DEVICES=$gpu python "$ROOT/corpus_pipeline/llm_salvage.py" \
      --input "$shard" \
      --output "$OUT/stage5_llm/${y}_salvaged_${gpu}.jsonl" \
      --year "$cutoff_year" &
  done
  wait
  if ls "$OUT/stage5_llm/${y}_salvaged_"*.jsonl 1>/dev/null 2>&1; then
    cat "$OUT/stage5_llm/${y}_salvaged_"*.jsonl >> "$OUT/final/${y}_train.jsonl"
  fi
done

# Stage 6: subsample non-2016 years (2016 already subsampled in Stage 1b)
log "Stage 6: subsample (70% for non-2016)"
for y in "${YEARS[@]}"; do
  if [[ "$y" == "2016" ]]; then
    continue
  fi
  in="$OUT/final/${y}_train.jsonl"
  if [[ ! -f "$in" ]]; then
    continue
  fi
  tmp="$OUT/final/${y}_train_sub.jsonl"
  python "$ROOT/corpus_pipeline/subsample.py" \
    --input "$in" --output "$tmp" --probability "$SUBSAMPLE_DEFAULT" --seed "$SUBSAMPLE_SEED"
  mv "$tmp" "$in"
done

log "DONE PART $PART -> $OUT/final/"
ls -lh "$OUT/final/"
