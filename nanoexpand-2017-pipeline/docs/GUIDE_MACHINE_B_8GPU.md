# Machine B — Corpus build (8× A100)

## Assignment

| Setting | Value |
|---------|--------|
| `PART` | `B` |
| Years | `2016`, `2017` |
| Est. wall time | **~8–12 hours** (heaviest machine) |

## Prerequisites

Same as Machine A — see `docs/GUIDE_MACHINE_A_8GPU.md`.

## One-time setup

```bash
cd /path/to/chrono-round8/nanoexpand-2017-pipeline
python3 -m venv .venv-corpus
source .venv-corpus/bin/activate
pip install -U pip wheel
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements-corpus.txt

mkdir -p logs corpus/.hf_cache
```

## Run

```bash
cd /path/to/chrono-round8/nanoexpand-2017-pipeline
source .venv-corpus/bin/activate
export PART=B
export HF_HOME="$(pwd)/corpus/.hf_cache"

nohup ./corpus_pipeline/run_part.sh > logs/corpus_part_B.log 2>&1 &
tail -f logs/corpus_part_B.log
```

## Output

```
corpus/processed/part_B/final/
  2016_train.jsonl   # subsampled to 40% at end of pipeline
  2017_train.jsonl   # subsampled to 70%
```

Sync to train machine:

```bash
rsync -avz corpus/processed/part_B/ user@train-host:/path/chrono-round8/corpus/processed/part_B/
```

## Notes

- **2016** is the largest raw file (~2.2M rows). Dedup + LLM take the most time here.
- **2017** primary training data lives on this machine.
- Stage 6 subsample: `2016` → **40%**, `2017` → **70%** (fixed in `run_part.sh`).

## GPU usage

| Stage | GPUs |
|-------|------|
| 3 Classifier | 8 data-parallel (`CUDA_VISIBLE_DEVICES=0..7`) |
| 5 LLM salvage | 8 data-parallel vLLM (Qwen2.5-7B, 1 GPU each) |

## Troubleshooting

| Issue | Fix |
|-------|-----|
| Dedup OOM / slow | Run only `2016` dedup alone; ensure other jobs finished |
| vLLM install fails | Try `pip install vllm==0.6.x` matching your CUDA |
| 2016 subsample too aggressive | Edit `run_part.sh` prob `0.40` → `0.50` and re-run Stage 6 only |

## Re-run subsample only (if needed)

```bash
source .venv-corpus/bin/activate
OUT=corpus/processed/part_B
for y in 2016 2017; do
  prob=0.70; [[ "$y" == "2016" ]] && prob=0.40
  python corpus_pipeline/subsample.py \
    --input "$OUT/final/${y}_train_full.jsonl" \
    --output "$OUT/final/${y}_train.jsonl" \
    --probability "$prob" --seed 42
done
```

(Keep a backup of full `final/` before subsample if you want to re-tune.)
