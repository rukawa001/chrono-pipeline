# Machine B — Corpus build (8× A100)

## Assignment

| Setting | Value |
|---------|--------|
| `PART` | `B` |
| Years | `2017`, `upto2012`, `2013`, `2014`, `2015` |
| Est. wall time | **~7–10 hours** |

Machine A handles `2016` only (40% of raw rows at Stage 0).

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
  2017_train.jsonl      # from 70% of raw rows (Stage 0)
  upto2012_train.jsonl
  2013_train.jsonl
  2014_train.jsonl
  2015_train.jsonl
```

Sync to train machine:

```bash
rsync -avz corpus/processed/part_B/ user@train-host:/path/chrono-round8/corpus/processed/part_B/
```

## Notes

- **2017** primary training data lives on this machine.
- Replay years (`upto2012`–`2015`) are smaller and run in parallel with Machine A's 2016 job.
- Stage 0 raw subsample: **70%** of rows per year on PART B; **2016 on Machine A: 40%**

## GPU usage

| Stage | GPUs |
|-------|------|
| 3 Classifier | 8 data-parallel (`CUDA_VISIBLE_DEVICES=0..7`) |
| 5 LLM salvage | 8 data-parallel vLLM (Qwen2.5-7B, 1 GPU each) |

## Troubleshooting

| Issue | Fix |
|-------|-----|
| OOM on classifier | Lower `CLASSIFIER_BATCH_SIZE=128` |
| vLLM install fails | Try `pip install vllm==0.6.x` matching your CUDA |
| Need more 2017 rows | Raise `SUBSAMPLE_DEFAULT` in `run_part.sh` and re-run from Stage 0 |

## Re-run raw subsample only (if needed)

```bash
source .venv-corpus/bin/activate
RAW=../corpus/resource
OUT=../corpus/processed/part_B/stage0_raw
for y in 2017 upto2012 2013 2014 2015; do
  python corpus_pipeline/subsample.py \
    --input "$RAW/${y}.jsonl" \
    --output "$OUT/${y}.jsonl" \
    --probability 0.70 --seed 42
done
```
