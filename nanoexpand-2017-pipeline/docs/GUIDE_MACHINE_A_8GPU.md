# Machine A — Corpus build (8× A100)

## Assignment

| Setting | Value |
|---------|--------|
| `PART` | `A` |
| Years | `2016`, `2013`, `2014`, `2015` |
| Subsampled raw rows | ~**1.0M** (2016 25%, others 50%) |
| Est. wall time | **~5–8 hours** |

Machine B handles `2017` + `upto2012` (~1.05M subsampled raw rows).

## Prerequisites

- Repo cloned with `corpus/resource/*.jsonl`
- 8× A100, ~200 vCPU, ~940 GB RAM recommended
- NVIDIA driver + CUDA 12.x

## One-time setup

```bash
cd /path/to/chrono-round8/nanoexpand-2017-pipeline
python3 -m venv .venv-corpus
source .venv-corpus/bin/activate
pip install -U pip wheel
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements-corpus.txt

mkdir -p logs corpus/.hf_cache
export HF_HOME="$(pwd)/corpus/.hf_cache"
```

## Run (no analysis — config is fixed)

```bash
cd /path/to/chrono-round8/nanoexpand-2017-pipeline
source .venv-corpus/bin/activate
export PART=A
export HF_HOME="$(pwd)/corpus/.hf_cache"

nohup ./corpus_pipeline/run_part.sh > logs/corpus_part_A.log 2>&1 &
tail -f logs/corpus_part_A.log
```

Monitor GPUs:

```bash
watch -n2 'nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv'
```

Expect **8 GPUs busy** during Stage 3 (classifier) and Stage 5 (LLM salvage).

## Output

```
corpus/processed/part_A/final/
  2016_train.jsonl   # 25% of raw (~560k rows)
  2013_train.jsonl   # 50% of raw (~153k rows)
  2014_train.jsonl
  2015_train.jsonl
```

Copy entire `corpus/processed/part_A/` to the train machine (or rsync `final/` only).

```bash
rsync -avz corpus/processed/part_A/ user@train-host:/path/chrono-round8/corpus/processed/part_A/
```

## Pipeline stages (automatic)

1. **Raw subsample** — 2016 **25%**, 2013–2015 **50%**
2. Rules — drop ABC boilerplate, wrong year, stubs
3. Dedup — URL + MinHash
4. Classifier — 8 GPU parallel (FineWeb-Edu, score ≥ 2.0)
5. Chunk long rows + score chunks
6. LLM salvage — 8 GPU parallel on borderline rows

## Troubleshooting

| Issue | Fix |
|-------|-----|
| Only GPU 0 active | Check `NUM_GPUS=8` in `run_part.sh`; 8 scorer/vLLM jobs must run |
| OOM on classifier | Lower `CLASSIFIER_BATCH_SIZE=128` |
| OOM on vLLM | One 7B model per GPU (default); reduce `gpu_memory_utilization` in `llm_salvage.py` |
| Slow Stage 1 | Increase `CPU_WORKERS=64` |

## After completion

Notify train machine operator to run `merge_parts.sh` once **both** part A and part B are done.
