# Train machine — 1× A100

## Overview

| Step | Time (typical) |
|------|----------------|
| Merge 900K mix | ~10 min |
| Tokenize bins | ~30–60 min |
| Train 2 epochs | ~8–14 h |
| **Total** | **~9–15 h** |

Assumes corpus parts A + B are already built on the 8×GPU machines.

## Prerequisites

1. Both corpus parts synced:
   - `corpus/processed/part_A/final/` — `2016`, `2013`, `2014`, `2015`
   - `corpus/processed/part_B/final/` — `2017`, `upto2012`
2. Base checkpoint: `nanoexpand-2016/`
3. `sn38/` package in repo (registers `sn38-nanoexpand`)

## Setup

```bash
cd /path/to/chrono-round8/nanoexpand-2017-pipeline
python3 -m venv .venv-train
source .venv-train/bin/activate
pip install -U pip wheel
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install transformers accelerate huggingface_hub orjson pyyaml numpy tqdm
pip install -r requirements-corpus.txt   # optional if only training

mkdir -p logs checkpoints
```

## Step 1 — Merge training mix (900K rows)

Fixed recipe (no re-analysis):

- **720,000** rows from `2017` (primary)
- **180,000** replay rows (20%): `2013`–`2016` (loaded from whichever part has each year; A=2016/2013–2015, B=2017 only for primary)

```bash
source .venv-train/bin/activate
export PYTHONPATH="$(pwd)/corpus_pipeline:$PYTHONPATH"

./corpus_pipeline/merge_parts.sh
# -> corpus/processed/train_2017_mix.jsonl
wc -l corpus/processed/train_2017_mix.jsonl
```

## Step 2 — Tokenize to .bin shards

```bash
python train/prepare_data.py
# -> corpus/processed/train_bins/
```

## Step 3 — Train (1 GPU, 2 epochs)

```bash
export PYTHONPATH="$(pwd)/sn38:$PYTHONPATH"
CUDA_VISIBLE_DEVICES=0 python train/train_continual.py
```

Defaults from `train/config_train.yaml`:

| Parameter | Value |
|-----------|--------|
| Base | `nanoexpand-2016` |
| Epochs | 2 |
| Batch | 4 × grad_accum 4 = effective 16 |
| LR base | 2e-5 |
| LR extra_mlp | 1e-4 |
| Frozen extra_mlps | 5,11,17,21,24,26,27 (from 2016 config) |
| Seq len | 2048 |

Checkpoints:

```
checkpoints/nanoexpand-2017/epoch_0/
checkpoints/nanoexpand-2017/epoch_1/
checkpoints/nanoexpand-2017/final/
```

## Step 4 — Validate

```bash
source .venv-corpus/bin/activate  # or .venv-train
export PYTHONPATH="$(pwd)/../sn38:$PYTHONPATH"

python ../sn38/debug/test_automodel.py checkpoints/nanoexpand-2017/final
```

Optional TEE self-test: see `sn38/docs/miner.md`.

## Tuning

Edit `train/config_train.yaml`:

| Knob | If… |
|------|-----|
| `epochs: 3` | 2017 unknown weak after 2 epochs |
| `learning_rate_extra_mlp: 2e-4` | New 2017 blocks learn slowly |
| `batch_size: 2` + `gradient_accumulation: 8` | OOM on 1× A100 |

## Full end-to-end timeline

| Phase | Where | Parallel? | Time |
|-------|-------|-----------|------|
| Corpus PART A | Machine A 8×GPU (2016 + 2013–2015) | Yes | ~5–8 h |
| Corpus PART B | Machine B 8×GPU (2017 + upto2012) | Yes | ~5–8 h |
| Merge + train | This machine 1×GPU | After corpus | ~9–15 h |
| **Wall clock** | | A ∥ B | **~14–23 h** |

## Upload for SN38

Export `checkpoints/nanoexpand-2017/final/` to HuggingFace:

- `config.json`, `model.safetensors`, `tokenizer.json`, etc.
- Pin commit SHA in `models.json` for subnet submission.

See `sn38/docs/miner.md`.
