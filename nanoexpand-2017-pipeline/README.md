# NanoExpand 2017 — corpus build + continual training

All pipeline scripts live here. Raw data and outputs stay in the parent repo:

```
chrono-round8/
  corpus/resource/          # raw JSONL (input)
  corpus/processed/         # pipeline output + train mix
  nanoexpand-2016/          # base checkpoint
  sn38/                     # architecture + validation
  nanoexpand-2017-pipeline/ # this directory
```

## Quick start

```bash
cd nanoexpand-2017-pipeline
python3 -m venv .venv-corpus && source .venv-corpus/bin/activate
pip install -r requirements-corpus.txt

# Machine A (8× A100)
export PART=A && nohup ./corpus_pipeline/run_part.sh > logs/corpus_part_A.log 2>&1 &

# Machine B (8× A100)
export PART=B && nohup ./corpus_pipeline/run_part.sh > logs/corpus_part_B.log 2>&1 &

# Train machine (1× A100)
./corpus_pipeline/merge_parts.sh
python train/prepare_data.py
CUDA_VISIBLE_DEVICES=0 python train/train_continual.py
```

## Guides

- `docs/GUIDE_MACHINE_A_8GPU.md` — PART=A (2016 + 2013–2015)
- `docs/GUIDE_MACHINE_B_8GPU.md` — PART=B (2017 + upto2012)
- `docs/GUIDE_TRAIN_1GPU.md` — merge, tokenize, train

## Config

Fixed parameters in `corpus_pipeline/config.yaml` and `train/config_train.yaml` (900K mix, 20% replay, no re-analysis).
