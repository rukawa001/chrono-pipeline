#!/usr/bin/env python3
"""Continual training on 1 GPU from nanoexpand-2016 checkpoint."""

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import IterableDataset, DataLoader

ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = ROOT.parent
sys.path.insert(0, str(REPO_ROOT / "sn38"))

import sn38.architectures  # noqa: F401 — register nanoexpand
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_train_config(path: Path | None = None) -> dict:
    cfg_path = path or ROOT / "train" / "config_train.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)


class PackedBinDataset(IterableDataset):
    """Yield random contiguous seq_len windows from .bin shards."""

    def __init__(self, bin_dir: Path, seq_len: int, seed: int = 42):
        self.bin_dir = bin_dir
        self.seq_len = seq_len
        self.rng = np.random.default_rng(seed)
        self.files = sorted(bin_dir.glob("train_*.bin"))

    def __iter__(self):
        worker = torch.utils.data.get_worker_info()
        files = self.files
        if worker is not None:
            files = files[worker.id :: worker.num_workers]

        for fp in files:
            data = np.fromfile(fp, dtype=np.uint16)
            if len(data) <= self.seq_len:
                continue
            max_start = len(data) - self.seq_len - 1
            # ~1 sample per seq_len tokens per epoch pass
            n_samples = max(1, len(data) // self.seq_len)
            for _ in range(n_samples):
                start = int(self.rng.integers(0, max_start + 1))
                chunk = data[start:start + self.seq_len + 1]
                x = torch.from_numpy(chunk[:-1].astype(np.int64))
                y = torch.from_numpy(chunk[1:].astype(np.int64))
                yield x, y


def freeze_extra_mlps(model, indices: list[str]):
    for name, param in model.named_parameters():
        if "extra_mlps" not in name:
            continue
        for idx in indices:
            if f"extra_mlps.{idx}." in name:
                param.requires_grad = False
                break


def build_param_groups(model, cfg: dict):
    base_lr = cfg["learning_rate"]
    extra_lr = cfg.get("learning_rate_extra_mlp", base_lr * 5)
    base_params = []
    extra_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "extra_mlps" in name:
            extra_params.append(p)
        else:
            base_params.append(p)
    groups = []
    if base_params:
        groups.append({"params": base_params, "lr": base_lr})
    if extra_params:
        groups.append({"params": extra_params, "lr": extra_lr})
    return groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--base-checkpoint", default=None)
    ap.add_argument("--data", default=None)
    ap.add_argument("--output", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()

    cfg = load_train_config(Path(args.config) if args.config else None)

    base_ckpt = args.base_checkpoint or cfg["base_checkpoint"]
    base_ckpt = str((ROOT / base_ckpt).resolve() if not Path(base_ckpt).is_absolute() else Path(base_ckpt))

    data_dir = Path(args.data or cfg["token_bins"])
    if not data_dir.is_absolute():
        data_dir = (ROOT / data_dir).resolve()

    out_dir = Path(args.output or cfg["output_dir"])
    if not out_dir.is_absolute():
        out_dir = (ROOT / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    epochs = args.epochs or cfg["epochs"]
    seq_len = cfg["seq_len"]
    batch_size = cfg["batch_size"]
    grad_accum = cfg.get("gradient_accumulation", 1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("WARNING: training without CUDA")

    model = AutoModelForCausalLM.from_pretrained(
        base_ckpt, torch_dtype=torch.bfloat16
    ).to(device)
    tok = AutoTokenizer.from_pretrained(base_ckpt)

    freeze_extra_mlps(model, cfg.get("freeze_extra_mlps", []))
    opt = torch.optim.AdamW(
        build_param_groups(model, cfg),
        weight_decay=cfg.get("weight_decay", 0.1),
    )
    base_lrs = [g["lr"] for g in opt.param_groups]

    ds = PackedBinDataset(data_dir, seq_len)
    dl = DataLoader(ds, batch_size=batch_size, num_workers=2)

    model.train()
    global_step = 0
    warmup = cfg.get("warmup_steps", 500)
    max_grad_norm = cfg.get("max_grad_norm", 1.0)

    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_steps = 0
        opt.zero_grad(set_to_none=True)
        accum_loss = 0.0

        for step, (x, y) in enumerate(dl):
            x = x.to(device)
            y = y.to(device)
            out = model(input_ids=x, labels=y)
            loss = out.loss / grad_accum
            loss.backward()
            accum_loss += out.loss.item()

            if (step + 1) % grad_accum != 0:
                continue

            if global_step < warmup:
                scale = min(1.0, (global_step + 1) / warmup)
                for g, base_lr in zip(opt.param_groups, base_lrs):
                    g["lr"] = base_lr * scale

            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], max_grad_norm
            )
            opt.step()
            opt.zero_grad(set_to_none=True)

            epoch_loss += accum_loss
            epoch_steps += 1
            global_step += 1
            accum_loss = 0.0

            if global_step % 50 == 0:
                print(f"epoch={epoch} step={global_step} loss={epoch_loss / max(1, epoch_steps):.4f}")

        avg = epoch_loss / max(1, epoch_steps)
        print(f"Epoch {epoch} avg_loss={avg:.4f}")

        ep_out = out_dir / f"epoch_{epoch}"
        model.save_pretrained(ep_out)
        tok.save_pretrained(ep_out)
        print(f"Saved {ep_out}")

    final_out = out_dir / "final"
    model.save_pretrained(final_out)
    tok.save_pretrained(final_out)
    print(f"Done -> {final_out}")


if __name__ == "__main__":
    main()
