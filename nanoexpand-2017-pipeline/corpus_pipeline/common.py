"""Shared helpers for corpus pipeline."""

import re
from pathlib import Path

import orjson
import yaml

YEAR_RE = re.compile(r"^(\d{4})")

ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = ROOT.parent


def load_config(path: Path | None = None) -> dict:
    cfg_path = path or Path(__file__).parent / "config.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def resolve_path(p: str) -> Path:
    path = Path(p)
    if path.is_absolute():
        return path
    return (ROOT / path).resolve()


def make_text(row: dict) -> str:
    t = (row.get("title") or "").strip()
    c = (row.get("content") or "").strip()
    if t and c:
        return f"{t}\n\n{c}"
    return t or c


def year_from_date(published_date: str | None) -> int | None:
    if not published_date:
        return None
    m = YEAR_RE.match(published_date.strip())
    return int(m.group(1)) if m else None


def write_jsonl(path: Path, record: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "ab") as f:
        f.write(orjson.dumps(record) + b"\n")


def read_jsonl(path: Path):
    with open(path, "rb") as f:
        for line in f:
            line = line.strip()
            if line:
                yield orjson.loads(line)
