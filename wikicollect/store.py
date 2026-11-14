"""JSONL helpers and the run manifest."""
import json
import time
from collections.abc import Iterable, Iterator
from pathlib import Path


def read_jsonl(path: Path) -> Iterator[dict]:
    if not path.exists():
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: Path, records: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def update_manifest(out_dir: Path, stage: str, config: dict, **counts) -> None:
    path = out_dir / "manifest.json"
    manifest = json.loads(path.read_text("utf-8")) if path.exists() else {}
    manifest["config"] = config
    manifest.setdefault("stages", {})[stage] = {
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **counts
    }
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), "utf-8")
