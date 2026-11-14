"""Build the compact release artifact that the l2diff prototype runs on.

From the full embedding output (data/embed_all, ~850 MB) keep, per language, the 20,000 most frequent
content/function words and the 2,000 most frequent proper nouns, as float16 mean-centered vectors
(~100 MB), plus the reconstruction scores and settings. The reconstruction uses each language's top
15,000 content/function words, so results are unchanged.

Usage: python scripts/make_release.py [--src data/embed_all] [--version v0.1]
Writes artifacts/release/ and artifacts/l2diff-artifacts-<version>.zip, and records the zip's SHA-256
in artifacts.lock.json (committed, so scripts/fetch_artifacts.py can verify downloads).
"""
import argparse
import hashlib
import json
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from wikicollect.store import read_jsonl, write_jsonl  # noqa: E402
from wikiembed.reconstruct import CONTENT_UPOS, FUNCTION_UPOS  # noqa: E402
from wikiembed.vectors import NATIVE_SCRIPT, lemma_script  # noqa: E402

REPO = "cocorowachi/L2-SLA-Hardness-Evaluator"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=ROOT / "data" / "embed_all")
    ap.add_argument("--version", default="v0.1")
    ap.add_argument("--content", type=int, default=20000, help="content/function words kept per language")
    ap.add_argument("--propn", type=int, default=2000, help="proper nouns kept per language")
    args = ap.parse_args()

    data = np.load(args.src / "vectors.npz")
    vocab = list(read_jsonl(args.src / "vocab.jsonl"))
    keep = []
    for lang in [str(l) for l in data["langs"]]:
        rows = [r for r in vocab if r["lang"] == lang and lemma_script(r["lemma"]) == NATIVE_SCRIPT[lang]]
        rows.sort(key=lambda r: -r["count"])  # stable: ties keep vocab order, as top_rows does
        cf = [r for r in rows if r["upos"] in CONTENT_UPOS | FUNCTION_UPOS][:args.content]
        pn = [r for r in rows if r["upos"] == "PROPN"][:args.propn]
        keep += sorted(cf + pn, key=lambda r: r["idx"])
    keep.sort(key=lambda r: r["idx"])

    out = ROOT / "artifacts" / "release"
    if out.exists():
        shutil.rmtree(out)
    (out / "reconstruction").mkdir(parents=True)
    idx = np.array([r["idx"] for r in keep])
    np.savez(out / "vectors.npz", centered=data["centered"][idx].astype(np.float16),
             lang_means=data["lang_means"].astype(np.float32), langs=data["langs"])
    write_jsonl(out / "vocab.jsonl", [r | {"idx": i, "orig_idx": r["idx"]} for i, r in enumerate(keep)])
    for f in (args.src / "reconstruction").glob("*"):
        shutil.copy2(f, out / "reconstruction" / f.name)
    shutil.copy2(args.src / "embed_config.json", out / "embed_config.json")
    (out / "RELEASE.json").write_text(json.dumps({
        "version": args.version, "vectors": len(keep), "dtype": "float16", "dims": int(data["centered"].shape[1]),
        "per_language": {"content_function": args.content, "propn": args.propn},
        "source": "data/embed_all (XLM-R layer 8 type vectors over 2,000 en/es/ja Wikipedia triples)",
        "license": "Derived from Wikipedia text (CC BY-SA 4.0)"}, indent=2), "utf-8")

    zip_path = ROOT / "artifacts" / f"l2diff-artifacts-{args.version}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(out.parent).as_posix())
    sha = hashlib.sha256(zip_path.read_bytes()).hexdigest()
    lock = {"version": args.version, "file": zip_path.name, "sha256": sha, "bytes": zip_path.stat().st_size,
            "url": f"https://github.com/{REPO}/releases/download/{args.version}/{zip_path.name}"}
    (ROOT / "artifacts.lock.json").write_text(json.dumps(lock, indent=2) + "\n", "utf-8")
    print(f"{len(keep)} vectors -> {out}")
    print(f"wrote {zip_path.relative_to(ROOT)} ({zip_path.stat().st_size / 1e6:.0f} MB), sha256 {sha[:16]}...")


if __name__ == "__main__":
    main()
