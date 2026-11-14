"""Baseline: Facebook's pre-aligned fastText Wikipedia vectors, looked up for our vocabulary.

Download wiki.{en,es,ja}.align.vec from https://fasttext.cc/docs/en/aligned-vectors.html
(several GB each; text format, first line "<n_words> <dim>"). They are already in one shared
space, trained on far more text than our corpus, but static (one vector per word form) and
keyed by surface form, so lemmas are matched as-is and, for en/es, lowercased.
Japanese entries follow fastText's own segmentation, so coverage for ja lemmas is lower.

    python -m wikiembed.fasttext --vec en=path/wiki.en.align.vec es=... ja=...
writes data/embed/fasttext.npz: `vectors` aligned with vocab.jsonl rows, `found` boolean mask.
"""
import argparse
from pathlib import Path

import numpy as np

from wikicollect.store import read_jsonl


def lookup(vec_path: Path, words: set[str]) -> dict[str, np.ndarray]:
    found = {}
    with open(vec_path, encoding="utf-8", errors="replace") as f:
        n, dim = map(int, f.readline().split())
        for i, line in enumerate(f):
            word, rest = line.split(" ", 1)
            if word in words and word not in found:
                found[word] = np.array(rest.split(), dtype=np.float32)
                if len(found) == len(words):
                    break
            if i % 200_000 == 0:
                print(f"  {vec_path.name}: {i}/{n} lines, {len(found)}/{len(words)} found", end="\r")
    print()
    return found


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vec", nargs="+", required=True, help="lang=path pairs")
    ap.add_argument("--out-dir", type=Path, default=Path("data/embed"))
    args = ap.parse_args()

    vocab = list(read_jsonl(args.out_dir / "vocab.jsonl"))
    key = lambda r: r["lemma"] if r["lang"] == "ja" else r["lemma"].lower()  # noqa: E731
    vectors, found = None, np.zeros(len(vocab), dtype=bool)
    for pair in args.vec:
        lang, path = pair.split("=", 1)
        rows = [r for r in vocab if r["lang"] == lang]
        table = lookup(Path(path), {key(r) for r in rows})
        for r in rows:
            v = table.get(key(r))
            if v is None:
                continue
            if vectors is None:
                vectors = np.zeros((len(vocab), len(v)), dtype=np.float32)
            vectors[r["idx"]] = v
            found[r["idx"]] = True
        print(f"  {lang}: {sum(found[r['idx']] for r in rows)}/{len(rows)} vocabulary words found")
    np.savez(args.out_dir / "fasttext.npz", vectors=vectors, found=found)
    print(f"wrote {args.out_dir / 'fasttext.npz'}")


if __name__ == "__main__":
    main()
