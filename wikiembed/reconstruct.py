"""Reconstruct L2 words from L1 words: a directional lexical-conceptual difficulty signal.

For a direction L1 -> L2 (an L1 speaker learning the L2), each frequent L2 word w is
approximated as a sparse non-negative combination of L1 word vectors:

    min_{a >= 0}  1/(2d) * ||v_w - B a||^2 + alpha * ||a||_1        (B = L1 basis, unit columns)

Per word:
  residual   ||v_w - B a||: the part of w no combination of L1 words covers (a gap)
  self       the same, but rebuilt from the L2's own other words: how unique w is anyway
  excess     residual - self: the gap attributable to the L1
  eff_n      effective number of L1 words used, exp(entropy of normalized a) (assembled from parts)
  collapse   take w's nearest L2 word w'; collapse = 1 - d(rec_w, rec_w') / d(v_w, v_w'),
             with d = cosine distance. Positive = the L1 view blurs a distinction the L2 makes
             (a split, e.g. ser/estar both rebuilt from English "be").

Fairness: every L1 basis has the same size (basis_size most frequent lemmas), and targets are
each L2's target_size most frequent lemmas. Proper nouns are excluded from both.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.linear_model import Lasso

from wikicollect.store import write_jsonl
from wikiembed.vectors import WordVectors

FUNCTION_UPOS = {"ADP", "AUX", "CCONJ", "DET", "PART", "PRON", "SCONJ"}
CONTENT_UPOS = {"NOUN", "VERB", "ADJ", "ADV"}


def top_rows(wv: WordVectors, lang: str, n: int) -> np.ndarray:
    rows = [r for r in wv.vocab if r["lang"] == lang and r["upos"] in FUNCTION_UPOS | CONTENT_UPOS]
    rows.sort(key=lambda r: -r["count"])
    return np.array([r["idx"] for r in rows[:n]])


def fit(v: np.ndarray, basis: np.ndarray, candidates: int, alpha: float):
    """Sparse non-negative reconstruction of v from the `candidates` most similar basis columns."""
    sims = basis @ v
    cand = np.argpartition(-sims, candidates)[:candidates]
    model = Lasso(alpha=alpha, positive=True, fit_intercept=False, max_iter=5000)
    model.fit(basis[cand].T, v)
    a = model.coef_
    rec = basis[cand].T @ a
    nz = a > 1e-6
    return cand[nz], a[nz], rec


def eff_n(a: np.ndarray) -> float:
    if a.sum() == 0:
        return 0.0
    p = a / a.sum()
    return float(np.exp(-(p * np.log(p)).sum()))


def cos_dist(x: np.ndarray, y: np.ndarray) -> float:
    nx, ny = np.linalg.norm(x), np.linalg.norm(y)
    return 1.0 - float(x @ y / (nx * ny)) if nx and ny else 1.0


def run_direction(wv: WordVectors, l1: str, l2: str, cfg: dict) -> list[dict]:
    basis_rows = top_rows(wv, l1, cfg["basis_size"])
    self_rows = top_rows(wv, l2, cfg["basis_size"])
    target_rows = top_rows(wv, l2, cfg["target_size"])
    B, S = wv.vectors[basis_rows], wv.vectors[self_rows]
    self_lemmas = np.array([wv.vocab[i]["lemma"] for i in self_rows])
    results, recs = [], np.zeros((len(target_rows), wv.vectors.shape[1]))
    for t, idx in enumerate(target_rows):
        r, v = wv.vocab[idx], wv.vectors[idx]
        atoms, a, rec = fit(v, B, cfg["candidates"], cfg["alpha"])
        # self baseline: L2's own basis minus every entry of the same lemma
        mask = self_lemmas != r["lemma"]
        _, _, rec_self = fit(v, S[mask], cfg["candidates"], cfg["alpha"])
        recs[t] = rec
        res, res_self = float(np.linalg.norm(v - rec)), float(np.linalg.norm(v - rec_self))
        order = np.argsort(-a)
        results.append({
            "lemma": r["lemma"], "upos": r["upos"], "count": r["count"],
            "class": "function" if r["upos"] in FUNCTION_UPOS else "content",
            "residual": res, "self": res_self, "excess": res - res_self, "eff_n": eff_n(a),
            "atoms": [[wv.vocab[basis_rows[atoms[i]]]["lemma"], wv.vocab[basis_rows[atoms[i]]]["upos"],
                       round(float(a[i]), 3)] for i in order[:6]],
        })
        if (t + 1) % 500 == 0:
            print(f"  {l1}->{l2}: {t + 1}/{len(target_rows)}", end="\r")
    print()

    # collapse: does the L1 view blur the distinction between w and its nearest L2 word?
    T = wv.vectors[target_rows]
    sims = T @ T.T
    lemmas = [wv.vocab[i]["lemma"] for i in target_rows]
    for t, res in enumerate(results):
        order = np.argsort(-sims[t])
        nb = next(j for j in order if lemmas[j] != lemmas[t])
        d_l2 = 1.0 - float(sims[t, nb])
        res["nearest_l2"] = lemmas[nb]
        res["collapse"] = 1.0 - cos_dist(recs[t], recs[nb]) / d_l2 if d_l2 > 0 else 0.0
    return results


def summarize(results: list[dict]) -> dict:
    out = {}
    for cls in ("content", "function", "all"):
        rows = [r for r in results if cls == "all" or r["class"] == cls]
        w = np.array([r["count"] for r in rows], dtype=float)
        w /= w.sum()
        out[cls] = {"n": len(rows)} | {m: float(np.dot(w, [r[m] for r in rows]))
                                         for m in ("residual", "self", "excess", "eff_n", "collapse")}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m wikiembed.reconstruct")
    ap.add_argument("--vectors", type=Path, default=Path("data/embed_all"))
    ap.add_argument("--langs", nargs="+", default=["en", "es", "ja"])
    ap.add_argument("--basis-size", type=int, default=15000)
    ap.add_argument("--target-size", type=int, default=5000)
    ap.add_argument("--candidates", type=int, default=100)
    ap.add_argument("--alpha", type=float, default=2e-4)
    args = ap.parse_args()
    cfg = {"basis_size": args.basis_size, "target_size": args.target_size,
           "candidates": args.candidates, "alpha": args.alpha}

    wv = WordVectors.load(args.vectors)
    out_dir = args.vectors / "reconstruction"
    summary = {"config": cfg, "directions": {}}
    for l1 in args.langs:
        for l2 in args.langs:
            if l1 == l2:
                continue
            results = run_direction(wv, l1, l2, cfg)
            write_jsonl(out_dir / f"{l1}-{l2}.jsonl", results)
            summary["directions"][f"{l1}->{l2}"] = summarize(results)
            s = summary["directions"][f"{l1}->{l2}"]["all"]
            print(f"  {l1}->{l2}: residual {s['residual']:.3f}  excess {s['excess']:.3f}  "
                  f"eff_n {s['eff_n']:.2f}  collapse {s['collapse']:.3f}")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), "utf-8")
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
