"""Initial evaluation of the l2diff prototype.

Task A: vector quality, translation retrieval on the MUSE test dictionaries (5000-6500 split).
        Metric: precision@1/5/10. System = mean-centered vectors; baseline = raw (uncentered) vectors.
Task B: difficulty detection on a hand-built, pre-registered case list (eval/cases.jsonl).
        Score = max(gap percentile, split percentile), fixed before evaluation.
        Metrics: AUC (P[hard item outranks easy item]) with a 95% bootstrap interval, and accuracy at
        the 50th-percentile threshold. Baseline = symmetric score: percentile of 1 - (cosine to the
        nearest L1 word), which cannot tell directions apart.
Task C: directional pairs (eval/directional_pairs.jsonl): does the splitting direction score higher
        than the merging direction?
Headline numbers use split == "test" items only; "dev" items were inspected or used for tuning.

Usage: python eval/run_eval.py [--artifacts data/embed_all] [--muse data/raw/muse]
Writes eval/results/results.json and eval/results/report.md.
"""
import argparse
import json
import sys
from bisect import bisect_left
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from l2diff.model import DifficultyModel  # noqa: E402
from wikiembed.vectors import WordVectors  # noqa: E402

OUT = ROOT / "eval" / "results"
RNG = np.random.default_rng(0)


def load_jsonl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


# ------------------------------------------------------------------ Task A
def norm(w: str, lang: str) -> str:
    return w if lang == "ja" else w.lower()


def bli(wv: WordVectors, path: Path, src: str, tgt: str) -> dict:
    gold: dict[str, set] = {}
    for line in open(path, encoding="utf-8"):
        parts = line.split()
        if len(parts) == 2:
            gold.setdefault(norm(parts[0], src), set()).add(norm(parts[1], tgt))
    best: dict[tuple[str, str], int] = {}  # (lang, normalized lemma) -> most frequent row
    for r in wv.vocab:
        k = (r["lang"], norm(r["lemma"], r["lang"]))
        if k not in best or r["count"] > wv.vocab[best[k]]["count"]:
            best[k] = r["idx"]
    tgt_rows = np.array([i for (l, _), i in best.items() if l == tgt])
    tgt_lemmas = [norm(wv.vocab[i]["lemma"], tgt) for i in tgt_rows]
    T = wv.vectors[tgt_rows]
    hits = {1: 0, 5: 0, 10: 0}
    n = 0
    examples = []
    for s, golds in gold.items():
        if (src, s) not in best or not any((tgt, g) in best for g in golds):
            continue
        n += 1
        sims = T @ wv.vectors[best[(src, s)]]
        ranked, seen = [], set()
        for j in np.argsort(-sims)[:40]:
            if tgt_lemmas[j] not in seen:
                seen.add(tgt_lemmas[j])
                ranked.append(tgt_lemmas[j])
        for k in hits:
            hits[k] += any(g in ranked[:k] for g in golds)
        if len(examples) < 400:
            examples.append({"src": s, "gold": sorted(golds), "top5": ranked[:5], "hit1": ranked[0] in golds})
    return {"pair": f"{src}-{tgt}", "evaluated": n, "dictionary_entries": len(gold),
            **{f"p@{k}": round(v / n, 3) for k, v in hits.items()}, "examples": examples}


# ------------------------------------------------------------------ Task B / C
class Scorer:
    def __init__(self, model: DifficultyModel):
        self.m = model
        self._sym_ref = {}

    def symmetric(self, l1: str, l2: str, v: np.ndarray) -> float:
        """Baseline: percentile of (1 - max cosine to any L1 basis word); no notion of direction."""
        d = self.m.direction(l1, l2)
        B = self.m.wv.vectors[d.basis]
        if (l1, l2) not in self._sym_ref:
            T = self.m.wv.vectors[d.target_rows]
            self._sym_ref[(l1, l2)] = sorted((1.0 - (T @ B.T).max(axis=1)).tolist())
        ref = self._sym_ref[(l1, l2)]
        return round(100.0 * bisect_left(ref, 1.0 - float((B @ v).max())) / len(ref), 1)

    def case(self, c: dict) -> dict:
        rows = self.m.wv.find(c["lemma"], c["l2"], c["upos"])
        if not rows:
            return c | {"missing": True}
        s = self.m.score_vector(self.m.wv.vectors[rows[0]], c["l1"], c["l2"], c["lemma"], c["upos"], "vocab")
        return c | {"gap_pct": s.gap_pct, "split_pct": s.split_pct,
                    "difficulty": max(s.gap_pct, s.split_pct),
                    "baseline": self.symmetric(c["l1"], c["l2"], self.m.wv.vectors[rows[0]]),
                    "nearest_l2": s.nearest_l2, "atoms": [a[0] for a in s.atoms[:3]],
                    "count": self.m.wv.vocab[rows[0]]["count"]}


def auc(scores_hard, scores_easy) -> float:
    if not scores_hard or not scores_easy:
        return float("nan")
    wins = sum((h > e) + 0.5 * (h == e) for h in scores_hard for e in scores_easy)
    return wins / (len(scores_hard) * len(scores_easy))


def auc_ci(rows, key, n=2000):
    hard = [r[key] for r in rows if r["label"] == "hard"]
    easy = [r[key] for r in rows if r["label"] == "easy"]
    point = auc(hard, easy)
    boots = [auc(list(RNG.choice(hard, len(hard))), list(RNG.choice(easy, len(easy)))) for _ in range(n)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return round(point, 3), round(float(lo), 3), round(float(hi), 3)


def accuracy(rows, key, threshold=50.0):
    ok = sum((r[key] >= threshold) == (r["label"] == "hard") for r in rows)
    return round(ok / len(rows), 3)


# ------------------------------------------------------------------ report
def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(x) for x in r) + " |" for r in rows]
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default=str(ROOT / "data" / "embed_all"))
    ap.add_argument("--muse", default=str(ROOT / "data" / "raw" / "muse"))
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    model = DifficultyModel(args.artifacts)
    scorer = Scorer(model)
    results = {"artifacts": Path(args.artifacts).resolve().relative_to(ROOT).as_posix()}

    # Task A
    muse = Path(args.muse)
    if muse.exists():
        raw = WordVectors.load(args.artifacts, centered=False) if "raw" in np.load(Path(args.artifacts) / "vectors.npz").files else None
        task_a = []
        for src, tgt in (("en", "es"), ("es", "en"), ("en", "ja"), ("ja", "en")):
            path = muse / f"{src}-{tgt}.5000-6500.txt"
            if not path.exists():
                continue
            sys_ = bli(model.wv, path, src, tgt)
            base = bli(raw, path, src, tgt) if raw is not None else None
            task_a.append({"system": sys_, "baseline_raw": base})
            print(f"  A {src}-{tgt}: n={sys_['evaluated']} P@1 {sys_['p@1']} (raw {base and base['p@1']})")
        results["task_a"] = task_a

    # Task B
    cases = [scorer.case(c) for c in load_jsonl(ROOT / "eval" / "cases.jsonl")]
    missing = [c["id"] for c in cases if c.get("missing")]
    scored = [c for c in cases if not c.get("missing")]
    test = [c for c in scored if c["split"] == "test"]
    dev = [c for c in scored if c["split"] == "dev"]
    task_b = {"missing": missing, "cases": scored}
    for name, rows in (("test", test), ("dev", dev), ("all", scored)):
        task_b[name] = {"n": len(rows), "n_hard": sum(r["label"] == "hard" for r in rows),
                        **{f"auc_{k}": auc_ci(rows, k) for k in ("difficulty", "gap_pct", "split_pct", "baseline")},
                        **{f"acc_{k}": accuracy(rows, k) for k in ("difficulty", "baseline")}}
    results["task_b"] = task_b
    t = task_b["test"]
    print(f"  B test: AUC difficulty {t['auc_difficulty']}  baseline {t['auc_baseline']}  "
          f"acc {t['acc_difficulty']} vs {t['acc_baseline']}")

    # Task C
    pairs = []
    for p in load_jsonl(ROOT / "eval" / "directional_pairs.jsonl"):
        h = scorer.case(p["hard"] | {"id": p["id"] + ":hard", "label": "hard", "split": p.get("split", "test")})
        e = scorer.case(p["easy"] | {"id": p["id"] + ":easy", "label": "easy", "split": p.get("split", "test")})
        if h.get("missing") or e.get("missing"):
            pairs.append({"id": p["id"], "missing": True})
            continue
        pairs.append({"id": p["id"], "split": p.get("split", "test"), "rationale": p["rationale"],
                      "hard": f"{p['hard']['l1']}->{p['hard']['l2']} {p['hard']['lemma']}",
                      "easy": f"{p['easy']['l1']}->{p['easy']['l2']} {p['easy']['lemma']}",
                      "hard_difficulty": h["difficulty"], "easy_difficulty": e["difficulty"],
                      "correct": h["difficulty"] > e["difficulty"],
                      "hard_baseline": h["baseline"], "easy_baseline": e["baseline"],
                      "baseline_correct": h["baseline"] > e["baseline"]})
    tp = [p for p in pairs if not p.get("missing") and p["split"] == "test"]
    results["task_c"] = {"pairs": pairs, "test_accuracy": round(sum(p["correct"] for p in tp) / len(tp), 3),
                         "test_baseline_accuracy": round(sum(p["baseline_correct"] for p in tp) / len(tp), 3),
                         "n_test": len(tp)}
    print(f"  C test: {sum(p['correct'] for p in tp)}/{len(tp)} directional pairs correct "
          f"(baseline {sum(p['baseline_correct'] for p in tp)}/{len(tp)})")

    # pair-level consistency with FSI (en speakers: Spanish = category I, Japanese = category IV)
    results["fsi_check"] = {d: model.summary[d]["all"]["excess"] for d in ("en->es", "en->ja")}

    (OUT / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), "utf-8")
    write_report(results)
    print(f"wrote {OUT / 'results.json'} and {OUT / 'report.md'}")


def write_report(r: dict) -> None:
    L = ["# l2diff initial evaluation\n",
         f"Artifacts: `{r['artifacts']}`. Task A is reported on the full vectors (`data/embed_all`); the compact "
         "release (`artifacts/release`) reproduces Tasks B and C exactly but has a smaller vocabulary, so Task A "
         "coverage and P@k differ on it and its raw-vector baseline is not included.\n"]
    if "task_a" in r:
        L.append("## Task A: translation retrieval (MUSE test dictionaries)\n")
        rows = []
        for a in r["task_a"]:
            s, b = a["system"], a["baseline_raw"] or {}
            rows.append([s["pair"], s["evaluated"], s["p@1"], s["p@5"], s["p@10"],
                         b.get("p@1", "-"), b.get("p@10", "-")])
        L.append(md_table(["pair", "n", "P@1", "P@5", "P@10", "raw P@1", "raw P@10"], rows) + "\n")
    b = r["task_b"]
    L.append("## Task B: difficulty detection (hard vs easy cases)\n")
    rows = []
    for name in ("test", "dev", "all"):
        x = b[name]
        f = lambda t: f"{t[0]} [{t[1]}, {t[2]}]"  # noqa: E731
        rows.append([name, x["n"], x["n_hard"], f(x["auc_difficulty"]), f(x["auc_gap_pct"]), f(x["auc_split_pct"]),
                     f(x["auc_baseline"]), x["acc_difficulty"], x["acc_baseline"]])
    L.append(md_table(["set", "n", "hard", "AUC difficulty", "AUC gap only", "AUC split only",
                       "AUC baseline", "acc", "baseline acc"], rows) + "\n")
    L.append(f"Missing from vocabulary: {', '.join(b['missing']) or 'none'}\n")
    L.append("### Per case\n")
    rows = [[c["id"], c["split"], f"{c['l1']}->{c['l2']}", c["lemma"], c["label"], c["category"],
             c["gap_pct"], c["split_pct"], c["difficulty"], c["baseline"], c["nearest_l2"], ", ".join(c["atoms"])]
            for c in sorted(b["cases"], key=lambda c: (c["split"] != "test", c["label"], -c["difficulty"]))]
    L.append(md_table(["id", "split", "dir", "lemma", "label", "type", "gap", "split", "difficulty", "baseline",
                       "nearest L2 word", "rebuilt from"], rows) + "\n")
    c = r["task_c"]
    L.append("## Task C: directional pairs (split direction should score higher)\n")
    rows = [[p["id"], p.get("split", "-"), p.get("hard", "-"), p.get("hard_difficulty", "-"), p.get("easy", "-"),
             p.get("easy_difficulty", "-"), p.get("correct", "missing"), p.get("baseline_correct", "-")]
            for p in c["pairs"]]
    L.append(md_table(["pair", "split", "hard side", "difficulty", "easy side", "difficulty", "correct",
                       "baseline correct"], rows) + "\n")
    L.append(f"Test accuracy: {c['test_accuracy']} (baseline {c['test_baseline_accuracy']}), n = {c['n_test']}\n")
    L.append(f"## FSI consistency\n\nOverall gap for English speakers: Spanish {r['fsi_check']['en->es']:+.3f}, "
             f"Japanese {r['fsi_check']['en->ja']:+.3f} (FSI: Spanish category I, Japanese category IV).\n")
    (OUT / "report.md").write_text("\n".join(L), "utf-8")


if __name__ == "__main__":
    main()
