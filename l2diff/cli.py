"""Command-line interface.

  python -m l2diff pair --l1 en --l2 ja
  python -m l2diff word --l1 en --l2 es estar [--upos AUX]
  python -m l2diff text --l1 en --l2 ja "光合成は光エネルギーを化学エネルギーに変換する。"
  python -m l2diff text --l1 ja --l2 en --file essay.txt
Add --json to any command for machine-readable output.
"""
import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

LANG_NAMES = {"en": "English", "es": "Spanish", "ja": "Japanese"}
DEFAULT_ARTIFACTS = ["artifacts/release", "data/embed_all"]


def find_artifacts(arg: str | None) -> Path:
    for cand in [arg, os.environ.get("L2DIFF_ARTIFACTS"), *DEFAULT_ARTIFACTS]:
        if cand and (Path(cand) / "vectors.npz").exists():
            return Path(cand)
    raise SystemExit("No artifacts found. Run `python scripts/fetch_artifacts.py` (or pass --artifacts DIR).")


def level(pct: float | None) -> str:
    if pct is None:
        return "-"
    return "very high" if pct >= 90 else "high" if pct >= 75 else "medium" if pct >= 40 else "low"


def atoms_str(atoms, n=3) -> str:
    return " + ".join(f"{a[0]}" for a in atoms[:n]) or "(nothing)"


def header(l1, l2):
    return f"{LANG_NAMES[l1]} speaker learning {LANG_NAMES[l2]} ({l1}->{l2})"


LEGEND = ("  gap   = how much of the word no combination of L1 words covers (percentile vs the L2's\n"
          "          5,000 most frequent words; 90+ = a concept the L1 lacks)\n"
          "  split = how much the L1 view blurs this word with its nearest L2 neighbour (90+ = the\n"
          "          learner must learn a distinction the L1 does not make)")


def print_word(scores, l1, l2):
    print(header(l1, l2))
    for s in scores:
        print(f"\n  {s.lemma} ({s.upos})")
        print(f"    gap   {s.gap_pct:5.1f} pct  [{level(s.gap_pct)}]   excess {s.excess:+.3f}")
        if s.split_pct is not None:
            print(f"    split {s.split_pct:5.1f} pct  [{level(s.split_pct)}]   nearest {LANG_NAMES[l2]} word: {s.nearest_l2}")
        print(f"    rebuilt from {LANG_NAMES[l1]}: " +
              ", ".join(f"{a[0]} ({a[2]:.2f})" for a in s.atoms[:5]) + f"   [{s.eff_n:.1f} effective words]")
    print("\n" + LEGEND)


def print_text(res):
    l1, l2 = res["l1"], res["l2"]
    print(header(l1, l2))
    print(f"  {res['sentence']}\n")
    print(f"  {'word':<14}{'lemma':<14}{'POS':<7}{'gap':>6}{'split':>7}   rebuilt from {LANG_NAMES[l1]}")
    for s in res["words"]:
        split = f"{s.split_pct:5.0f}" if s.split_pct is not None else "    -"
        mark = "*" if s.source == "context" else " "
        print(f"  {s.text:<14}{s.lemma:<14}{s.upos:<7}{s.gap_pct:5.0f}{mark}{split}    {atoms_str(s.atoms)}")
    if res["sentence_gap_pct"] is not None:
        print(f"\n  sentence gap (mean word percentile): {res['sentence_gap_pct']} [{level(res['sentence_gap_pct'])}]")
        print("  hardest words: " + ", ".join(f"{s.text} ({s.gap_pct:.0f})" for s in res["hardest"]))
    print("  * = word not in the vocabulary; scored from a fresh contextual vector (less reliable)\n")
    print(LEGEND)


def print_pair(res):
    for key, d in res.items():
        a, b = key.split("->")
        s = d["summary"]
        print(f"\n{header(a, b)}")
        for cls in ("content", "function"):
            x = s[cls]
            print(f"  {cls:<9} words: gap (excess) {x['excess']:+.3f}   split (collapse) {x['collapse']:.3f}   "
                  f"L1 words per L2 word {x['eff_n']:.1f}")
        print("  biggest gaps:   " + ", ".join(f"{r['lemma']}/{r['upos']}" for r in d["top_gaps"]))
        print("  biggest splits: " + ", ".join(f"{r['lemma']}~{r['nearest_l2']}" for r in d["top_splits"]))
        print("  easiest:        " + ", ".join(r["lemma"] for r in d["easiest"]))
    keys = list(res)
    g = [res[k]["summary"]["all"]["excess"] for k in keys]
    harder = keys[0] if g[0] > g[1] else keys[1]
    print(f"\n  Overall gap: {keys[0]} {g[0]:+.3f} vs {keys[1]} {g[1]:+.3f} -> {harder} is the harder direction.")


def to_json(obj):
    def conv(o):
        if hasattr(o, "__dataclass_fields__"):
            return asdict(o)
        raise TypeError(type(o))
    return json.dumps(obj, ensure_ascii=False, indent=2, default=conv)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="l2diff", description="Directional L2 difficulty (en, es, ja).")
    ap.add_argument("--artifacts", help="directory with vectors.npz, vocab.jsonl, reconstruction/")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, help_ in (("pair", "summary for a language pair, both directions"),
                        ("word", "score one L2 word"), ("text", "score every word of a new L2 sentence")):
        p = sub.add_parser(name, help=help_)
        p.add_argument("--l1", required=True, choices=["en", "es", "ja"], help="learner's native language")
        p.add_argument("--l2", required=True, choices=["en", "es", "ja"], help="language being learned")
        if name == "word":
            p.add_argument("lemma", help="dictionary form of the L2 word, e.g. estar, 先輩, the")
            p.add_argument("--upos", help="restrict to one part of speech (NOUN, VERB, AUX, ...)")
        if name == "text":
            p.add_argument("sentence", nargs="?", help="L2 text to score")
            p.add_argument("--file", type=Path, help="read the L2 text from a file instead")
    args = ap.parse_args(argv)
    if args.l1 == args.l2:
        ap.error("--l1 and --l2 must differ")
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")

    from l2diff.model import DifficultyModel
    model = DifficultyModel(find_artifacts(args.artifacts))

    if args.cmd == "pair":
        res = model.pair(args.l1, args.l2)
        print(to_json(res)) if args.json else print_pair(res)
    elif args.cmd == "word":
        scores = model.word(args.l1, args.l2, args.lemma, args.upos)
        if not scores:
            raise SystemExit(f"'{args.lemma}' is not in the {LANG_NAMES[args.l2]} vocabulary. "
                             f"Try `l2diff text` with a sentence containing it.")
        print(to_json(scores)) if args.json else print_word(scores, args.l1, args.l2)
    else:
        text = args.file.read_text("utf-8").strip() if args.file else args.sentence
        if not text:
            ap.error("give a sentence or --file")
        res = model.text(args.l1, args.l2, text)
        print(to_json(res)) if args.json else print_text(res)


if __name__ == "__main__":
    main()
