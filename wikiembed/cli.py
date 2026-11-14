"""python -m wikiembed {parse,embed,neighbors} [--config config.toml]

  parse      Stanza words/lemmas/UPOS for the [embed] topic subset -> data/embed/parsed/{lang}.jsonl
  embed      XLM-R type-level vectors -> data/embed/vectors.npz + vocab.jsonl
  neighbors  nearest words across languages, e.g. neighbors 光合成 --lang ja --target en
"""
import argparse
import tomllib
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(prog="wikiembed")
    ap.add_argument("stage", choices=["parse", "embed", "neighbors"])
    ap.add_argument("word", nargs="?", help="neighbors: the lemma to look up")
    ap.add_argument("--config", type=Path, default=Path("config.toml"))
    ap.add_argument("--langs", nargs="+", help="override run.langs")
    ap.add_argument("--limit", type=int, help="parse: only the first N selected sentences per language (pilot)")
    ap.add_argument("--out-dir", type=Path, help="override embed.out_dir")
    ap.add_argument("--wiki-dir", type=Path, help="override run.out_dir (where wikicollect wrote its data)")
    ap.add_argument("--all-topics", action="store_true", help="ignore embed.topics: use every topic")
    ap.add_argument("--max-per-article", type=int, help="override embed.max_sentences_per_article (0 = no cap)")
    ap.add_argument("--batch-size", type=int, help="override embed.batch_size (larger on a GPU)")
    ap.add_argument("--lang", help="neighbors: language of the word")
    ap.add_argument("--target", help="neighbors: restrict results to this language")
    ap.add_argument("--raw", action="store_true", help="neighbors: use uncentered vectors")
    ap.add_argument("-k", type=int, default=10)
    args = ap.parse_args()

    cfg = tomllib.loads(args.config.read_text("utf-8"))
    if args.all_topics:
        cfg["embed"]["topics"] = []
    if args.max_per_article is not None:
        cfg["embed"]["max_sentences_per_article"] = args.max_per_article
    if args.batch_size:
        cfg["embed"]["batch_size"] = args.batch_size
    langs = args.langs or cfg["run"]["langs"]
    out_dir = args.out_dir or Path(cfg["embed"]["out_dir"])
    wiki_dir = args.wiki_dir or Path(cfg["run"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.stage == "parse":
        from wikiembed.parse import parse
        for lang in langs:
            parse(wiki_dir, out_dir, cfg, lang, args.limit)
    elif args.stage == "embed":
        from wikiembed.embed import embed
        n = embed(out_dir, cfg, langs)
        print(f"wrote {n} word vectors to {out_dir / 'vectors.npz'} and {out_dir / 'vocab.jsonl'}")
    else:
        from wikiembed.vectors import WordVectors
        if not args.word or not args.lang:
            raise SystemExit("usage: python -m wikiembed neighbors WORD --lang LANG [--target LANG]")
        wv = WordVectors.load(out_dir, centered=not args.raw)
        for lemma, upos, lang, sim in wv.neighbors(args.word, args.lang, args.target, args.k):
            print(f"  {sim:.3f}  {lang}  {lemma}  ({upos})")
