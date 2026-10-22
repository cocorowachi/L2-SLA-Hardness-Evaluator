"""Download a sentence-aligned OPUS corpus (default: Tatoeba) for a language pair.

Usage: python -m splitprobe.download en es [--corpus Tatoeba] [--out data/raw]
Writes data/raw/<corpus>-<src>-<tgt>/<Corpus>.<src>-<tgt>.{src,tgt}
"""
import argparse
import io
import json
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

OPUS_API = "https://opus.nlpl.eu/opusapi/"


def opus_url(src: str, tgt: str, corpus: str) -> str:
    query = urllib.parse.urlencode(
        {"source": src, "target": tgt, "corpus": corpus, "preprocessing": "moses", "version": "latest"}
    )
    with urllib.request.urlopen(f"{OPUS_API}?{query}") as resp:
        corpora = json.load(resp)["corpora"]
    if not corpora:
        raise SystemExit(f"OPUS has no {corpus} corpus for {src}-{tgt}")
    return corpora[0]["url"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("tgt")
    ap.add_argument("--corpus", default="Tatoeba")
    ap.add_argument("--out", default="data/raw")
    args = ap.parse_args()

    url = opus_url(args.src, args.tgt, args.corpus)
    out = Path(args.out) / f"{args.corpus.lower()}-{args.src}-{args.tgt}"
    out.mkdir(parents=True, exist_ok=True)
    print(f"downloading {url}")
    with urllib.request.urlopen(url) as resp:
        zipfile.ZipFile(io.BytesIO(resp.read())).extractall(out)
    print(f"extracted to {out}")


if __name__ == "__main__":
    main()
