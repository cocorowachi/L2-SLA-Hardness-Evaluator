"""Build outputs/colab_bundle.zip for colab_embed.ipynb: the code, config, and the two wiki
data files the embedding needs (not the HTTP cache or raw pages).

Usage: python scripts/make_colab_bundle.py
Then upload outputs/colab_bundle.zip to Google Drive as MyDrive/nlp/colab_bundle.zip.
"""
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FILES = ["config.toml", "requirements.txt", "data/wiki/articles.jsonl", "data/wiki/sentences.jsonl"]
PACKAGES = ["wikicollect", "wikiembed"]


def main() -> None:
    out = ROOT / "outputs" / "colab_bundle.zip"
    out.parent.mkdir(exist_ok=True)
    paths = [ROOT / f for f in FILES] + [p for pkg in PACKAGES for p in sorted((ROOT / pkg).rglob("*.py"))]
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise SystemExit(f"missing: {missing} (run wikicollect first)")
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in paths:
            z.write(p, p.relative_to(ROOT).as_posix())
            print(f"  + {p.relative_to(ROOT).as_posix()}")
    print(f"wrote {out} ({out.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
