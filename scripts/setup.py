"""One-command setup for the l2diff prototype.

    python scripts/setup.py            # install deps, fetch artifacts, download models, smoke test
    python scripts/setup.py --no-install   # skip pip (dependencies already installed)

Steps:
  1. check Python >= 3.11
  2. pip install -r requirements.txt
  3. download the prebuilt word vectors + scores from the GitHub Release (scripts/fetch_artifacts.py)
  4. download the Stanza models for en/es/ja and the XLM-R model, so the first query is not slow
  5. run a quick smoke test of the three commands
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run(cmd: list[str]) -> None:
    print(f"\n$ {' '.join(cmd)}")
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-install", action="store_true", help="skip pip install")
    ap.add_argument("--no-smoke", action="store_true", help="skip the smoke test")
    args = ap.parse_args()

    if sys.version_info < (3, 11):
        sys.exit(f"Python 3.11+ required (found {sys.version.split()[0]})")
    if not args.no_install:
        run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    run([sys.executable, "scripts/fetch_artifacts.py"])

    print("\n$ downloading Stanza (en, es, ja) and XLM-R models")
    import stanza
    from transformers import AutoModel, AutoTokenizer
    for lang in ("en", "es", "ja"):
        stanza.download(lang, processors="tokenize,mwt,pos,lemma" if lang == "es" else "tokenize,pos,lemma",
                        verbose=False)
    AutoTokenizer.from_pretrained("FacebookAI/xlm-roberta-base")
    AutoModel.from_pretrained("FacebookAI/xlm-roberta-base")

    if not args.no_smoke:
        run([sys.executable, "scripts/smoke_test.py"])
    print("\nSetup complete. Try:  python -m l2diff pair --l1 en --l2 ja")


if __name__ == "__main__":
    main()
