"""Download the prebuilt l2diff artifacts (word vectors + reconstruction scores) from the GitHub
Release named in artifacts.lock.json, verify the SHA-256, and unpack to artifacts/release/.

Usage: python scripts/fetch_artifacts.py [--force]
"""
import argparse
import hashlib
import json
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="download again even if artifacts/release exists")
    args = ap.parse_args()
    lock = json.loads((ROOT / "artifacts.lock.json").read_text("utf-8"))
    dest = ROOT / "artifacts" / "release"
    if (dest / "vectors.npz").exists() and not args.force:
        print(f"artifacts already present in {dest} (use --force to re-download)")
        return

    zip_path = ROOT / "artifacts" / lock["file"]
    zip_path.parent.mkdir(exist_ok=True)
    print(f"downloading {lock['url']} ({lock['bytes'] / 1e6:.0f} MB)")
    with urllib.request.urlopen(lock["url"]) as resp, open(zip_path, "wb") as f:
        done = 0
        while chunk := resp.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            print(f"  {done / 1e6:6.0f} / {lock['bytes'] / 1e6:.0f} MB", end="\r")
    print()

    sha = hashlib.sha256(zip_path.read_bytes()).hexdigest()
    if sha != lock["sha256"]:
        zip_path.unlink()
        sys.exit(f"checksum mismatch (got {sha}, expected {lock['sha256']}); download removed")
    if dest.exists():
        shutil.rmtree(dest)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(ROOT / "artifacts")
    print(f"verified and unpacked to {dest}")


if __name__ == "__main__":
    main()
