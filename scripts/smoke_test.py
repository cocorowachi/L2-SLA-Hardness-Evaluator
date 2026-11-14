"""Run each l2diff command once and check the output has the expected shape.

Usage: python scripts/smoke_test.py   (exits non-zero on failure)
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CHECKS = [
    (["pair", "--l1", "en", "--l2", "ja"], lambda o: "en->ja" in o and "summary" in o["en->ja"]),
    (["word", "--l1", "en", "--l2", "es", "estar"], lambda o: o and "gap_pct" in o[0]),
    (["text", "--l1", "ja", "--l2", "en", "The quokka lives on islands near Perth."],
     lambda o: o["words"] and any(w["source"] == "context" for w in o["words"])),
]


def main() -> None:
    failed = 0
    for args, ok in CHECKS:
        cmd = [sys.executable, "-m", "l2diff", "--json", *args]
        res = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
        try:
            passed = res.returncode == 0 and ok(json.loads(res.stdout))
        except (json.JSONDecodeError, KeyError, IndexError, TypeError):
            passed = False
        print(f"  {'PASS' if passed else 'FAIL'}  l2diff {' '.join(args[:5])}")
        if not passed:
            failed += 1
            print(res.stderr[-2000:])
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
