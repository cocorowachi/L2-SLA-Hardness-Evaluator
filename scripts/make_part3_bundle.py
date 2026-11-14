"""Build submission/part3_artifacts.zip: the collection and preparation code for the milestone.

The contact address in config.toml is blanked in the packaged copy.
Usage: python scripts/make_part3_bundle.py
"""
import re
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PACKAGES = ["wikicollect", "wikiembed"]
FILES = ["colab_embed.ipynb", "requirements.txt", "docs/wiki-collection-architecture.md",
         "scripts/make_colab_bundle.py", "scripts/make_submission_samples.py", "scripts/make_part3_bundle.py"]
TOP = "part3_artifacts"


def main() -> None:
    out = ROOT / "submission" / f"{TOP}.zip"
    paths = [p for pkg in PACKAGES for p in sorted((ROOT / pkg).rglob("*.py"))] + [ROOT / f for f in FILES]
    config = re.sub(r'^contact = ".*"', 'contact = ""   # set to your email or project URL before running',
                    (ROOT / "config.toml").read_text("utf-8"), flags=re.M)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(ROOT / "submission" / "PART3_README.md", f"{TOP}/README.md")
        z.writestr(f"{TOP}/config.toml", config)
        for p in paths:
            z.write(p, f"{TOP}/{p.relative_to(ROOT).as_posix()}")
    with zipfile.ZipFile(out) as z:
        names = z.namelist()
    for n in names:
        print(f"  {n}")
    print(f"wrote {out.relative_to(ROOT).as_posix()} ({out.stat().st_size / 1024:.0f} KB, {len(names)} files)")


if __name__ == "__main__":
    main()
