"""Seed stage: English Vital Articles (Level 4) pages -> seeds.jsonl (title_en, topic, subtopic)."""
import re

from wikicollect.api import WikiAPI

HEADING = re.compile(r"^(=+)\s*(.*?)\s*\1\s*$")
LINK = re.compile(r"\[\[([^\]|#]+)(?:#[^\]|]*)?(?:\|[^\]]*)?\]\]")
NAMESPACES = {"wikipedia", "wp", "category", "file", "image", "template", "help", "portal",
              "special", "talk", "user", "m", "meta", "wikt", "draft"}


def clean_heading(h: str) -> str:
    h = re.sub(r"\{\{[^}]*\}\}", "", h)
    h = re.sub(r"\[\[(?:[^\]|]*\|)?([^\]]*)\]\]", r"\1", h)
    h = re.sub(r"\(\s*\d+\s+articles?\s*\)", "", h)
    return h.strip(" '")


def parse_vital_page(wikitext: str, topic: str) -> list[dict]:
    """First article link on each numbered list line, tagged with the enclosing == heading ==."""
    subtopic, out = None, []
    for line in wikitext.splitlines():
        h = HEADING.match(line)
        if h:
            if len(h.group(1)) == 2:
                subtopic = clean_heading(h.group(2))
            continue
        if not line.startswith("#"):
            continue
        for m in LINK.finditer(line):
            target = m.group(1).strip()
            prefix = target.split(":", 1)[0].lower() if ":" in target else None
            if target.startswith(":") or prefix in NAMESPACES:
                continue
            out.append({"title_en": target.replace("_", " "), "topic": topic, "subtopic": subtopic})
            break
    return out


def seed(api: WikiAPI, pages: list[str]) -> list[dict]:
    seen, seeds = set(), []
    for page in pages:
        topic = page.rsplit("/", 1)[-1]
        data = api.get("en", action="query", titles=page, prop="revisions",
                       rvprop="content", rvslots="main", redirects=1)
        p = data["query"]["pages"][0]
        if p.get("missing"):
            raise SystemExit(f"seed page not found: {page}")
        rows = parse_vital_page(p["revisions"][0]["slots"]["main"]["content"], topic)
        new = [r for r in rows if r["title_en"] not in seen]
        seen.update(r["title_en"] for r in new)
        seeds.extend(new)
        print(f"  {topic:<32} {len(new):>5} articles")
    return seeds
