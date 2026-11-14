"""Fetch stage: raw page records for triples, until each topic has its quota of kept triples.

Per chunk of triples:
  1. batched, per language: existence, disambiguation, talk-page translation template
  2. per page, only for triples still in play (en last, as it rarely fails):
     full plain-text extract + current revision, then the first revision
     (creation time, Content Translation tag)
Every triple considered is appended to fetched.jsonl with kept/dropped and reasons, so a
re-run resumes where it stopped (and the API cache makes repeated requests free).
"""
import json
import random
from collections import defaultdict
from pathlib import Path

from wikicollect.api import WikiAPI, chunks, query_pages
from wikicollect.clean import build_article
from wikicollect.filters import cheap_reasons, full_reasons
from wikicollect.store import append_jsonl, read_jsonl

CHUNK = 20


def fetch_meta(api: WikiAPI, lang: str, titles: list[str], template: str) -> dict[str, dict | None]:
    pages = query_pages(api, lang, titles, prop="pageprops", ppprop="disambiguation", redirects=1)
    talk_of = {t: f"Talk:{p['title']}" for t, p in pages.items() if p}
    talk = query_pages(api, lang, list(talk_of.values()), prop="templates",
                       tltemplates=template, tllimit="max") if talk_of else {}
    meta = {}
    for t, p in pages.items():
        if p is None:
            meta[t] = None
            continue
        talk_page = talk.get(talk_of[t])
        meta[t] = {"title": p["title"], "pageid": p["pageid"],
                   "is_disambiguation": "disambiguation" in p.get("pageprops", {}),
                   "translated_template": bool(talk_page and talk_page.get("templates"))}
    return meta


def fetch_page(api: WikiAPI, qid: str, lang: str, meta: dict) -> dict:
    title = meta["title"]
    p = api.get(lang, action="query", titles=title, prop="extracts|revisions", explaintext=1,
                exsectionformat="wiki", rvprop="ids|timestamp")["query"]["pages"][0]
    first = api.get(lang, action="query", titles=title, prop="revisions", rvdir="newer",
                    rvlimit=1, rvprop="ids|timestamp|tags")["query"]["pages"][0]["revisions"][0]
    rev = p["revisions"][0]
    return {
        "qid": qid, "lang": lang, "title": title, "pageid": meta["pageid"],
        "revid": rev["revid"], "rev_timestamp": rev["timestamp"], "extract": p.get("extract", ""),
        "is_disambiguation": meta["is_disambiguation"],
        "translated_template": meta["translated_template"],
        "cx_created": any(tag.startswith("contenttranslation") for tag in first.get("tags", [])),
        "created_at": first["timestamp"], "first_revid": first["revid"],
    }


def save_raw(out_dir: Path, raw: dict) -> str:
    rel = Path("raw") / raw["lang"] / f"{raw['pageid']}-{raw['revid']}.json"
    (out_dir / rel).parent.mkdir(parents=True, exist_ok=True)
    (out_dir / rel).write_text(json.dumps(raw, ensure_ascii=False), "utf-8")
    return rel.as_posix()


def quotas(topics: list[str], total: int) -> dict[str, int]:
    base, extra = divmod(total, len(topics))
    return {t: base + (i < extra) for i, t in enumerate(topics)}


def process_triple(api, triple, meta, cfg, out_dir) -> dict:
    langs = sorted(cfg["run"]["langs"], key=lambda l: l == "en")
    rec = {k: triple[k] for k in ("qid", "topic", "subtopic", "titles")} | {"raw": {}}
    reasons = cheap_reasons(meta, cfg)
    articles = {}
    for lang in langs:
        if reasons:
            break
        raw = fetch_page(api, triple["qid"], lang, meta[lang])
        rec["raw"][lang] = save_raw(out_dir, raw)
        articles[lang] = build_article(raw, cfg["back_matter"][lang])
        reasons = full_reasons({lang: articles[lang]}, cfg)
    if not reasons:
        reasons = full_reasons(articles, cfg)
    return rec | {"status": "dropped" if reasons else "kept", "reasons": reasons}


def fetch(api: WikiAPI, triples: list[dict], cfg: dict, out_dir: Path) -> dict[str, int]:
    langs = cfg["run"]["langs"]
    log = out_dir / "fetched.jsonl"
    done = {r["qid"]: r for r in read_jsonl(log)}
    by_topic: dict[str, list[dict]] = defaultdict(list)
    for t in triples:
        by_topic[t["topic"]].append(t)
    quota = quotas(list(by_topic), cfg["run"]["n_triples"])
    kept_per_topic = {}

    for topic, pool in by_topic.items():
        pool = sorted(pool, key=lambda t: t["qid"])
        random.Random(f"{cfg['run']['seed']}:{topic}").shuffle(pool)
        kept = sum(1 for r in done.values() if r["topic"] == topic and r["status"] == "kept")
        todo = [t for t in pool if t["qid"] not in done]
        for chunk in chunks(todo, CHUNK):
            if kept >= quota[topic]:
                break
            meta = {lang: fetch_meta(api, lang, [t["titles"][lang] for t in chunk],
                                     cfg["translation_templates"][lang]) for lang in langs}
            for t in chunk:
                if kept >= quota[topic]:
                    break
                rec = process_triple(api, t, {l: meta[l][t["titles"][l]] for l in langs}, cfg, out_dir)
                append_jsonl(log, rec)
                done[t["qid"]] = rec
                kept += rec["status"] == "kept"
            print(f"  {topic:<32} kept {kept}/{quota[topic]}  (requests so far: {api.n_requests})", end="\r")
        print()
        if kept < quota[topic]:
            print(f"  ! {topic}: only {kept} triples passed the filters (quota {quota[topic]})")
        kept_per_topic[topic] = kept
    return kept_per_topic
