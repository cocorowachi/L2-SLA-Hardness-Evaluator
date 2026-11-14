"""python -m wikicollect {seed,resolve,fetch,filter,segment,all} [--config config.toml]"""
import argparse
import json
import os
import random
import tomllib
from collections import Counter
from pathlib import Path

from wikicollect.api import WikiAPI
from wikicollect.clean import build_article
from wikicollect.fetch import fetch
from wikicollect.filters import cheap_reasons, full_reasons, origin_lang
from wikicollect.resolve import resolve
from wikicollect.seed import seed
from wikicollect.segment import article_sentences
from wikicollect.store import read_jsonl, update_manifest, write_jsonl

STAGES = ["seed", "resolve", "fetch", "filter", "segment"]
NO_CONTACT_REQUEST_CAP = 300


def make_api(cfg: dict, out_dir: Path) -> WikiAPI:
    # WIKICOLLECT_CONTACT keeps a personal address out of the shared config file
    contact = (os.environ.get("WIKICOLLECT_CONTACT") or cfg["api"]["contact"]).strip()
    api = WikiAPI(contact, out_dir / "http_cache", cfg["api"]["min_interval"], cfg["api"]["maxlag"])
    if not contact:
        # Without contact info Wikimedia treats us as unidentified: 10 requests/minute.
        print(f"! no contact set: limited to {NO_CONTACT_REQUEST_CAP} uncached requests at 8/min. Set the "
              "WIKICOLLECT_CONTACT environment variable (or api.contact in config.toml) for a full run.")
        api.max_requests = NO_CONTACT_REQUEST_CAP
        api.min_interval = max(api.min_interval, 7.5)
    return api


def pilot_sample(seeds: list[dict], n: int, seed_value: int) -> list[dict]:
    rng = random.Random(seed_value)
    by_topic: dict[str, list[dict]] = {}
    for s in seeds:
        by_topic.setdefault(s["topic"], []).append(s)
    return [s for group in by_topic.values() for s in rng.sample(group, min(n, len(group)))]


def run_filter(cfg: dict, out_dir: Path) -> Counter:
    """Re-apply the filters to everything fetched, using the current config, and write articles.jsonl.
    (Loosening a threshold cannot recover triples that fetch skipped; re-run fetch for that.)"""
    langs, stats, records = cfg["run"]["langs"], Counter(), []
    for rec in read_jsonl(out_dir / "fetched.jsonl"):
        raws = {l: json.loads((out_dir / p).read_text("utf-8")) for l, p in rec["raw"].items()}
        articles = {l: build_article(r, cfg["back_matter"][l]) for l, r in raws.items()}
        if len(articles) == len(langs):
            reasons = cheap_reasons(raws, cfg) + full_reasons(articles, cfg)
        else:
            reasons = rec["reasons"]  # dropped before all pages were fetched
        origin = origin_lang(articles) if len(articles) == len(langs) else None
        for a in articles.values():
            records.append(a | {"topic": rec["topic"], "subtopic": rec["subtopic"],
                                "origin_lang": origin, "keep": not reasons, "drop_reasons": reasons})
        stats["kept" if not reasons else "dropped"] += 1
        stats.update(r.split(":")[-1] for r in reasons)
    write_jsonl(out_dir / "articles.jsonl", records)
    return stats


def run_segment(out_dir: Path) -> int:
    kept = (a for a in read_jsonl(out_dir / "articles.jsonl") if a["keep"])
    return write_jsonl(out_dir / "sentences.jsonl", (s for a in kept for s in article_sentences(a)))


def main() -> None:
    ap = argparse.ArgumentParser(prog="wikicollect")
    ap.add_argument("stage", choices=STAGES + ["all"])
    ap.add_argument("--config", type=Path, default=Path("config.toml"))
    ap.add_argument("--n-triples", type=int, help="override run.n_triples")
    ap.add_argument("--out-dir", type=Path, help="override run.out_dir")
    ap.add_argument("--max-seeds-per-topic", type=int,
                    help="pilot runs: resolve only a random N seeds per topic")
    args = ap.parse_args()

    cfg = tomllib.loads(args.config.read_text("utf-8"))
    if args.n_triples:
        cfg["run"]["n_triples"] = args.n_triples
    if args.out_dir:
        cfg["run"]["out_dir"] = str(args.out_dir)
    out_dir = Path(cfg["run"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    stages = STAGES if args.stage == "all" else [args.stage]
    api = make_api(cfg, out_dir) if {"seed", "resolve", "fetch"} & set(stages) else None

    for stage in stages:
        print(f"[{stage}]")
        if stage == "seed":
            n = write_jsonl(out_dir / "seeds.jsonl", seed(api, cfg["seed"]["pages"]))
            update_manifest(out_dir, stage, cfg, seeds=n)
        elif stage == "resolve":
            seeds = list(read_jsonl(out_dir / "seeds.jsonl"))
            if args.max_seeds_per_topic:
                seeds = pilot_sample(seeds, args.max_seeds_per_topic, cfg["run"]["seed"])
            triples = resolve(api, seeds, cfg["run"]["langs"])
            n = write_jsonl(out_dir / "triples.jsonl", triples)
            print(f"  {n} items have articles in all of {cfg['run']['langs']}")
            update_manifest(out_dir, stage, cfg, triples=n)
        elif stage == "fetch":
            kept = fetch(api, list(read_jsonl(out_dir / "triples.jsonl")), cfg, out_dir)
            update_manifest(out_dir, stage, cfg, kept_per_topic=kept, requests=api.n_requests)
        elif stage == "filter":
            stats = run_filter(cfg, out_dir)
            print(f"  {dict(stats)}")
            update_manifest(out_dir, stage, cfg, **stats)
        elif stage == "segment":
            n = run_segment(out_dir)
            print(f"  {n} sentences")
            update_manifest(out_dir, stage, cfg, sentences=n)
