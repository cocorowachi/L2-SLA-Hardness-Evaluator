"""Write raw vs processed data samples for the data-collection milestone into submission/samples/.

Wikipedia example: one triple (default Q40164, Lever / Palanca / てこ), traced through every stage:
  raw/       the MediaWiki API responses exactly as cached during collection
  processed/ the article records, sentence records, Stanza words for the lead, word-vector
             entries and reconstruction scores
Tatoeba example: aligned raw lines and the ser/estar probe examples extracted from them.

Usage: python scripts/make_submission_samples.py [--qid Q40164]
"""
import argparse
import hashlib
import json
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from wikicollect.store import read_jsonl, write_jsonl  # noqa: E402

OUT = ROOT / "submission" / "samples"
WIKI = ROOT / "data" / "wiki"


def cached_response(site: str, maxlag: int, **params) -> dict:
    """Same key as wikicollect.api.WikiAPI.get, so this is the response saved during collection."""
    host = site if "." in site else f"{site}.wikipedia.org"
    params = {"format": "json", "formatversion": "2", "maxlag": maxlag, **params}
    key = hashlib.sha1(json.dumps([host, sorted(params.items())], ensure_ascii=False).encode()).hexdigest()
    path = WIKI / "http_cache" / key[:2] / f"{key}.json"
    return {"request": {"url": f"https://{host}/w/api.php", "params": params},
            "response": json.loads(path.read_text("utf-8"))}


def dump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), "utf-8")


def wiki_samples(qid: str, cfg: dict) -> None:
    rec = next(r for r in read_jsonl(WIKI / "fetched.jsonl") if r["qid"] == qid)
    maxlag = cfg["api"]["maxlag"]
    for lang, rel in rec["raw"].items():
        raw = json.loads((WIKI / rel).read_text("utf-8"))
        dump(OUT / "raw" / f"{qid}_{lang}_extract_api_response.json",
             cached_response(lang, maxlag, action="query", titles=raw["title"], prop="extracts|revisions",
                             explaintext=1, exsectionformat="wiki", rvprop="ids|timestamp"))
        dump(OUT / "raw" / f"{qid}_{lang}_first_revision_api_response.json",
             cached_response(lang, maxlag, action="query", titles=raw["title"], prop="revisions",
                             rvdir="newer", rvlimit=1, rvprop="ids|timestamp|tags"))
    dump(OUT / "raw" / f"{qid}_wikidata_sitelinks.json",
         {"note": "Excerpt: the batched wbgetentities response for this item (sitelinks en/es/ja).",
          "titles": rec["titles"]})

    articles = [a for a in read_jsonl(WIKI / "articles.jsonl") if a["qid"] == qid]
    write_jsonl(OUT / "processed" / f"{qid}_articles.jsonl", articles)
    sentences = [s for s in read_jsonl(WIKI / "sentences.jsonl") if s["qid"] == qid]
    write_jsonl(OUT / "processed" / f"{qid}_sentences.jsonl", sentences)

    # Stanza words for the lead (what the embedding stage consumes)
    from wikiembed.parse import make_pipeline, word_rows
    from stanza import Document
    parsed = []
    for lang in ("en", "es", "ja"):
        lead = [s for s in sentences if s["lang"] == lang and s["part"] == "lead" and not s["is_list"]]
        nlp = make_pipeline(lang, cfg["embed"]["processors"])
        for s, doc in zip(lead, nlp([Document([], text=s["text"]) for s in lead])):
            parsed.append({"qid": qid, "lang": lang, "text": s["text"],
                           "words": [dict(zip(["text", "lemma", "upos", "start", "end", "head", "deprel"], w))
                                     for w in word_rows(doc)]})
    write_jsonl(OUT / "processed" / f"{qid}_lead_parsed.jsonl", parsed)

    # word-vector entries + reconstruction scores for the topic's head words
    from wikiembed.vectors import WordVectors
    import numpy as np
    wv = WordVectors.load(ROOT / "data" / "embed_all")
    heads = [("lever", "en"), ("palanca", "es"), ("てこ", "ja"), ("fuerza", "es"), ("力", "ja"), ("force", "en")]
    rows = []
    for lemma, lang in heads:
        for i in wv.find(lemma, lang):
            r = wv.vocab[i]
            rows.append(r | {"vector_first_8_dims": np.round(wv.vectors[i][:8], 4).tolist(),
                             "vector_dims": int(wv.vectors.shape[1]),
                             "nearest_other_languages": [
                                 {"lang": t, "neighbors": [[n[0], n[1], round(n[3], 3)] for n in
                                                           wv.neighbors(lemma, lang, target_lang=t, k=3, upos=r["upos"])]}
                                 for t in ("en", "es", "ja") if t != lang]})
    write_jsonl(OUT / "processed" / f"{qid}_word_vectors_excerpt.jsonl", rows)
    recon = []
    for d in ("en-es", "es-en", "en-ja", "ja-en", "es-ja", "ja-es"):
        path = ROOT / "data" / "embed_all" / "reconstruction" / f"{d}.jsonl"
        recon += [{"direction": d} | r for r in read_jsonl(path) if r["lemma"] in {h[0] for h in heads}]
    write_jsonl(OUT / "processed" / f"{qid}_reconstruction_scores.jsonl", recon)


def tatoeba_samples(n: int = 12) -> None:
    try:
        from splitprobe.cases.ser_estar import extract, read_pairs
    except ImportError:
        print("  (splitprobe not available: skipping the Tatoeba sample)")
        return
    data = ROOT / "data" / "raw" / "tatoeba-en-es"
    if not data.exists():
        print("  (no Tatoeba data: skipping the Tatoeba sample)")
        return
    pairs = list(read_pairs(data))
    (en_es, _), _ = extract(pairs)
    picked = sorted(en_es, key=lambda ex: (ex.label, ex.group, ex.text))[::max(1, len(en_es) // n)][:n]
    texts = {ex.text for ex in picked}
    raw = [(en, es) for en, es in pairs if en in texts]
    (OUT / "raw").mkdir(parents=True, exist_ok=True)
    (OUT / "raw" / "tatoeba_en-es_excerpt.en").write_text("\n".join(en for en, _ in raw) + "\n", "utf-8")
    (OUT / "raw" / "tatoeba_en-es_excerpt.es").write_text("\n".join(es for _, es in raw) + "\n", "utf-8")
    write_jsonl(OUT / "processed" / "tatoeba_ser_estar_examples.jsonl",
                [{"text": ex.text, "be_span": list(ex.span), "be_token": ex.text[ex.span[0]:ex.span[1]],
                  "label": ex.label, "group": ex.group} for ex in picked])


PART2_RAW = ["{q}_wikidata_sitelinks.json"] + [f"{{q}}_{l}_{kind}_api_response.json" for l in ("en", "es", "ja")
                                                for kind in ("extract", "first_revision")]
PART2_PROCESSED = ["{q}_articles.jsonl", "{q}_sentences.jsonl", "{q}_lead_parsed.jsonl",
                   "{q}_word_vectors_excerpt.jsonl"]


def lead_before_after(qid: str) -> dict:
    """The overview (lead) section per language: raw API text before the first heading vs the
    cleaned lead and its sentences, side by side."""
    articles = {a["lang"]: a for a in read_jsonl(OUT / "processed" / f"{qid}_articles.jsonl")}
    sentences = list(read_jsonl(OUT / "processed" / f"{qid}_sentences.jsonl"))
    out = {"qid": qid, "languages": {}}
    for lang in ("en", "es", "ja"):
        resp = json.loads((OUT / "raw" / f"{qid}_{lang}_extract_api_response.json").read_text("utf-8"))
        extract = resp["response"]["query"]["pages"][0]["extract"]
        out["languages"][lang] = {
            "title": articles[lang]["title"],
            "url": articles[lang]["url"],
            "raw_lead_from_api": extract.split("\n== ", 1)[0],
            "processed_lead": [p["text"] for p in articles[lang]["lead"]],
            "processed_lead_sentences": [s["text"] for s in sentences
                                         if s["lang"] == lang and s["part"] == "lead"],
        }
    return out


def package_part2(qid: str) -> Path:
    import shutil
    import zipfile
    dest = ROOT / "submission" / "part2_samples"
    if dest.exists():
        shutil.rmtree(dest)
    for sub, names in (("raw", PART2_RAW), ("processed", PART2_PROCESSED)):
        (dest / sub).mkdir(parents=True)
        for name in names:
            shutil.copy2(OUT / sub / name.format(q=qid), dest / sub / name.format(q=qid))
    dump(dest / "processed" / f"{qid}_lead_before_after.json", lead_before_after(qid))
    shutil.copy2(ROOT / "submission" / "TRANSFORMATIONS.md", dest / "TRANSFORMATIONS.md")
    zip_path = ROOT / "submission" / "part2_samples.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(dest.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(dest.parent).as_posix())
    return zip_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qid", default="Q40164")
    ap.add_argument("--package-only", action="store_true", help="only rebuild submission/part2_samples(.zip)")
    args = ap.parse_args()
    cfg = tomllib.loads((ROOT / "config.toml").read_text("utf-8"))
    if not args.package_only:
        wiki_samples(args.qid, cfg)
        tatoeba_samples()
    zip_path = package_part2(args.qid)
    for p in sorted((ROOT / "submission" / "part2_samples").rglob("*")):
        if p.is_file():
            print(f"  {p.relative_to(ROOT).as_posix():<75} {p.stat().st_size / 1024:8.1f} KB")
    print(f"wrote {zip_path.relative_to(ROOT).as_posix()} ({zip_path.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
