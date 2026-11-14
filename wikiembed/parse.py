"""Parse stage: Stanza over the selected wiki sentences -> parsed/{lang}.jsonl.

One record per sentence:
  {sid, qid, lang, part, text, words: [[text, lemma, upos, start, end, head, deprel], ...]}
start/end are character offsets of the word's token in `text` (the two words of a Spanish
contraction like "del" = de + el share their token's span). head/deprel are null unless
`depparse` is among the processors.

Resumable: the selection order is deterministic, so a re-run skips the lines already written.
"""
import json
import logging
from collections import defaultdict
from pathlib import Path

from wikicollect.store import read_jsonl

MWT_LANGS = {"es"}  # Stanza needs the multi-word-token expander for these
CHUNK = 256


def select_sentences(wiki_dir: Path, cfg: dict, lang: str) -> list[dict]:
    """Prose sentences of kept articles in the configured topics (empty list = all topics):
    lead first, then body, at most max_sentences_per_article per article (0 = no cap)."""
    e = cfg["embed"]
    topics = set(e["topics"])
    cap = e["max_sentences_per_article"] or float("inf")
    qids = {a["qid"] for a in read_jsonl(wiki_dir / "articles.jsonl")
            if a["keep"] and a["lang"] == lang and (not topics or a["topic"] in topics)}
    per_article: dict[str, list[dict]] = defaultdict(list)
    for s in read_jsonl(wiki_dir / "sentences.jsonl"):
        if (s["lang"] == lang and s["qid"] in qids and not s["is_list"]
                and s["part"] in e["parts"] and len(s["text"]) <= e["max_sentence_chars"]
                and len(per_article[s["qid"]]) < cap):
            s["sid"] = f"{s['qid']}-{lang}-{s['part']}-{s['section_idx']}-{s['para_idx']}-{s['sent_idx']}"
            per_article[s["qid"]].append(s)
    return [s for qid in sorted(per_article) for s in per_article[qid]]


def make_pipeline(lang: str, processors: str):
    import stanza
    logging.getLogger("stanza").setLevel(logging.ERROR)
    procs = processors.split(",")
    if lang in MWT_LANGS and "mwt" not in procs:
        procs.insert(procs.index("tokenize") + 1, "mwt")
    return stanza.Pipeline(lang, processors=",".join(procs), tokenize_no_ssplit=True, verbose=False,
                           download_method=stanza.DownloadMethod.REUSE_RESOURCES)


def word_rows(doc) -> list[list]:
    rows = []
    for sent in doc.sentences:  # tokenize_no_ssplit: normally exactly one
        offset = len(rows)
        for w in sent.words:
            # head: index into this record's word list, -1 for the root, None without depparse
            head = None if w.head is None else (-1 if w.head == 0 else offset + w.head - 1)
            rows.append([w.text, w.lemma, w.upos, w.parent.start_char, w.parent.end_char, head, w.deprel])
    return rows


def parse(wiki_dir: Path, out_dir: Path, cfg: dict, lang: str, limit: int | None = None) -> int:
    from stanza import Document

    sentences = select_sentences(wiki_dir, cfg, lang)[:limit]
    out = out_dir / "parsed" / f"{lang}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = sum(1 for _ in open(out, encoding="utf-8")) if out.exists() else 0
    todo = sentences[done:]
    print(f"  {lang}: {len(sentences)} sentences selected, {done} already parsed")
    if not todo:
        return len(sentences)

    nlp = make_pipeline(lang, cfg["embed"]["processors"])
    with open(out, "a", encoding="utf-8") as f:
        for i in range(0, len(todo), CHUNK):
            batch = todo[i:i + CHUNK]
            docs = nlp([Document([], text=s["text"]) for s in batch])
            for s, doc in zip(batch, docs):
                rec = {k: s[k] for k in ("sid", "qid", "lang", "part", "text")} | {"words": word_rows(doc)}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            print(f"  {lang}: parsed {done + i + len(batch)}/{len(sentences)}", end="\r")
    print()
    return len(sentences)
