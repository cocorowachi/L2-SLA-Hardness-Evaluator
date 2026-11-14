"""Embed stage: parsed/{lang}.jsonl -> one vector per (lang, lemma, UPOS).

Pass 1 counts every (lemma, UPOS) and reservoir-samples up to max_contexts of its occurrences,
so the contexts averaged are a random sample rather than the first articles in file order.
Pass 2 runs XLM-R only over sentences holding a sampled occurrence, averages the subword pieces
covering each sampled word (middle layer), and sums them per word.
Vectors are then mean-centered per language: multilingual encoders give every language its own
offset, and removing it makes cross-lingual distances reflect meaning (Libovický et al. 2020).
"""
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from wikicollect.store import read_jsonl, write_jsonl


def sample_occurrences(path: Path, cfg: dict, seed=0):
    """-> counts, articles-per-key, and {key: [(line, word_idx), ...]} reservoirs."""
    e = cfg["embed"]
    skip_upos, skip_lemmas = set(e["skip_upos"]), set(e["skip_lemmas"])
    rng = random.Random(seed)
    counts: dict[tuple, int] = defaultdict(int)
    articles: dict[tuple, set] = defaultdict(set)
    reservoir: dict[tuple, list] = defaultdict(list)
    for line, rec in enumerate(read_jsonl(path)):
        for w_idx, (_, lemma, upos, *_rest) in enumerate(rec["words"]):
            if upos in skip_upos or lemma in skip_lemmas or not lemma:
                continue
            key = (lemma, upos)
            counts[key] += 1
            articles[key].add(rec["qid"])
            r = reservoir[key]
            if len(r) < e["max_contexts"]:
                r.append((line, w_idx))
            else:
                j = rng.randrange(counts[key])
                if j < e["max_contexts"]:
                    r[j] = (line, w_idx)
    keep = {k for k, c in counts.items() if c >= e["min_count"]}
    return ({k: counts[k] for k in keep}, {k: len(articles[k]) for k in keep},
            {k: reservoir[k] for k in keep})


def embed_lang(path: Path, cfg: dict, tok, model, layer: int):
    counts, n_articles, reservoir = sample_occurrences(path, cfg)
    keys = sorted(counts)
    key_idx = {k: i for i, k in enumerate(keys)}
    wanted: dict[int, list[tuple[int, int]]] = defaultdict(list)  # line -> [(word_idx, key_idx)]
    for k, occ in reservoir.items():
        for line, w_idx in occ:
            wanted[line].append((w_idx, key_idx[k]))

    # Keep only the text and the sampled words' spans: (text, [(start, end, key_idx), ...])
    sentences = [(rec["text"], [(rec["words"][w][3], rec["words"][w][4], k) for w, k in wanted[line]])
                 for line, rec in enumerate(read_jsonl(path)) if line in wanted]
    del wanted
    sentences.sort(key=lambda x: len(x[0]))  # length-sorted batches waste less padding
    device = next(model.parameters()).device
    dim = model.config.hidden_size
    sums = np.zeros((len(keys), dim), dtype=np.float64)
    n_ctx = np.zeros(len(keys), dtype=np.int64)
    bs = cfg["embed"]["batch_size"]
    for i in range(0, len(sentences), bs):
        batch = sentences[i:i + bs]
        enc = tok([text for text, _ in batch], return_offsets_mapping=True, padding=True,
                  truncation=True, max_length=256, return_tensors="pt")
        offsets = enc.pop("offset_mapping")
        with torch.no_grad():
            h = model(**enc.to(device), output_hidden_states=True).hidden_states[layer].float().cpu()
        for b, (_, spans) in enumerate(batch):
            o = offsets[b]
            for start, end, k in spans:
                mask = (o[:, 0] < end) & (o[:, 1] > start) & (o[:, 1] > o[:, 0])
                if mask.any():  # empty only if the word fell past truncation
                    sums[k] += h[b][mask].mean(0).numpy()
                    n_ctx[k] += 1
        print(f"  embedded {min(i + bs, len(sentences))}/{len(sentences)} sentences", end="\r")
    print()
    ok = n_ctx > 0
    vectors = (sums[ok] / n_ctx[ok, None]).astype(np.float32)
    rows = [{"lemma": k[0], "upos": k[1], "count": counts[k], "n_articles": n_articles[k],
             "n_contexts": int(n)} for k, n, good in zip(keys, n_ctx, ok) if good]
    return rows, vectors


def embed(out_dir: Path, cfg: dict, langs: list[str]) -> int:
    e = cfg["embed"]
    tok = AutoTokenizer.from_pretrained(e["model"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained(e["model"]).eval().to(device)
    print(f"  model on {device}")
    all_rows, raw, means = [], [], {}
    for lang in langs:
        path = out_dir / "parsed" / f"{lang}.jsonl"
        if not path.exists():
            raise SystemExit(f"{path} missing: run `python -m wikiembed parse` first")
        print(f"  {lang}:")
        rows, vecs = embed_lang(path, cfg, tok, model, e["layer"])
        means[lang] = vecs.mean(0)
        all_rows += [{"idx": len(all_rows) + i, "lang": lang} | r for i, r in enumerate(rows)]
        raw.append(vecs)
        print(f"  {lang}: {len(rows)} word vectors")
    raw_all = np.concatenate(raw)
    centered = raw_all - np.stack([means[r["lang"]] for r in all_rows])
    np.savez(out_dir / "vectors.npz", raw=raw_all, centered=centered,
             lang_means=np.stack([means[l] for l in langs]), langs=np.array(langs))
    write_jsonl(out_dir / "vocab.jsonl", all_rows)
    (out_dir / "embed_config.json").write_text(json.dumps(e, indent=2, ensure_ascii=False), "utf-8")
    return len(all_rows)
