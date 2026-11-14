"""Load the embed stage's output and query it.

    from wikiembed.vectors import WordVectors
    wv = WordVectors.load("data/embed")
    wv.neighbors("光合成", "ja", target_lang="en")
"""
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from wikicollect.store import read_jsonl

NATIVE_SCRIPT = {"en": "latin", "es": "latin", "ja": "cjk"}


def lemma_script(lemma: str) -> str:
    """Majority script of a lemma's letters: latin, cjk (kanji/kana), other, or none."""
    c = Counter()
    for ch in lemma:
        if ch.isalpha():
            name = unicodedata.name(ch, "")
            c["latin" if "LATIN" in name else
              "cjk" if any(s in name for s in ("CJK", "HIRAGANA", "KATAKANA")) else "other"] += 1
    return c.most_common(1)[0][0] if c else "none"


@dataclass
class WordVectors:
    vocab: list[dict]      # idx, lang, lemma, upos, count, n_articles, n_contexts
    vectors: np.ndarray    # rows aligned with vocab, unit-normalized
    lang_means: dict[str, np.ndarray] | None = None  # raw-space mean per language (centers new vectors)

    @classmethod
    def load(cls, out_dir="data/embed", centered=True, native_script_only=True) -> "WordVectors":
        """native_script_only drops lemmas not written in their language's script: mostly
        English words, units and variables inside Japanese text (about 7% of ja types), which
        would otherwise let reconstruction 'cheat' by finding English inside the ja vocabulary.
        Works with the full embed output and with the compact float16 release (centered only)."""
        out_dir = Path(out_dir)
        data = np.load(out_dir / "vectors.npz")
        v = data["centered" if centered else "raw"].astype(np.float32)
        vocab = list(read_jsonl(out_dir / "vocab.jsonl"))
        if native_script_only:
            keep = [r["idx"] for r in vocab if lemma_script(r["lemma"]) == NATIVE_SCRIPT.get(r["lang"])]
            v = v[keep]
            vocab = [vocab[i] | {"idx": new, "orig_idx": r.get("orig_idx", i)}
                     for new, (i, r) in enumerate((i, vocab[i]) for i in keep)]
        means = None
        if "lang_means" in data.files:
            means = {str(l): m.astype(np.float32) for l, m in zip(data["langs"], data["lang_means"])}
        return cls(vocab, v / np.linalg.norm(v, axis=1, keepdims=True), means)

    def __post_init__(self):
        self._index: dict[tuple[str, str], list[int]] = {}
        for r in self.vocab:
            self._index.setdefault((r["lang"], r["lemma"]), []).append(r["idx"])

    def find(self, lemma: str, lang: str, upos: str | None = None) -> list[int]:
        rows = self._index.get((lang, lemma), [])
        return [i for i in rows if upos is None or self.vocab[i]["upos"] == upos]

    def lang_rows(self, lang: str) -> np.ndarray:
        return np.array([r["idx"] for r in self.vocab if r["lang"] == lang])

    def neighbors(self, lemma: str, lang: str, target_lang: str | None = None, k=10, upos=None):
        """Nearest words by cosine, optionally restricted to one language."""
        idx = self.find(lemma, lang, upos)
        if not idx:
            raise KeyError(f"{lemma!r} ({lang}) not in vocabulary")
        q = self.vectors[idx[0]]
        rows = self.lang_rows(target_lang) if target_lang else np.arange(len(self.vocab))
        sims = self.vectors[rows] @ q
        order = rows[np.argsort(-sims)]
        out = [(self.vocab[i]["lemma"], self.vocab[i]["upos"], self.vocab[i]["lang"], float(self.vectors[i] @ q))
               for i in order if i != idx[0]]
        return out[:k]
