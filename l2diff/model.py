"""DifficultyModel: scores how hard an L2 word, sentence or language is for an L1 speaker.

Built on the shared word-vector space (wikiembed) and the reconstruction measure
(wikiembed.reconstruct): an L2 word is rebuilt as a sparse non-negative combination of the
L1's most frequent words.
  gap    = excess: reconstruction error beyond the word's own-language baseline
  split  = collapse: how much closer the word gets to its nearest L2 word through the L1
Raw scores are also given as percentiles against every scored word in that direction, so
"gap 92" means harder to rebuild than 92% of the L2's 5,000 most frequent words.
"""
import json
from bisect import bisect_left
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from wikicollect.store import read_jsonl
from wikiembed.reconstruct import FUNCTION_UPOS, cos_dist, eff_n, fit, top_rows
from wikiembed.vectors import WordVectors

LANGS = ("en", "es", "ja")
SKIP_UPOS = {"PUNCT", "SYM", "NUM", "X", "SPACE"}


@dataclass
class Score:
    lemma: str
    upos: str
    source: str            # "vocab" (stored type vector) or "context" (fresh XLM-R vector)
    residual: float
    self_residual: float
    excess: float
    gap_pct: float
    eff_n: float
    atoms: list            # [[L1 lemma, upos, weight], ...]
    nearest_l2: str | None = None
    collapse: float | None = None
    split_pct: float | None = None
    text: str | None = None

    @property
    def word_class(self) -> str:
        return "function" if self.upos in FUNCTION_UPOS else "content"


@dataclass
class Direction:
    """Everything needed to score words for one L1 -> L2 direction."""
    l1: str
    l2: str
    basis: np.ndarray          # row indices of the L1 basis
    self_rows: np.ndarray      # row indices of the L2's own basis
    self_lemmas: np.ndarray
    target_rows: np.ndarray    # the L2 words the reference distribution was computed on
    excess_ref: list = field(default_factory=list)
    collapse_ref: list = field(default_factory=list)


class DifficultyModel:
    def __init__(self, artifacts: Path | str):
        self.dir = Path(artifacts)
        self.wv = WordVectors.load(self.dir)
        summary = json.loads((self.dir / "reconstruction" / "summary.json").read_text("utf-8"))
        self.cfg = summary["config"]
        self.summary = summary["directions"]
        cfg_path = self.dir / "embed_config.json"
        self.embed_cfg = json.loads(cfg_path.read_text("utf-8")) if cfg_path.exists() else {}
        self._dirs: dict[tuple[str, str], Direction] = {}
        self._results: dict[tuple[str, str], list[dict]] = {}
        self._encoder = None

    # ------------------------------------------------------------------ setup
    def direction(self, l1: str, l2: str) -> Direction:
        if l1 == l2 or l1 not in LANGS or l2 not in LANGS:
            raise ValueError(f"unsupported direction {l1}->{l2}; languages: {', '.join(LANGS)}")
        key = (l1, l2)
        if key not in self._dirs:
            c = self.cfg
            self_rows = top_rows(self.wv, l2, c["basis_size"])
            d = Direction(l1, l2, top_rows(self.wv, l1, c["basis_size"]), self_rows,
                          np.array([self.wv.vocab[i]["lemma"] for i in self_rows]),
                          top_rows(self.wv, l2, c["target_size"]))
            res = self.results(l1, l2)
            d.excess_ref = sorted(r["excess"] for r in res)
            d.collapse_ref = sorted(r["collapse"] for r in res)
            self._dirs[key] = d
        return self._dirs[key]

    def results(self, l1: str, l2: str) -> list[dict]:
        """Precomputed per-word scores for the direction's 5,000 target words."""
        key = (l1, l2)
        if key not in self._results:
            self._results[key] = list(read_jsonl(self.dir / "reconstruction" / f"{l1}-{l2}.jsonl"))
        return self._results[key]

    @staticmethod
    def pct(ref: list, x: float) -> float:
        return round(100.0 * bisect_left(ref, x) / len(ref), 1) if ref else float("nan")

    # ------------------------------------------------------------------ scoring
    def _reconstruct(self, v: np.ndarray, d: Direction, lemma: str):
        c = self.cfg
        atoms, a, rec = fit(v, self.wv.vectors[d.basis], c["candidates"], c["alpha"])
        mask = d.self_lemmas != lemma
        _, _, rec_self = fit(v, self.wv.vectors[d.self_rows][mask], c["candidates"], c["alpha"])
        order = np.argsort(-a)
        atom_list = [[self.wv.vocab[d.basis[atoms[i]]]["lemma"], self.wv.vocab[d.basis[atoms[i]]]["upos"],
                      round(float(a[i]), 3)] for i in order[:6]]
        return rec, float(np.linalg.norm(v - rec)), float(np.linalg.norm(v - rec_self)), eff_n(a), atom_list

    def score_vector(self, v: np.ndarray, l1: str, l2: str, lemma: str, upos: str,
                     source: str, with_split=True) -> Score:
        d = self.direction(l1, l2)
        v = v / np.linalg.norm(v)
        rec, res, res_self, n, atoms = self._reconstruct(v, d, lemma)
        s = Score(lemma, upos, source, round(res, 4), round(res_self, 4), round(res - res_self, 4),
                  self.pct(d.excess_ref, res - res_self), round(n, 2), atoms)
        if with_split:
            # nearest other L2 word among the direction's targets, as in the batch computation
            sims = self.wv.vectors[d.target_rows] @ v
            for j in np.argsort(-sims):
                nb = d.target_rows[j]
                if self.wv.vocab[nb]["lemma"] != lemma:
                    break
            d_l2 = 1.0 - float(sims[j])
            rec_nb, *_ = self._reconstruct(self.wv.vectors[nb], d, self.wv.vocab[nb]["lemma"])
            collapse = 1.0 - cos_dist(rec, rec_nb) / d_l2 if d_l2 > 0 else 0.0
            s.nearest_l2, s.collapse = self.wv.vocab[nb]["lemma"], round(collapse, 4)
            s.split_pct = self.pct(d.collapse_ref, collapse)
        return s

    def word(self, l1: str, l2: str, lemma: str, upos: str | None = None) -> list[Score]:
        """Every vocabulary entry for this lemma (one per part of speech), most frequent first."""
        rows = sorted(self.wv.find(lemma, l2, upos), key=lambda i: -self.wv.vocab[i]["count"])
        return [self.score_vector(self.wv.vectors[i], l1, l2, lemma, self.wv.vocab[i]["upos"], "vocab")
                for i in rows]

    def text(self, l1: str, l2: str, sentence: str) -> dict:
        """Tag a new L2 sentence and score every word. Known (lemma, POS) pairs use the stored
        type vector; unknown words get a contextual XLM-R vector from this sentence."""
        from l2diff.encoder import Encoder
        if self._encoder is None:
            self._encoder = Encoder(self.embed_cfg, self.wv.lang_means)
        words = self._encoder.tag(sentence, l2)
        unknown = [w for w in words if w["upos"] not in SKIP_UPOS and not self.wv.find(w["lemma"], l2, w["upos"])]
        ctx = self._encoder.contextual(sentence, l2, unknown) if unknown else {}
        scores = []
        for k, w in enumerate(words):
            if w["upos"] in SKIP_UPOS:
                continue
            rows = self.wv.find(w["lemma"], l2, w["upos"])
            if rows:
                s = self.score_vector(self.wv.vectors[rows[0]], l1, l2, w["lemma"], w["upos"], "vocab")
            elif id(w) in ctx:
                s = self.score_vector(ctx[id(w)], l1, l2, w["lemma"], w["upos"], "context", with_split=False)
            else:
                continue
            s.text = w["text"]
            scores.append(s)
        gaps = [s.gap_pct for s in scores]
        return {"l1": l1, "l2": l2, "sentence": sentence, "words": scores,
                "sentence_gap_pct": round(float(np.mean(gaps)), 1) if gaps else None,
                "hardest": sorted(scores, key=lambda s: -s.gap_pct)[:3]}

    def pair(self, l1: str, l2: str, k=8, min_count=200) -> dict:
        """Pair-level summary for both directions plus the words that drive each score."""
        out = {}
        for a, b in ((l1, l2), (l2, l1)):
            res = [r for r in self.results(a, b) if r["count"] >= min_count]
            out[f"{a}->{b}"] = {
                "summary": self.summary[f"{a}->{b}"],
                "top_gaps": sorted(res, key=lambda r: -r["excess"])[:k],
                "top_splits": sorted(res, key=lambda r: -r["collapse"])[:k],
                "easiest": sorted(res, key=lambda r: r["excess"])[:k],
            }
        return out
