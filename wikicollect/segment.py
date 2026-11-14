"""Sentence splitting: pysbd for en/es, a bracket-aware rule for ja.

The later parse stage (Stanza) re-tokenizes everything consistently; this split only needs
to be good enough for counting and for the sentences.jsonl unit."""
import pysbd

_segmenters: dict[str, pysbd.Segmenter] = {}
JA_OPEN, JA_CLOSE, JA_END = "「『（(【", "」』）)】", "。！？!?"


def split_sentences(text: str, lang: str) -> list[str]:
    if lang == "ja":
        return split_ja(text)
    seg = _segmenters.setdefault(lang, pysbd.Segmenter(language=lang, clean=False))
    return [s.strip() for s in seg.segment(text) if s.strip()]


def split_ja(text: str) -> list[str]:
    """Split after 。！？ unless inside 「」『』（）【】."""
    out, buf, depth = [], [], 0
    for c in text:
        buf.append(c)
        if c in JA_OPEN:
            depth += 1
        elif c in JA_CLOSE and depth:
            depth -= 1
        elif c in JA_END and depth == 0:
            out.append("".join(buf).strip())
            buf = []
    if "".join(buf).strip():
        out.append("".join(buf).strip())
    return [s for s in out if s]


def article_sentences(article: dict) -> list[dict]:
    """Sentence records for one article (lead first, then body sections)."""
    rows = []
    parts = [("lead", None, None, article["lead"])]
    parts += [("section", i, s["heading"], s["paragraphs"]) for i, s in enumerate(article["sections"])]
    for part, sec_idx, heading, paragraphs in parts:
        for p_idx, para in enumerate(paragraphs):
            for s_idx, sent in enumerate(split_sentences(para["text"], article["lang"])):
                rows.append({"qid": article["qid"], "lang": article["lang"], "part": part,
                             "section_idx": sec_idx, "heading": heading, "para_idx": p_idx,
                             "sent_idx": s_idx, "is_list": para["is_list"], "text": sent})
    return rows
