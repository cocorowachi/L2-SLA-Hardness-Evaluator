"""Keep/drop rules for a triple (architecture §3). A triple is dropped if any language fails.

Rules are split so fetch can skip the expensive full-text requests early:
  cheap_reasons: from batched page metadata (missing, disambiguation, translation template)
  full_reasons:  from built articles (Content Translation origin, lead length, length ratio)
"""


def counts_translation_flag(lang: str, cfg: dict) -> bool:
    mode = cfg["filter"]["drop_translated"]
    if mode == "any":
        return True
    if mode == "es_ja":
        return lang != "en"
    raise ValueError(f"filter.drop_translated must be 'any' or 'es_ja', got {mode!r}")


def cheap_reasons(meta: dict[str, dict | None], cfg: dict) -> list[str]:
    reasons = []
    for lang, m in meta.items():
        if m is None:
            reasons.append(f"{lang}:missing")
            continue
        if m["is_disambiguation"]:
            reasons.append(f"{lang}:disambiguation")
        if m["translated_template"] and counts_translation_flag(lang, cfg):
            reasons.append(f"{lang}:translated_template")
    return reasons


def full_reasons(articles: dict[str, dict], cfg: dict) -> list[str]:
    f = cfg["filter"]
    reasons = []
    for lang, a in articles.items():
        if a["cx_created"] and counts_translation_flag(lang, cfg):
            reasons.append(f"{lang}:cx_created")
        if a["lead_sentences"] < f["min_lead_sentences"]:
            reasons.append(f"{lang}:short_lead")
    weighted = [a["lead_chars"] * (f["ja_char_weight"] if lang == "ja" else 1.0)
                for lang, a in articles.items()]
    if min(weighted) > 0 and max(weighted) / min(weighted) > f["max_lead_ratio"]:
        reasons.append("lead_ratio")
    return reasons


def origin_lang(articles: dict[str, dict]) -> str | None:
    dated = [(a["created_at"], lang) for lang, a in articles.items() if a.get("created_at")]
    return min(dated)[1] if dated else None
