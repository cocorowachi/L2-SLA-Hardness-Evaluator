"""Turn a raw page record (plain-text extract with == section == markers) into the article
fields of architecture §1.2: lead, lead_raw, sections, heading_skeleton."""
import re
import urllib.parse

from wikicollect.segment import split_sentences

SECTION = re.compile(r"^(={2,6})\s*(.*?)\s*\1\s*$")
OPEN, CLOSE = "(（", ")）"
TERMINALS = (".", "!", "?", ":", "。", "！", "？", "：", "」", "）", ")", '"', "”")


MATH = "MATH"  # placeholder for a formula; parses as a noun phrase
MAINTENANCE_TAG = re.compile(
    r"\[(?:cita requerida|¿quién\?|¿cuál\?|¿cuándo\?|¿dónde\?|¿por quién\?|actualizar"
    r"|link removed|[^\[\]\n]{1,10}語版|注釈)\]"
)


def is_indented(line: str) -> bool:
    return line[:1] in (" ", "\t")


def collapse_math(text: str) -> str:
    """TextExtracts renders each formula as a run of indented MathML-token lines ending in a
    `{\\displaystyle ...}` line, which splits the surrounding sentence across "paragraphs".
    Replace each such block with MATH and rejoin the prose before and after it."""
    lines, out, i = text.split("\n"), [], 0
    while i < len(lines):
        if is_indented(lines[i]):
            j = i
            while j < len(lines) and is_indented(lines[j]) and "{\\displaystyle" not in lines[j]:
                j += 1
            if j < len(lines) and is_indented(lines[j]):
                k = j + 1
                while k < len(lines) and lines[k] and not lines[k].strip():
                    k += 1
                prev = out.pop().rstrip() if out and out[-1].strip() else ""
                merged = f"{prev} {MATH}".strip()
                if k < len(lines) and lines[k].strip() and not SECTION.match(lines[k].strip()):
                    merged += " " + lines[k].strip()
                    k += 1
                out.append(merged)
                i = k
                continue
        out.append(lines[i])
        i += 1
    return "\n".join(out)


def normalize_extract(text: str, lang: str) -> str:
    """Formulas -> MATH; drop maintenance tags ([cita requerida], [英語版], ...).
    eswiki leaves a zero-width space where each footnote reference was, often with no real
    space at the sentence boundary ("académico.​​Con ..."). Turn them into a single
    space (removed entirely for ja, which has no inter-word spaces)."""
    text = MAINTENANCE_TAG.sub("", collapse_math(text))
    if lang == "ja":
        text = re.sub(rf" ?{MATH} ?", MATH, text)
    text = text.replace("​", "" if lang == "ja" else " ").replace(" ", " ")
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r" +([,.;:])", r"\1", text)
    return re.sub(r" +\n", "\n", text)


def parse_extract(text: str) -> tuple[list[str], list[dict]]:
    """Split an extract into lead paragraphs and [{level, heading, paragraphs}]."""
    lead: list[str] = []
    sections: list[dict] = []
    current = lead
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        m = SECTION.match(line)
        if m:
            sections.append({"level": len(m.group(1)), "heading": m.group(2), "paragraphs": []})
            current = sections[-1]["paragraphs"]
        else:
            current.append(line)
    return lead, sections


def drop_back_matter(sections: list[dict], headings: list[str]) -> list[dict]:
    """Remove sections whose heading is back matter, together with their subsections."""
    drop = {h.casefold() for h in headings}
    kept, skip_below = [], None
    for s in sections:
        if skip_below is not None and s["level"] > skip_below:
            continue
        skip_below = None
        if s["heading"].casefold() in drop:
            skip_below = s["level"]
            continue
        kept.append(s)
    return [s for s in kept if s["paragraphs"]]


def strip_first_sentence_brackets(para: str) -> str:
    """Remove top-level (…)/（…） groups in the first sentence: readings, pronunciations,
    etymologies, foreign-term glosses. Left unchanged if the brackets are unbalanced."""
    out, depth = [], 0
    for i, c in enumerate(para):
        if c in OPEN:
            depth += 1
        elif c in CLOSE and depth:
            depth -= 1
        elif depth == 0:
            out.append(c)
            if c == "。" or (c in ".!?" and (i + 1 == len(para) or para[i + 1] == " ")):
                rest = para[i + 1:]
                break
    else:
        if depth:
            return para
        rest = ""
    first = "".join(out)
    first = re.sub(r"\s{2,}", " ", first)
    first = re.sub(r"\s+([,.;:、。])", r"\1", first)
    return first + rest


def is_list_like(para: str) -> bool:
    return not para.rstrip().endswith(TERMINALS)


def permalink(lang: str, title: str, revid: int) -> str:
    return f"https://{lang}.wikipedia.org/w/index.php?title={urllib.parse.quote(title)}&oldid={revid}"


def build_article(raw: dict, back_matter: list[str]) -> dict:
    """Article record from a raw page record (everything except keep/drop decisions)."""
    lead_raw, sections = parse_extract(normalize_extract(raw["extract"], raw["lang"]))
    lead = list(lead_raw)
    if lead:
        lead[0] = strip_first_sentence_brackets(lead[0])
    sections = drop_back_matter(sections, back_matter)
    paras = lambda ps: [{"text": p, "is_list": is_list_like(p)} for p in ps]  # noqa: E731
    lead_text = [p for p in lead if not is_list_like(p)]
    return {
        "qid": raw["qid"], "lang": raw["lang"], "title": raw["title"], "pageid": raw["pageid"],
        "revid": raw["revid"], "rev_timestamp": raw["rev_timestamp"],
        "url": permalink(raw["lang"], raw["title"], raw["revid"]), "license": "CC BY-SA 4.0",
        "lead": paras(lead),
        "lead_raw": "\n".join(lead_raw),
        "sections": [{"level": s["level"], "heading": s["heading"], "paragraphs": paras(s["paragraphs"])}
                     for s in sections],
        "heading_skeleton": [[s["level"], s["heading"]] for s in sections],
        "lead_chars": sum(len(p) for p in lead_text),
        "lead_sentences": sum(len(split_sentences(p, raw["lang"])) for p in lead_text),
        "translated_template": raw["translated_template"],
        "cx_created": raw["cx_created"],
        "created_at": raw["created_at"],
        "is_disambiguation": raw["is_disambiguation"],
    }
