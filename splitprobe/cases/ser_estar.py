"""Case study: English "be" -> Spanish ser/estar (a split), and the reverse (a merge).

en->es: L1 = English. For each English "be", which of ser/estar did the translator use,
        and can an English-only model's embedding of "be" predict it?
es->en: L1 = Spanish. For each ser/estar, does the English translation use "be",
        probed with a Spanish-only model.

Alignment is by filtering rather than a word aligner: we keep pairs where each side has
exactly one unambiguous candidate form, so the two must correspond. Crude but transparent.

Usage: python -m splitprobe.cases.ser_estar --data data/raw/tatoeba-en-es [--n 4000]
"""
import argparse
import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

from splitprobe.probe import Example, format_result, run_probe

EN_MODEL = "google-bert/bert-base-cased"
ES_MODEL = "dccuchile/bert-base-spanish-wwm-cased"

# The "be" token itself is the named group `be`.
EN_BE = [
    re.compile(r"\b(?P<be>am|is|are|was|were|be|been|being)\b", re.I),
    re.compile(r"\b(?P<be>is|are|was|were)n['’]t\b", re.I),
    re.compile(r"[A-Za-z](?P<be>['’](?:m|re))\b", re.I),
    # 's only where it is almost always "is", not possessive/"has"
    re.compile(r"\b(?:it|that|what|there|here|he|she|who|where|how|this)(?P<be>['’]s)\b", re.I),
]
# "be" immediately followed by an -ing word: mostly progressive, which forces estar.
EN_PROGRESSIVE = re.compile(r"^(?:n['’]t)?\s+(?:not\s+|never\s+|still\s+|just\s+)?[a-z]+ing\b", re.I)

SER = set("""soy eres es somos sois son era eras éramos erais eran seré serás será seremos seréis
serán sería serías seríamos seríais serían sea seas seamos seáis sean siendo sido""".split())
ESTAR = set("""estoy estás está estamos estáis están estaba estabas estábamos estabais estaban
estuve estuviste estuvo estuvimos estuvisteis estuvieron estaré estarás estará estaremos estaréis
estarán estaría estarías estaríamos estaríais estarían esté estés estemos estéis estén estuviera
estuvieras estuviéramos estuvierais estuvieran estuviese estuvieses estuviésemos estuvieseis
estuviesen estando estar""".split())
# Forms shared with ir / saber / nouns: a pair containing any of these is dropped.
ES_AMBIGUOUS = set("""fui fuiste fue fuimos fuisteis fueron fuera fueras fuéramos fuerais fueran
fuese fueses fuésemos fueseis fuesen sé sed ser seres estado estados""".split())


EN_ANY_S = re.compile(r"[A-Za-z]['’]s\b")


def find_en_be(text: str) -> tuple[list[tuple[int, int]], bool]:
    """Spans of "be" tokens, and whether the sentence has an 's we cannot classify
    (possessive vs. "is", e.g. "Today's your lucky day")."""
    spans = sorted({m.span("be") for pat in EN_BE for m in pat.finditer(text)})
    other_s = sum(1 for _ in EN_ANY_S.finditer(text)) > sum(text[s:e] in ("'s", "’s") for s, e in spans)
    return spans, other_s


def find_es_copula(text: str) -> tuple[list[tuple[int, int, str]], bool]:
    hits, ambiguous = [], False
    for m in re.finditer(r"\w+", text):
        w = m.group().lower()
        if w in SER:
            hits.append((*m.span(), "ser"))
        elif w in ESTAR:
            hits.append((*m.span(), "estar"))
        elif w in ES_AMBIGUOUS:
            ambiguous = True
    return hits, ambiguous


def read_pairs(data_dir: Path):
    en_file = next(data_dir.glob("*.en-es.en"))
    es_file = en_file.with_suffix(".es")
    with open(en_file, encoding="utf-8") as fen, open(es_file, encoding="utf-8") as fes:
        for en, es in zip(fen, fes):
            yield en.strip(), es.strip()


def majority(candidates: dict[str, list[Example]]) -> tuple[list[Example], int]:
    """One example per sentence, keeping the majority label; drop sentences with tied labels.

    Without this, identical sentences with several translations land in both train and
    test folds. Returns (examples, number of sentences translated both ways).
    """
    kept, conflicting = [], 0
    for exs in candidates.values():
        counts = Counter(ex.label for ex in exs).most_common()
        if len(counts) > 1:
            conflicting += 1
            if counts[0][1] == counts[1][1]:
                continue
        kept.append(next(ex for ex in exs if ex.label == counts[0][0]))
    return kept, conflicting


def extract(pairs):
    en_to_es, es_to_en = defaultdict(list), defaultdict(list)
    for en, es in pairs:
        be, en_ambiguous = find_en_be(en)
        cop, es_ambiguous = find_es_copula(es)
        if en_ambiguous or es_ambiguous or len(cop) != 1:
            continue
        s, e, lemma = cop[0]
        if len(be) == 1:
            bs, b_e = be[0]
            group = "be + V-ing (progressive)" if EN_PROGRESSIVE.match(en[b_e:]) else "other be"
            en_to_es[en].append(Example(en, (bs, b_e), "be", lemma, group))
        if len(be) <= 1:
            es_to_en[es].append(Example(es, (s, e), lemma, "be" if be else "no be", lemma))
    return majority(en_to_es), majority(es_to_en)


def sample(examples, n, seed):
    examples = sorted(examples, key=lambda ex: ex.text)
    random.Random(seed).shuffle(examples)
    return examples[:n]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/raw/tatoeba-en-es"))
    ap.add_argument("--n", type=int, default=4000, help="max examples per direction")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--leakage-model", default=None,
                    help="also probe en->es with a multilingual model, e.g. FacebookAI/xlm-roberta-base")
    ap.add_argument("--out", type=Path, default=Path("outputs/ser_estar.json"))
    args = ap.parse_args()

    (en_es, en_es_conf), (es_en, es_en_conf) = extract(read_pairs(args.data))
    print(f"en->es: {len(en_es)} sentences ({en_es_conf} translated with both ser and estar)")
    print(f"es->en: {len(es_en)} sentences ({es_en_conf} translated both with and without 'be')")
    en_es, es_en = sample(en_es, args.n, args.seed), sample(es_en, args.n, args.seed)

    runs = [("en->es  L1 English 'be' -> ser/estar", en_es, EN_MODEL),
            ("es->en  L1 Spanish ser/estar -> 'be'?", es_en, ES_MODEL)]
    if args.leakage_model:
        runs.append(("en->es  leakage control (multilingual)", en_es, args.leakage_model))

    results = {}
    for title, exs, model in runs:
        print(f"\nprobing: {title}")
        r = run_probe(exs, model, seed=args.seed)
        results[title] = asdict(r) | {"unexplained": r.unexplained}
        print(format_result(title, r))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
