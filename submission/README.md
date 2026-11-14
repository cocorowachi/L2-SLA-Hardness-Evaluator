# Data Collection and Preparation: samples and artifacts

## Part 2: raw and processed data samples (`samples/`)

### Example A: one Wikipedia triple, *Lever* / *Palanca* / てこ (Wikidata Q40164)

| Stage | File(s) | Format |
|---|---|---|
| Raw: cross-language link | `raw/Q40164_wikidata_sitelinks.json` | Wikidata API JSON |
| Raw: article text + revision | `raw/Q40164_{en,es,ja}_extract_api_response.json` | MediaWiki API JSON, exactly as cached during collection (request params included) |
| Raw: first revision (creation date, translation-tool tag) | `raw/Q40164_{en,es,ja}_first_revision_api_response.json` | MediaWiki API JSON |
| Processed: article records | `processed/Q40164_articles.jsonl` | JSONL, 1 record per language |
| Processed: sentence records | `processed/Q40164_sentences.jsonl` | JSONL, 245 sentences |
| Processed: words for the lead | `processed/Q40164_lead_parsed.jsonl` | JSONL, Stanza words (lemma, part of speech, character offsets) |
| Processed: word vectors | `processed/Q40164_word_vectors_excerpt.jsonl` | JSONL, vocabulary entries for *lever / palanca / てこ / force / fuerza / 力*, first 8 of 768 dimensions, nearest words in the other languages |
| Processed: system output | `processed/Q40164_reconstruction_scores.jsonl` | JSONL, gap/split scores for those words in all six learner directions |

**What changed, and why** (all examples are from these files):

1. **Matched across languages.** Wikidata links *Lever*, *Palanca* and てこ as one item. The Wikidata ID (`qid`) is kept on every record. *Why:* it's the key that makes the three articles comparable.
2. **Checked for translation.** None of the three talk pages has a translation template, and none of the first revisions carries a translation-tool tag, so the triple is kept (`keep: true`). The first-revision timestamps give `origin_lang: "en"`. *Why:* translated articles would carry the source language's structure.
3. **Spanish zero-width spaces.** Raw: `La palanca​​ es una máquina simple​​ cuya…`. Processed: `La palanca es una máquina simple cuya…`. *Why:* these invisible characters mark removed footnotes, and they hid sentence boundaries.
4. **Japanese reading gloss.** Raw: `てこ（梃子、梃，英語: lever）とは、…`. Processed `lead`: `てことは、…`; the original stays in `lead_raw`. *Why:* the gloss is a language-specific convention that inflates sentence-length measures.
5. **Formulas.** Raw English contains each formula as dozens of lines of MathML tokens plus `{\displaystyle …}`. Processed: `Since the moments of torque must be balanced, MATH.` *Why:* otherwise every formula breaks its sentence into fake fragments.
6. **Back matter.** Spanish `Véase también`, `Referencias`, `Bibliografía` and `Enlaces externos` are removed. Their equivalents are removed in en/ja too. The remaining outline is kept in `heading_skeleton`. *Why:* not explanatory prose.
7. **Structure kept.** Lead and body sections are kept separately, with headings, paragraph and sentence indices, a list flag (`is_list`), and the revision permalink and license (CC BY-SA 4.0). *Why:* analyses compare leads across languages, and the license requires attribution.
8. **Sentences → words.** Stanza adds lemma, part of speech and character offsets (`A lever is…` → `a/DET lever/NOUN be/AUX …`). *Why:* the analysis unit is the dictionary word; offsets map words onto the embedding model's subword pieces.
9. **Words → vectors → scores.** Each lemma gets one 768-dimension vector in a space shared by the three languages; the reconstruction scores whether each L2 word can be rebuilt from L1 words. *Why:* this is the lexical-conceptual signal the difficulty model consumes.

**Removed:** footnote markers, reference and link sections, infoboxes and tables (dropped by the API's plain-text export), maintenance tags, raw formula markup. **Preserved:** all prose, section order and headings, the original lead text, provenance (URL + revision ID), license, and filtering metadata.

### Example B: Tatoeba English–Spanish pairs for the *ser/estar* probe

| Stage | File | Format |
|---|---|---|
| Raw | `raw/tatoeba_en-es_excerpt.en` + `.es` | Aligned plain text, line *i* = translation pair *i* |
| Processed | `processed/tatoeba_ser_estar_examples.jsonl` | JSONL: English sentence, character span of its *be* token, Spanish label (`ser`/`estar`), group (progressive or other) |

**What changed:** a pair is kept only when the English side has exactly one unambiguous *be* and the Spanish side exactly one unambiguous *ser/estar* form. Forms shared with *ir*/*saber*/nouns, and English *'s* that might be possessive, are excluded. Duplicate English sentences are merged to one majority label. *Why:* this aligns *be* with its Spanish translation without a word aligner, and the merging stops duplicates appearing in both training and test folds.

## Part 3: collection and processing artifacts

| Artifact | What it does |
|---|---|
| `wikicollect/` | Wikipedia collector: `python -m wikicollect {seed,resolve,fetch,filter,segment,all}` |
| `wikicollect/api.py` | API client: contact User-Agent, rate limit, retry/back-off, response cache |
| `wikicollect/clean.py`, `filters.py`, `segment.py` | Cleaning, keep/drop rules, sentence splitting |
| `config.toml` | All settings (sample size, filters, template names, back-matter headings) |
| `docs/wiki-collection-architecture.md` | Design, pilot findings, full-run statistics |
| `wikiembed/` | Stanza parsing, XLM-R word vectors, reconstruction (`python -m wikiembed …`) |
| `colab_embed.ipynb`, `scripts/make_colab_bundle.py` | GPU run of the embedding pipeline on Google Colab |
| `splitprobe/`, `splitprobe/download.py` | Tatoeba download and the *ser/estar* probe |
| `scripts/make_submission_samples.py` | Regenerates every file in `samples/` |

**Dataset location.** The full dataset is too large to submit directly (about 1.1 GB with the API cache):

- `data/wiki/articles.jsonl` (200 MB) and `data/wiki/sentences.jsonl` (353 MB): the processed Wikipedia corpus
- `data/wiki/raw/`: the stored raw page record for every fetched article
- `data/embed_all/`: word vectors (`vectors.npz`, `vocab.jsonl`) and reconstruction scores
- `data/raw/tatoeba-en-es/`: Tatoeba as downloaded from OPUS

The processed corpus is also packaged as `colab_bundle.zip` (141 MB) in the group's Google Drive folder, `MyDrive/nlp/`. All Wikipedia text is CC BY-SA 4.0 and Tatoeba is CC BY 2.0 FR, so the data can be shared with attribution. No restricted data is included.
