# Part 2: Raw and processed data sample

**Example:** one topic, *Lever* (en) / *Palanca* (es) / てこ (ja), Wikidata ID **Q40164**, traced from the
raw Wikipedia API responses to the records our system uses. We chose it because its raw text contains
every cleaning problem we handle: hidden characters (Spanish), a reading gloss (Japanese), formulas
(English) and reference sections (all three).

## Files

### `raw/`: original data, exactly as the APIs returned it

| File | Source | Contents |
|---|---|---|
| `Q40164_wikidata_sitelinks.json` | Wikidata API | Links the item to its en/es/ja article titles (*Lever*, *Palanca*, てこ) |
| `Q40164_{en,es,ja}_extract_api_response.json` | MediaWiki API | The request we sent and the full response: the article's plain text with `== Section ==` markers, plus its revision ID |
| `Q40164_{en,es,ja}_first_revision_api_response.json` | MediaWiki API | The article's first edit: creation date and tags (used to detect Wikipedia's translation tool) |

### `processed/`: what the system consumes

| File | One record per | Key fields |
|---|---|---|
| `Q40164_articles.jsonl` | article (3) | `qid`, `lang`, `title`, `url` (link to the exact revision), `license`, `topic`, `subtopic`, `lead` (cleaned overview paragraphs), `lead_raw` (original overview), `sections` (heading + paragraphs), `heading_skeleton`, `lead_chars`, `lead_sentences`, `origin_lang`, `keep`, `drop_reasons` |
| `Q40164_sentences.jsonl` | sentence (245) | `qid`, `lang`, `part` (`lead`/`section`), `heading`, `section_idx`, `para_idx`, `sent_idx`, `is_list`, `text` |
| `Q40164_lead_parsed.jsonl` | overview sentence | `text`, `words` (each with `text`, `lemma` = dictionary form, `upos` = part of speech, `start`/`end` = character offsets) |
| `Q40164_word_vectors_excerpt.jsonl` | word (*lever, palanca, てこ, force, fuerza, 力*) | `lang`, `lemma`, `upos`, `count`, `n_articles`, the first 8 of its 768 vector dimensions, its nearest words in the other two languages |
| `Q40164_lead_before_after.json` | language | The overview section side by side: raw API text, cleaned paragraphs, and the resulting sentences |

## What changed and why

| # | Change | Before (raw) | After (processed) | Why it is necessary |
|---|---|---|---|---|
| 1 | Linked across languages | Three unrelated pages | `qid: "Q40164"` on every record | Lets us compare the same topic in all three languages |
| 2 | Translation check | Talk-page templates + first-revision tags | `keep: true`, `drop_reasons: []`, `origin_lang: "en"` | Translated articles copy the source language's structure, which is what we compare against |
| 3 | Hidden characters removed (es) | `La palanca` + two invisible U+200B characters + ` es una máquina simple` | `La palanca es una máquina simple` | These characters mark removed footnotes and broke sentence splitting |
| 4 | Reading gloss removed (ja) | `てこ（梃子、梃，英語: lever）とは、弱い力で…` | `てことは、弱い力で…` (original kept in `lead_raw`) | A language-specific convention that would inflate sentence-length measures |
| 5 | Formulas replaced (en) | Dozens of lines of formula markup ending in `{\displaystyle …}` | `Since the moments of torque must be balanced, MATH.` | Each formula otherwise split its sentence into fake fragments |
| 6 | Reference sections dropped | es: *Véase también*, *Referencias*, *Bibliografía*, *Enlaces externos* (and their en/ja equivalents) | Removed; the remaining headings kept in `heading_skeleton` | Not explanatory text |
| 7 | Sentence splitting | Paragraphs | 245 sentences (en 86, es 82, ja 77) | The sentence is the unit we parse and embed |
| 8 | Words tagged | `A lever is a simple machine…` | `a/DET lever/NOUN be/AUX a/DET simple/ADJ machine/NOUN…` | Analysis works on dictionary words (*is*, *was* → *be*); offsets map words onto the embedding model's pieces |
| 9 | Word vectors | Words in many sentences | One 768-dimension vector per word, shared space for all three languages | The input to our cross-language comparison (e.g. *fuerza*'s nearest English word is *force*, similarity 0.74) |

## Preserved and removed

**Preserved:** all explanatory prose, the order and headings of sections, the original overview text
(`lead_raw`), the source link and revision ID, the license (CC BY-SA 4.0), the topic label, and the
filtering metadata.

**Removed:** footnote markers, reference/link sections, infoboxes and tables (already excluded by the
API's plain-text mode), editor tags such as `[cita requerida]` ("citation needed"), and formula markup.

**Flagged, not removed:** list-like lines (`is_list: true`), so analyses can exclude them without
losing them.

## Issues visible in this sample

We show these deliberately; they are the open problems listed in our report.

- **Japanese word segmentation.** The tagger splits the first words of the Japanese overview,
  てことは ("a lever is…"), as て + こと + は instead of てこ + とは (`Q40164_lead_parsed.jsonl`). As a
  result, てこ itself gets no word vector. We plan to re-segment Japanese into dictionary-sized words.
- **Weak vectors for less frequent words.** *lever* (156 occurrences in the corpus) and *palanca* (111)
  have weaker vectors than common words: *palanca*'s nearest English word is *crane*, not *lever*. By
  contrast, *fuerza* → *force* is a clear match (0.74). This is why the system weights results by word
  frequency and validates against known word pairs.

## Reproducing

All files are generated from the collected data by `scripts/make_submission_samples.py`; the collection
and cleaning code is in `wikicollect/` (see Part 3).
