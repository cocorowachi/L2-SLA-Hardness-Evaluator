# Wikipedia collection: architecture (draft for approval)

**Goal.** Collect topic-matched but independently written Wikipedia text in English, Spanish
and Japanese, so we can compare how each language community structures explanation
(connectives, ordering, cohesion, clause packaging, stance) with topic held constant.

**Unit of comparison.** One Wikidata item (QID) with an article in all three languages: a
*triple*. The **lead section** is the primary unit for analysis. It has the same function
everywhere (define and summarize), whereas body sections differ in coverage. Body text is
stored as well, for analyses of article organization.

---

## 1. What we extract from each page

### 1.1 Identity and provenance
| Field | Source | Why |
|---|---|---|
| `qid` | Wikidata | Joins the three languages |
| `lang`, `title`, `pageid` | MediaWiki API | Identity |
| `revid`, `rev_timestamp` | `prop=revisions` | Reproducibility: everything is pinned to one revision |
| `url`, `license` (CC BY-SA 4.0) | constructed | Attribution, which the license requires |
| `topic`, `subtopic` | seed list (§2) | Stratification / random effects |

### 1.2 Text
| Field | Content |
|---|---|
| `lead` | Plain text before the first heading, with paragraph breaks kept |
| `sections` | Ordered list of `{level, heading, paragraphs[]}` for the body |
| `heading_skeleton` | Just the heading tree, for comparing article organization |

**Removed:** infoboxes, tables, captions, references and footnote markers (TextExtracts
plain-text mode already drops most of these), plus whole back-matter sections, matched by
heading:

| en | es | ja |
|---|---|---|
| References, Notes, Citations, Sources, Bibliography, Further reading, See also, External links | Referencias, Notas, Bibliografía, Véase también, Enlaces externos, Fuentes | 脚注, 注釈, 出典, 参考文献, 関連項目, 外部リンク |

**Normalized:** eswiki leaves a zero-width space (U+200B) where each footnote reference was,
often with no real space at the sentence boundary (*académico.​​Con…*). These become a single
space in en/es and are removed in ja. Without this, many Spanish leads look like lists and
sentences fail to split.

**Normalized:** each formula is replaced with the placeholder `MATH`. TextExtracts renders a
formula as a run of indented MathML-token lines plus a `{\displaystyle …}` line, which splits the
surrounding sentence across several "paragraphs". The prose before and after is rejoined
(*"the Taylor series of MATH about x = 0 converges…"*). `MATH` parses as a noun phrase.

**Removed:** maintenance tags left in the text: es `[cita requerida]`, `[¿quién?]`, `[¿cuándo?]`
etc.; ja `[英語版]` and other `[…語版]` interlanguage markers, `[注釈]`; en `[link removed]`.

**Normalized:** the bracketed gloss in the lead's first sentence (readings, pronunciations,
foreign-term glosses, e.g. *光合成（こうごうせい、ひかり-、英語: photosynthesis）とは…*, en
"(/ˌfoʊtəˈsɪnθəsɪs/)") is removed from `lead`, and the original is kept in `lead_raw`. These
glosses would otherwise inflate sentence-length and clause measures, and they vary by language
convention, not by how the topic is explained.

**Kept but flagged:** list-like paragraphs (most lines short, without sentence-final
punctuation) get `is_list=true`, so analyses can exclude them without losing them.

### 1.3 Filtering metadata (used to drop translated or unusable articles)
| Field | How | Reason |
|---|---|---|
| `translated_template` | Templates on the talk page: en *Translated page*, es *Traducido ref*, ja *翻訳告知* | Translated articles carry the source language's rhetoric |
| `cx_created` | Tags on the article's **first** revision include `contenttranslation` | Created with Wikipedia's translation tool |
| `created_at` | Timestamp of the first revision | Lets us compute `origin_lang` (the edition that existed first) for each triple |
| `is_disambiguation` | `pageprops` | Unusable (stubs are caught by the lead-length rule) |
| `lead_chars`, `lead_sentences` | computed | Minimum-length filter |

### 1.4 Derived (computed after fetching, not scraped)
- `sentences`: sentence split per paragraph, with `{para_idx, sent_idx, text}`
- Later stage, out of scope here: Stanza UD parse → CoNLL-U, so all three languages get
  the same parser and annotation scheme.

---

## 2. Choosing articles (seeding)

1. **Seed list:** English *Wikipedia:Vital articles/Level/4* (about 10k articles in 11 topic
   areas: People, History, Geography, Arts, Philosophy & religion, Everyday life, Society,
   Biology & health, Physical sciences, Technology, Mathematics). These articles mostly
   exist in all three languages, have real leads, and come with topic labels for free.
2. **English-centric bias:** the seed list is chosen by English editors. We counter this by
   recording `origin_lang` (§1.3) and reporting results stratified by it. If needed, we
   add seeds from es and ja quality-article lists in a later round.
3. **Sampling:** stratified by topic area, for an equal number of triples per area.
   Proposed first run: **~2,000 triples** (about 180 per area). That's about 6,000 pages.

## 3. Pipeline

```
seed ──► resolve ──► fetch ──► filter ──► segment ──► (parse: later stage)
 QIDs    sitelinks    raw JSON   keep/drop   sentences
         en/es/ja     per page   + reasons
```

| Stage | Input → output | API calls (batched) |
|---|---|---|
| **seed** | Vital-articles subpages → `(title_en, topic, subtopic)` | `prop=links` on each subpage |
| **resolve** | en titles → QID → es/ja titles; keep only items with all three | `prop=pageprops` (QID); Wikidata `wbgetentities&props=sitelinks&sitefilter=enwiki\|eswiki\|jawiki`, 50 per call |
| **fetch** | for each (lang, title): full plain text with section markers, revision, page props, first revision, talk-page templates | `prop=extracts&explaintext&exsectionformat=wiki` (1 per call for full text); `prop=revisions\|pageprops`, 50 per call |
| **filter** | Drops a whole triple if any language fails; logs the reason | none |
| **segment** | paragraphs → sentences | none |

**Stages are resumable.** Each stage reads the previous stage's files and skips work
already done. Raw API responses are cached by `(lang, revid)`, so a re-run never fetches
again.

**Filter rules (defaults, all configurable):**
- Drop if `translated_template` or `cx_created` is true in any language.
- Drop disambiguation pages.
- Lead must be non-empty (at least 1 sentence) in every language. Lead length is recorded, not
  filtered on; the ratio filter is available but disabled. See §8 for why.

**Sentence splitting:**
- en/es: `pysbd` (rule-based, handles abbreviations).
- ja: split after 。！？, never inside 「」『』（）.
- Stanza is not used at this stage. It will re-tokenize consistently in the parse stage.

## 4. Storage

```
data/wiki/
  raw/{lang}/{pageid}-{revid}.json   immutable API cache
  seeds.jsonl                        title_en, topic, subtopic
  triples.jsonl                      qid, titles per lang, topic
  articles.jsonl                     one record per (qid, lang): §1 fields + keep/drop + reason
  sentences.jsonl                    qid, lang, part (lead|section), section_idx, para_idx, sent_idx, text
  manifest.json                      run config, code version, timestamps, counts per stage
```

## 5. Etiquette and compliance
- **API only, no HTML scraping.** Wikimedia prefers this, and plain-text extracts are cleaner.
- **Descriptive `User-Agent`** including a contact address. This is required in practice:
  since 2026, Wikimedia limits clients without contact information ("unidentified") to
  **10 requests/min**. A compliant User-Agent gets **200/min**. The pilot hit 429s without one.
  If `api.contact` is empty, the client caps itself at 300 requests at 8/min.
- **Sequential requests** (no parallelism), `maxlag=5`, honoring `Retry-After` on 429/5xx
  (at least 5 s with exponential backoff when it is absent), and ≤ 120 requests/min.
- **License:** text is CC BY-SA 4.0. Every record keeps `url` + `revid` for attribution, and
  any released derived dataset must be CC BY-SA.
- **Estimated load:** about 6 requests per fully fetched triple plus batched checks. Fetching
  stops early for a triple as soon as one language fails. The total depends on the keep rate
  (§8): roughly 15–25k requests, or 2–4 hours at 120/min, done once and then cached.

## 6. Code layout

```
wikicollect/
  api.py        session, User-Agent, maxlag/backoff, response cache
  seed.py       vital-articles subpages -> seeds.jsonl
  resolve.py    titles -> QIDs -> triples.jsonl
  fetch.py      per-page extraction (§1) -> raw/ + articles.jsonl
  clean.py      section parsing, back-matter removal, list flagging
  filters.py    keep/drop rules
  segment.py    sentence splitting -> sentences.jsonl
  store.py      JSONL helpers, manifest
  cli.py        python -m wikicollect {seed,resolve,fetch,filter,segment,all} --config ...
config.toml     sample size, filter thresholds, user agent contact, template names, back-matter headings
```

`articles.jsonl` holds every page that was fetched, with `keep` and `drop_reasons`. Triples
dropped before any full text was fetched (missing page, disambiguation, translation template)
appear only in `fetched.jsonl`. `filter` can be re-run with a stricter config at no cost. A
looser config needs `fetch` again, because pages skipped earlier were never downloaded.

## 7. Decisions (approved 2026-10-22)
1. Text scope: lead and body are both stored; the lead is analyzed first.
2. Sample size: about 2,000 triples.
3. Seeds: Vital articles L4, with `origin_lang` stratification.
4. User-Agent contact: set in `config.toml` (`api.contact`).
5. Translation filter: strict (drop if flagged in any language).

## 8. Pilot findings (18 triples, Arts / Biology / Everyday life / Geography)
- About 95% of Vital Articles L4 items have articles in all three languages.
- Translation-template detection works (*Lake Onega*: en talk page flagged).
- **Japanese leads are short by convention:** 1–4 sentences and about 15–250 characters, around
  an eighth of the English length. Many Spanish leads are also short. With `min_lead_sentences = 3`
  and `max_lead_ratio = 4`, only about 1 in 5 triples survives, so even all ~9,500 L4 triples
  would not reach 2,000.
- **The length filters bias the sample against what we are measuring.** They keep only ja
  articles whose leads look English-like. Lead length is itself a rhetorical-convention
  variable, so it should be recorded as an outcome rather than used to filter.
  **Proposed:** `min_lead_sentences = 1` (drops only empty leads), and the ratio filter
  removed (`max_lead_ratio = inf`), with `lead_chars` kept as a covariate. **Approved 2026-10-22.**

## 9. Full run (2026-10-22)
- 10,017 seeds → 9,589 items with articles in all three languages → **2,000 triples kept**,
  with every topic quota met (181–182 each). About 12k API requests, about 2 hours, no errors.
- Dropped: 81 triples (43 translation template, 27 Content Translation, 8 empty lead,
  3 disambiguation). Translated articles are rarer among Vital Articles than expected.
- `origin_lang`: en 1,848 / ja 93 / es 59. enwiki is older, so the English article usually came first.
  This makes a stratified analysis by origin underpowered for es/ja origin. Seeding from
  ja/es quality lists (§2.2) is needed if origin effects matter.
- Median lead: en 12 sentences / 1,693 chars, es 8 / 1,323, ja 4 / 182.
  Median body sections: en 19, es 11, ja 10.
- After cleaning: **1.16M sentences** (en 533k, es 331k, ja 297k), of which lead sentences are
  en 25k, es 20k, ja 11k. List-flagged: en 6.4%, es 7.0%, ja 13.4%.
- Two cleaning fixes were found after the run and applied offline from the cached raw pages, with no
  re-scraping: formula collapsing (removed about 437k spurious "sentences" of MathML tokens) and
  maintenance-tag removal.
