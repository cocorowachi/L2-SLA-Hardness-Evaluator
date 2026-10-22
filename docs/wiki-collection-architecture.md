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
| `is_disambiguation`, `is_stub` | `pageprops` / stub templates | Unusable |
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
- Drop disambiguation pages and stubs.
- Lead must have at least 3 sentences in every language.
- Lead length ratio (max/min chars, ja counted at about 1 char ≈ 2.5 en chars) ≤ 4.

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
- **Descriptive `User-Agent`** including a contact address, which Wikimedia policy requires.
- **Sequential requests** (no parallelism), `maxlag=5`, exponential backoff on 429 / `maxlag`
  errors, and about 1 request/sec at most.
- **License:** text is CC BY-SA 4.0. Every record keeps `url` + `revid` for attribution, and
  any released derived dataset must be CC BY-SA.
- **Estimated load:** about 6,000 full-text calls plus a few hundred batched calls, roughly
  2 hours at 1 req/s, done once and then cached.

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
  cli.py        python -m wikicollect {seed,resolve,fetch,filter,segment,all} --config ...
config.yaml     sample size, filter thresholds, user agent
```

## 7. Decisions needed before building
1. **Text scope:** lead plus body (recommended: store both, analyze the lead first), or lead only.
2. **Sample size:** about 2,000 triples to start?
3. **Seed source:** Vital articles L4, plus `origin_lang` stratification (recommended), or something else.
4. **User-Agent contact:** which email or URL to put in it.
5. **Translation filter:** drop triples flagged in *any* language (recommended, strict), or only
   when the flag is on es/ja.
