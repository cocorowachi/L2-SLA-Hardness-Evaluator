# Part 3: Collection and processing artifacts

All data collection and processing was done with the scripts in this package; no step was manual.

## Contents

| Path | What it does |
|---|---|
| `wikicollect/` | **Collection and cleaning pipeline** (Wikipedia) |
| `wikicollect/cli.py` | Entry point: `python -m wikicollect {seed,resolve,fetch,filter,segment,all}` |
| `wikicollect/api.py` | API client: identifies us to Wikimedia (User-Agent with contact), one request at a time (≤ 2/s), honors server back-off (`maxlag`, HTTP 429 `Retry-After`), caches every response on disk |
| `wikicollect/seed.py` | Reads the English *Vital Articles, Level 4* pages → 10,017 titles with topic labels |
| `wikicollect/resolve.py` | English title → Wikidata ID → Spanish and Japanese titles; keeps items with all three |
| `wikicollect/fetch.py` | Downloads each article's plain text, current and first revision, and talk-page templates, until each topic has its quota |
| `wikicollect/filters.py` | Keep/drop rules: translation templates, translation-tool tag, disambiguation pages, empty overviews |
| `wikicollect/clean.py` | Removes zero-width characters, reading glosses, formulas (→ `MATH`), editor tags, back-matter sections; splits lead and sections |
| `wikicollect/segment.py` | Sentence splitting (pysbd for en/es, bracket-aware rule for ja) |
| `wikicollect/store.py` | JSONL read/write and the run manifest |
| `wikiembed/` | **Preparation for the system**: Stanza word tagging (`parse.py`), XLM-R word vectors (`embed.py`), loading/querying vectors (`vectors.py`); `reconstruct.py` is the first analysis built on them |
| `colab_embed.ipynb` | Runs `wikiembed` on the full corpus on a Google Colab GPU |
| `scripts/make_colab_bundle.py` | Packages code + processed data for the Colab notebook |
| `scripts/make_submission_samples.py` | Regenerates the Part 2 sample files |
| `config.toml` | All settings: languages, sample size, filter thresholds, translation-template names, back-matter headings, embedding settings |
| `docs/wiki-collection-architecture.md` | Design notes, pilot findings, full-run statistics |
| `requirements.txt` | Python dependencies |

## How to run

```bash
pip install -r requirements.txt

# 1. Set api.contact in config.toml to an email or project URL.
#    Wikimedia requires contact info in the User-Agent; without it requests are limited to 10/min.

# 2. Collect and clean (≈ 2 hours, ≈ 12,000 API requests; resumable, responses cached)
python -m wikicollect all                 # seed -> resolve -> fetch -> filter -> segment

# Re-apply cleaning/filters after changing config.toml (offline, from the cache)
python -m wikicollect filter
python -m wikicollect segment

# 3. Prepare for the system: word tagging + word vectors
#    (≈ 6–7 h on a CPU for the science/tech subset; ≈ 1–2 h on a Colab GPU for everything)
python -m wikiembed parse
python -m wikiembed embed
python -m wikiembed neighbors 光合成 --lang ja --target en   # quick check
```

A small pilot run: `python -m wikicollect all --out-dir data/wiki-pilot --n-triples 11 --max-seeds-per-topic 15`.

## Outputs (written to `data/`)

| File | Contents |
|---|---|
| `data/wiki/seeds.jsonl` | 10,017 seed titles with topic and subtopic |
| `data/wiki/triples.jsonl` | 9,589 Wikidata items with en/es/ja titles |
| `data/wiki/fetched.jsonl` | Every triple considered, kept/dropped with reasons |
| `data/wiki/raw/{lang}/` | Raw page record per fetched article (text + metadata) |
| `data/wiki/articles.jsonl` | 6,000 cleaned articles (2,000 triples × 3 languages) |
| `data/wiki/sentences.jsonl` | 1.16M sentence records |
| `data/wiki/manifest.json` | Config and counts for each stage |
| `data/embed_all/vectors.npz`, `vocab.jsonl` | 140,027 word vectors and their vocabulary |

## Dataset location

The full dataset (about 1.1 GB including the API cache) is too large to submit. It is kept in our
project folder under `data/`. The processed corpus (`articles.jsonl`, `sentences.jsonl`) is also
packaged as `colab_bundle.zip` (141 MB) in our group's Google Drive folder. All text is from Wikipedia
under CC BY-SA 4.0 and may be shared with attribution; it contains no private or restricted data.
