# l2diff: directional second-language difficulty (English, Spanish, Japanese)

`l2diff` estimates **how hard words and sentences of a language (L2) are for a speaker of another
language (L1)**, separately for each direction: an English speaker learning Japanese is scored differently
from a Japanese speaker learning English.

It runs on our own data: **2,000 topic-matched Wikipedia articles in each of en/es/ja** (1.16M sentences),
collected with the API pipeline in `wikicollect/`. Words are embedded with XLM-R into one shared space
(140k word vectors). An L2 word is then **rebuilt as a sparse combination of L1 words**, which gives two scores:

- **gap**: how much of the word no combination of L1 words covers (a concept the L1 lacks)
- **split**: how much the L1 view blurs the word with its nearest L2 neighbour (a distinction the L1 does not make, e.g. Spanish *ser/estar* through English *be*)

Scores are given as **percentiles** against the L2's 5,000 most frequent words, so "gap 90" means harder
to rebuild than 90% of them.

## Requirements

- **Python 3.11+**: tested on 3.14 (Windows, CPU) and 3.12 (Google Colab)
- Libraries in `requirements.txt` (exact tested versions in `requirements-lock.txt`): PyTorch, Hugging Face
  Transformers, Stanza, scikit-learn, NumPy, pysbd, requests
- About **2 GB of disk** and an internet connection on first run, for:
  - our prebuilt artifacts, **99 MB**, from this repo's [GitHub Release v0.1](https://github.com/cocorowachi/nlp/releases/tag/v0.1)
  - XLM-R base (~1.1 GB, Hugging Face `FacebookAI/xlm-roberta-base`) and Stanza en/es/ja models (~1 GB), downloaded automatically
- No GPU, API keys or accounts needed. Everything runs locally on CPU.

## Setup

```bash
git clone https://github.com/cocorowachi/nlp.git
cd nlp
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
python scripts/setup.py
```

`scripts/setup.py` installs the requirements, downloads and checksum-verifies the artifacts into
`artifacts/release/`, downloads the Stanza and XLM-R models, and runs a smoke test (all three commands
should print `PASS`). To do it by hand instead:

```bash
pip install -r requirements.txt
python scripts/fetch_artifacts.py     # artifacts -> artifacts/release/
python scripts/smoke_test.py
```

## Running

```bash
python -m l2diff pair --l1 en --l2 ja                 # a language pair, both directions
python -m l2diff word --l1 en --l2 es ser             # one L2 word (dictionary form)
python -m l2diff text --l1 en --l2 es "Your sentence"  # any new L2 sentence
python -m l2diff text --l1 ja --l2 en --file essay.txt
```

`--l1` is the learner's native language and `--l2` the language being learned (`en`, `es`, `ja`).
Add `--json` before the command for machine-readable output. `pair` and `word` take ~6 s (loading
vectors). The first `text` call takes ~20 s, because it loads Stanza and XLM-R.

## Examples

### 1. Score a new sentence: English speaker reading Spanish

```
$ python -m l2diff text --l1 en --l2 es "La fotosíntesis es el proceso por el cual las plantas convierten la luz en energía química."
English speaker learning Spanish (en->es)
  word          lemma         POS       gap  split   rebuilt from English
  La            el            DET       36     5    the + a + closely
  fotosíntesis  fotosíntesis  NOUN      76    34    photosynthesis + hydrolysis + intestine
  es            ser           AUX        5    96    be + become + correspond
  proceso       proceso       NOUN      18    28    process + cycle + effort
  por           por           ADP       11    10    by + for + porous
  plantas       planta        NOUN      26    25    plant + plantation + Plant
  convierten    convertir     VERB      62    47    become + convert + transform
  luz           luz           NOUN      29    52    light + sunlight + luminous
  en            en            ADP       11    98    in + on + encode
  energía       energía       NOUN      20    73    energy + electricity + hydroelectricity
  química       químico       ADJ       90    71    chemical + chemical + chemically
  ...
  sentence gap (mean word percentile): 34.5 [low]
  hardest words: química (90), fotosíntesis (76), convierten (62)
```

**How to read it:** each word is tagged by Stanza (lemma, part of speech) and rebuilt from English words
(last column). *es* (= *ser*) has a low **gap** (5): English *be* covers it. But it has a very high **split** (96):
through English, *ser* becomes almost indistinguishable from *estar*, which is exactly the distinction English
speakers struggle with. Cognates like *proceso* and *plantas* score low on both. Words not in our vocabulary
are scored from a fresh XLM-R vector of the sentence and marked `*`.

### 2. One word: why is *ser* hard?

```
$ python -m l2diff word --l1 en --l2 es ser --upos AUX
English speaker learning Spanish (en->es)

  ser (AUX)
    gap     5.0 pct  [low]   excess -0.112
    split  96.0 pct  [very high]   nearest Spanish word: estar
    rebuilt from English: be (0.69), become (0.03), correspond (0.02), s (0.00)   [1.4 effective words]
```

One English word (*be*, weight 0.69) rebuilds *ser* almost completely, so there is no gap. The difficulty
is the split: *ser* and its nearest Spanish neighbour *estar* both collapse onto *be*.

### 3. A language pair in both directions

```
$ python -m l2diff pair --l1 en --l2 ja
English speaker learning Japanese (en->ja)
  content   words: gap (excess) +0.206   split (collapse) 0.407   L1 words per L2 word 9.7
  function  words: gap (excess) +0.268   split (collapse) 0.472   L1 words per L2 word 10.9
  biggest gaps:   ず/SCONJ, 関わる/SCONJ, に/SCONJ, 為/SCONJ, 凡そ/ADV, 極/NOUN, 末期/NOUN, せる/AUX
  easiest:        経済, 投資, 情報, 世紀, 写真, プロジェクト, 国際, ホテル

Japanese speaker learning English (ja->en)
  content   words: gap (excess) +0.155   split (collapse) 0.448   L1 words per L2 word 9.4
  function  words: gap (excess) +0.134   split (collapse) 0.465   L1 words per L2 word 6.0
  ...
  Overall gap: en->ja +0.238 vs ja->en +0.145 -> en->ja is the harder direction.
```

Japanese is harder for English speakers than English is for Japanese speakers, mostly because of grammatical
morphemes (negation ず, purpose 為, causative せる). Loanwords and shared vocabulary (プロジェクト, ホテル)
are easiest. For comparison, en↔es scores close to 0 in both directions.

## Evaluation

```bash
python eval/run_eval.py --artifacts data/embed_all   # as reported (needs the full vectors)
python eval/run_eval.py --artifacts artifacts/release  # with the downloadable artifacts
```

Results are written to `eval/results/report.md` and `results.json`. The test cases in `eval/cases.jsonl` and
`eval/directional_pairs.jsonl` were written **before** scoring. Items whose scores had been seen during
development are marked `dev` and kept out of the headline numbers.

| Task | Metric | l2diff | Baseline |
|---|---|---|---|
| A. Translation retrieval, MUSE test dictionaries | P@1, en→es / es→en / en→ja / ja→en | 0.68 / 0.69 / 0.19 / 0.22 | uncentered vectors: 0.67 / 0.65 / 0.16 / 0.18 |
| B. Hard vs easy words, 40 test cases, 6 directions | AUC [95% CI] | 0.59 [0.42, 0.78] (gap only: 0.90) | nearest-translation distance: 0.98 |
| C. Split direction harder than merge direction, 7 pairs | accuracy | 3/7 | 6/7 |

The current combined score is **below the simple baseline**. The report explains why: the split score
mistakes near-synonyms (*television*~*TV*) for splits, and the gap score under-rates function words. These
are the first things to fix.

## Data and artifacts

| What | Where | How to get it |
|---|---|---|
| Word vectors (66k, float16), reconstruction scores, settings | `artifacts/release/` | `python scripts/fetch_artifacts.py` (GitHub Release, SHA-256 in `artifacts.lock.json`) |
| Full vectors (140k) used for the reported evaluation | `data/embed_all/` | rebuild (below), ~850 MB |
| Wikipedia corpus: 6,000 articles, 1.16M sentences | `data/wiki/` | rebuild with `wikicollect` |
| Evaluation cases and results | `eval/` | in the repo |
| MUSE test dictionaries | `data/raw/muse/` | `https://dl.fbaipublicfiles.com/arrival/dictionaries/{en-es,es-en,en-ja,ja-en}.5000-6500.txt` |
| Pretrained models | Hugging Face / Stanza servers | downloaded automatically |

### Rebuilding everything from scratch

```bash
export WIKICOLLECT_CONTACT="you@example.com"   # Wikimedia asks for contact info in the User-Agent
python -m wikicollect all                       # ~2 h, ~12k API requests -> data/wiki/
python -m wikiembed parse --all-topics --max-per-article 0 --out-dir data/embed_all   # or colab_embed.ipynb on a GPU
python -m wikiembed embed --all-topics --max-per-article 0 --out-dir data/embed_all
python -m wikiembed.reconstruct --vectors data/embed_all      # gap/split scores, all 6 directions
python scripts/make_release.py                                 # -> artifacts/
```

Collection design and statistics are in `docs/wiki-collection-architecture.md`. All settings are in `config.toml`.

## Known limitations

- **Three languages only** (en, es, ja), so pair-level scores rest on 6 directions.
- **The combined difficulty score is weaker than a simple baseline** (see Evaluation). Gap alone is reasonable.
  The split score flags near-synonyms, and function words are under-scored.
- **Japanese word segmentation:** Stanza sometimes splits words wrongly (光合成 → 光 + 合成, てことは → て + こと + は),
  which produces fake gaps.
- **Spanish clitics:** Stanza lemmatizes *se, lo, le, la* all to *él*, so they share one vector.
- **One vector per word:** words with several senses get a blend (Japanese 米 is mostly "America" on
  Wikipedia, not "rice").
- **Encyclopedic text only:** formal written Wikipedia, not conversation. The English article was written first for 92% of topics.
- **Translation retrieval** is weak for Japanese (P@1 ≈ 0.2).
- The compact release reproduces Tasks B and C exactly. Task A needs the full vectors to match the reported numbers.

## Project layout

```
l2diff/         the prototype: model (scoring), encoder (Stanza + XLM-R), CLI
wikicollect/    Wikipedia collection and cleaning pipeline
wikiembed/      word tagging, XLM-R word vectors, reconstruction
eval/           evaluation cases, harness, results
scripts/        setup, artifact fetch/release, smoke test, Colab and submission bundles
splitprobe/     earlier ser/estar probe on Tatoeba (not used by l2diff)
submission/     milestone deliverables (data samples, artifacts)
docs/           design notes and status reports
```

## License and attribution

Corpus text and derived data come from Wikipedia (CC BY-SA 4.0). Every article record keeps its revision
URL. Models: XLM-R (MIT) and Stanza (Apache 2.0). MUSE dictionaries: Facebook Research (CC BY-NC 4.0).
