# Directional L2 difficulty

Pairwise, directional difficulty of second-language acquisition (L1 → L2).

## Split probe (`splitprobe/`)

Contrastive analysis predicts that a **split** (one L1 category → several L2 categories,
e.g. English *be* → Spanish *ser/estar*) is harder than a **merge** (the reverse).
The split probe measures this with embeddings:

- **H(T | u)**: translation entropy. Given L1 word *u*, how much choice the L2 forces.
- **H(T | h_L1)**: held-out cross-entropy of a linear probe that predicts the L2 choice
  from a *monolingual* L1 model's contextual embedding of *u*. This is the part of the L2
  distinction the L1 representation cannot see.

High H(T|u) with a probe that fails = a distinction that is invisible from the L1 = hard.
Swapping which language is "L1" gives the other direction.

A shuffled-label control probe should score ≥ H(T|u), which shows the probe is not memorising.
Running the same probe with a multilingual model (`--leakage-model`) shows how much
L2 knowledge leaks into a model that has seen both languages. This is why the L1 side uses
monolingual models.

### Run

```
pip install -r requirements.txt
python -m splitprobe.download en es          # OPUS Tatoeba -> data/raw/tatoeba-en-es
python -m splitprobe.cases.ser_estar --n 4000 --leakage-model FacebookAI/xlm-roberta-base
```

Results are printed and written to `outputs/ser_estar.json`.

### Caveats

- Alignment works by filtering: a pair is kept only when each side has exactly one
  unambiguous candidate form. Forms shared with *ir*/*saber*/nouns (*fue*, *sé*, *estado*)
  and English *'s* that could be possessive are dropped.
- The "progressive" subset is a heuristic (*be* + an *-ing* word) and includes
  adjectives like *interesting*.
- In es→en, "no be" mixes genuine restructurings (*become interested*, *has blond hair*)
  with free translations.
