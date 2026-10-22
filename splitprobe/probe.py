"""Generic split probe.

For occurrences of an L1 word u, we know which L2 word T each was translated into.
  H(T | u)          translation entropy: how much choice the L2 forces on the learner
  H(T | h_L1(u,ctx)) estimated by the held-out cross-entropy of a linear probe on the
                    L1 model's contextual embedding of u: how much of that choice the
                    L1 representation cannot see.
The gap between them is the part of the L2 distinction already encoded in L1.
Swapping which side is "L1" gives the other direction.
"""
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import torch
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoModel, AutoTokenizer


@dataclass
class Example:
    text: str               # sentence in the representation (L1) language
    span: tuple[int, int]   # character span of the L1 word in `text`
    word: str               # L1 lemma u
    label: str              # L2 translation category T
    group: str = "all"      # subset tag for breakdowns


@dataclass
class ProbeResult:
    model: str
    n: int
    labels: dict[str, int]
    translation_entropy: float   # H(T | u), bits
    probe_ce: float              # H(T | h_L1), bits (held-out)
    control_ce: float            # probe on shuffled labels; should be >= translation_entropy
    probe_acc: float
    majority_acc: float
    groups: dict[str, dict] = field(default_factory=dict)

    @property
    def unexplained(self) -> float:
        """Fraction of the L2 choice the L1 representation cannot predict."""
        return self.probe_ce / self.translation_entropy if self.translation_entropy else 0.0


def entropy_bits(labels) -> float:
    counts = np.array(list(Counter(labels).values()), dtype=float)
    p = counts / counts.sum()
    return float(-(p * np.log2(p)).sum())


def conditional_entropy_bits(words, labels) -> float:
    """H(T | U) = sum_u p(u) H(T | U=u)."""
    words, labels = np.asarray(words), np.asarray(labels)
    return float(sum((words == u).mean() * entropy_bits(labels[words == u]) for u in np.unique(words)))


def embed_spans(texts, spans, model_name, layers=(-4, -3, -2, -1), batch_size=32) -> np.ndarray:
    """Mean of the chosen hidden layers, averaged over the subword pieces covering each span."""
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).eval()
    vecs = []
    for i in range(0, len(texts), batch_size):
        enc = tok(texts[i:i + batch_size], return_offsets_mapping=True, padding=True,
                  truncation=True, max_length=128, return_tensors="pt")
        offsets = enc.pop("offset_mapping")
        with torch.no_grad():
            hidden = model(**enc, output_hidden_states=True).hidden_states
        h = torch.stack([hidden[l] for l in layers]).mean(0)
        for b, (s, e) in enumerate(spans[i:i + batch_size]):
            o = offsets[b]
            mask = (o[:, 0] < e) & (o[:, 1] > s) & (o[:, 1] > o[:, 0])
            if not mask.any():
                raise ValueError(f"span {s}:{e} not covered by any token in {texts[i + b]!r}")
            vecs.append(h[b][mask].mean(0).numpy())
        print(f"  embedded {min(i + batch_size, len(texts))}/{len(texts)}", end="\r")
    print()
    return np.stack(vecs)


def heldout_probs(X, y, folds=5, seed=0) -> np.ndarray:
    """Out-of-fold class probabilities from an L2-regularised linear probe (C tuned on log loss)."""
    probe = make_pipeline(
        StandardScaler(),
        LogisticRegressionCV(Cs=[1e-3, 1e-2, 1e-1, 1.0], scoring="neg_log_loss", max_iter=3000),
    )
    cv = StratifiedKFold(folds, shuffle=True, random_state=seed)
    return cross_val_predict(probe, X, y, cv=cv, method="predict_proba")


def cross_entropy_bits(P, y_idx) -> np.ndarray:
    return -np.log2(np.clip(P[np.arange(len(y_idx)), y_idx], 1e-12, 1.0))


def run_probe(examples: list[Example], model_name: str, seed=0) -> ProbeResult:
    words = [ex.word for ex in examples]
    labels = np.array([ex.label for ex in examples])
    groups = np.array([ex.group for ex in examples])
    classes, y = np.unique(labels, return_inverse=True)

    X = embed_spans([ex.text for ex in examples], [ex.span for ex in examples], model_name)
    P = heldout_probs(X, y, seed=seed)
    ce = cross_entropy_bits(P, y)

    y_shuffled = np.random.default_rng(seed).permutation(y)
    control = cross_entropy_bits(heldout_probs(X, y_shuffled, seed=seed), y_shuffled)

    by_group = {}
    for g in np.unique(groups):
        m = groups == g
        by_group[str(g)] = {
            "n": int(m.sum()),
            "labels": dict(Counter(labels[m])),
            "translation_entropy": conditional_entropy_bits(np.asarray(words)[m], labels[m]),
            "probe_ce": float(ce[m].mean()),
            "probe_acc": float((P[m].argmax(1) == y[m]).mean()),
        }

    return ProbeResult(
        model=model_name,
        n=len(examples),
        labels={str(c): int((y == i).sum()) for i, c in enumerate(classes)},
        translation_entropy=conditional_entropy_bits(words, labels),
        probe_ce=float(ce.mean()),
        control_ce=float(control.mean()),
        probe_acc=float((P.argmax(1) == y).mean()),
        majority_acc=float(np.bincount(y).max() / len(y)),
        groups=by_group,
    )


def format_result(title: str, r: ProbeResult) -> str:
    lines = [
        f"== {title}  [{r.model}]",
        f"   n={r.n}  labels={r.labels}",
        f"   H(T|u)          translation entropy   {r.translation_entropy:.3f} bits",
        f"   H(T|h_L1)       probe cross-entropy   {r.probe_ce:.3f} bits"
        f"   (acc {r.probe_acc:.3f} vs majority {r.majority_acc:.3f})",
        f"   control         shuffled-label probe  {r.control_ce:.3f} bits",
        f"   unexplained fraction of the L2 choice  {r.unexplained:.2f}",
    ]
    if len(r.groups) > 1:
        for g, s in r.groups.items():
            lines.append(f"     - {g:<28} n={s['n']:<5} H(T|u)={s['translation_entropy']:.3f}"
                         f"  probe={s['probe_ce']:.3f}  acc={s['probe_acc']:.3f}  {s['labels']}")
    return "\n".join(lines)
