"""Turn a new sentence into scored-ready units: Stanza words, and XLM-R contextual vectors
computed exactly as the stored vectors were (same model, same layer, same per-language centering)."""
import numpy as np


class Encoder:
    def __init__(self, embed_cfg: dict, lang_means: dict | None):
        self.model_name = embed_cfg.get("model", "FacebookAI/xlm-roberta-base")
        self.layer = embed_cfg.get("layer", 8)
        self.processors = embed_cfg.get("processors", "tokenize,pos,lemma")
        self.lang_means = lang_means or {}
        self._nlp = {}
        self._tok = self._model = None

    def tag(self, sentence: str, lang: str) -> list[dict]:
        from stanza import Document
        from wikiembed.parse import make_pipeline, word_rows
        if lang not in self._nlp:
            self._nlp[lang] = make_pipeline(lang, self.processors)
        doc = self._nlp[lang]([Document([], text=sentence)])[0]
        return [dict(zip(("text", "lemma", "upos", "start", "end"), row[:5])) for row in word_rows(doc)]

    def contextual(self, sentence: str, lang: str, words: list[dict]) -> dict[int, np.ndarray]:
        """{id(word): centered vector} for the given words of this sentence."""
        import torch
        from transformers import AutoModel, AutoTokenizer
        if self._model is None:
            self._tok = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModel.from_pretrained(self.model_name).eval()
        enc = self._tok(sentence, return_offsets_mapping=True, truncation=True, max_length=256, return_tensors="pt")
        offsets = enc.pop("offset_mapping")[0]
        with torch.no_grad():
            h = self._model(**enc, output_hidden_states=True).hidden_states[self.layer][0].float()
        mean = self.lang_means.get(lang)
        out = {}
        for w in words:
            mask = (offsets[:, 0] < w["end"]) & (offsets[:, 1] > w["start"]) & (offsets[:, 1] > offsets[:, 0])
            if mask.any():
                v = h[mask].mean(0).numpy()
                out[id(w)] = v - mean if mean is not None else v
        return out
