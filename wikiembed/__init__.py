"""Type-level contextual word vectors for the wiki corpus.

parse: Stanza words (lemma, UPOS, char offsets) for a topic subset of data/wiki/sentences.jsonl
embed: XLM-R middle-layer vectors, subword pieces averaged per word, words averaged over up to
       max_contexts random occurrences per (lang, lemma, UPOS), then per-language mean-centering.
"""
