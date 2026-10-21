"""Standalone toki pona word2vec (d=24, Huang & Wu 2026) on corpus/mono.jsonl.

Two corpora, as in the paper:
  full  every sentence, named entities and community words kept
  pure  only sentences made entirely of typst.toml words

    uv run --project ../../tp-mt python train_w2v.py
"""

import json
import os
import re
import tomllib

import numpy as np
from gensim.models import Word2Vec

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
WORD_RE = re.compile(r"[A-Za-z]+")
SENT_RE = re.compile(r"(?<=[.!?:;])\s+")


def typst_words():
    with open(os.path.join(ROOT, "categories", "typst.toml"), "rb") as f:
        return [w.removesuffix(".toml") for w in tomllib.load(f)["wordlist"]["list"]]


def sentences():
    with open(os.path.join(ROOT, "corpus", "mono.jsonl")) as f:
        for line in f:
            for sent in SENT_RE.split(json.loads(line)["text"]):
                # names keep their capital (they are distinct types from content words)
                toks = [t if t[0].isupper() and not t.isupper() else t.lower() for t in WORD_RE.findall(sent)]
                if len(toks) >= 2:
                    yield toks


def train(sents, seed=0):
    return Word2Vec(sents, vector_size=24, window=5, min_count=5, sg=1, negative=10,
                    epochs=15, workers=16, seed=seed)


def save(model, name):
    words = list(model.wv.index_to_key)
    np.savez(os.path.join(HERE, f"w2v_{name}_d24.npz"), words=np.array(words), vectors=model.wv.vectors)


def main():
    core = set(typst_words())
    full = list(sentences())
    pure = [s for s in full if all(t in core for t in s)]
    print(f"full: {len(full)} sentences / {sum(map(len, full))} tokens")
    print(f"pure: {len(pure)} sentences / {sum(map(len, pure))} tokens")
    for name, sents in (("full", full), ("pure", pure)):
        save(train(sents), name)
        print(f"saved w2v_{name}_d24.npz")


if __name__ == "__main__":
    main()
