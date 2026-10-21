"""Collates toki pona monolingual + parallel (tok-en, tok-zh) corpora.

Raw sources are expected under corpus/raw/ (see fetch_raw.sh). Outputs:

    corpus/mono.jsonl          {"text", "source", "n_words"}     monolingual toki pona passages
    corpus/parallel.jsonl      {"tok", "en", "zh", "source", "quality", "en_quality", "zh_quality", "unit", "split"}
    corpus/dictionary.jsonl    {"word", "en", "zh", "ku_en"}     lipu Linku definitions
    corpus/stats.json

Per Huang & Wu (2026), sentences with proper names (capitalized tokiponized words)
and community words are kept; only text that is mostly not toki pona is dropped.

    uv run --project ../tp-mt python build_corpus.py
"""

import bz2
import csv
import glob
import gzip
import hashlib
import json
import os
import random
import re
import sys
import tomllib
from collections import Counter

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "corpus", "raw")
OUT = os.path.join(HERE, "corpus")
TP_GPT2 = os.path.join(HERE, "..", "tp-gpt2")

csv.field_size_limit(sys.maxsize)

TOKEN_RE = re.compile(r"[A-Za-z]+|[0-9]+|[^\sA-Za-z0-9]")
NAME_RE = re.compile(r"^[A-Z][a-z]*$")
# tokiponized proper names follow (C)V(n) syllables
NAME_PHONO_RE = re.compile(r"^(?:[jklmnpstw]?[aeiou]n?)+$")
INTERJ_RE = re.compile(r"^(?:a+|n+|m+u+|o+|e+)$")


def load_vocab():
    with open(os.path.join(HERE, "categories", "typst.toml"), "rb") as f:
        typst = [w.removesuffix(".toml") for w in tomllib.load(f)["wordlist"]["list"]]
    with open(os.path.join(RAW, "linku_words.json")) as f:
        linku = json.load(f)
    return typst, linku


def tp_score(text, vocab):
    """Fraction of word tokens that are dictionary words or tokiponized names."""
    words = [t for t in TOKEN_RE.findall(text) if t.isalpha()]
    if not words:
        return 0.0, 0
    ok = sum(1 for w in words
             # all-caps is shouting/speaker labels; "aaa", "nnn", "muuu" are elongated interjections
             if w.lower() in vocab or INTERJ_RE.match(w.lower())
             or (NAME_RE.match(w) and NAME_PHONO_RE.match(w.lower())))
    return ok / len(words), len(words)


def norm_space(text):
    return re.sub(r"[ \t]+", " ", text.replace(" ", " ")).strip()


def norm_key(text):
    return hashlib.md5(re.sub(r"[^a-z]", "", text.lower()).encode()).hexdigest()


# ------------------------------------------------------------------ mono

def strip_markdown(md):
    md = re.sub(r"^---\n.*?\n---\n", "", md, flags=re.S)
    md = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", md)
    md = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", md)
    md = re.sub(r"<[^>]+>", "", md)
    md = re.sub(r"^[#>*\-+]+\s*", "", md, flags=re.M)
    md = md.replace("**", "").replace("__", "").replace("`", "")
    return md


def paragraphs(text):
    for para in re.split(r"\n\s*\n", text):
        para = norm_space(" ".join(line.strip() for line in para.splitlines()))
        if para:
            yield para


def mono_sources():
    for path in sorted(glob.glob(os.path.join(RAW, "poki", "plaintext", "**", "*.md"), recursive=True)):
        with open(path) as f:
            md = f.read()
        meta = re.match(r"^---\n(.*?)\n---\n", md, flags=re.S)
        source = "poki_lapo"
        if meta and "liputenpo" in meta.group(1):
            source = "lipu_tenpo"
        for para in paragraphs(strip_markdown(md)):
            yield para, source

    with gzip.open(os.path.join(RAW, "tokwiki.txt.gz"), "rt") as f:
        for para in paragraphs(f.read()):
            if "__NOTOC__" in para or para.count("*") > 3:
                continue
            yield para, "tok_wikipedia"

    for split in ("train", "validation"):
        df = pd.read_parquet(os.path.join(RAW, "tokipona-monolingual", f"{split}.parquet"))
        for text in df.text:
            yield norm_space(text), "netherquartz_mono"

    with bz2.open(os.path.join(RAW, "tatoeba", "tok_sentences.tsv.bz2"), "rt") as f:
        for row in csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            yield norm_space(row[2]), "tatoeba"

    docs = pd.read_csv(os.path.join(TP_GPT2, "documents.tsv"), sep="\t")
    for text in docs.tok.dropna():
        for para in paragraphs(text):
            yield para, "tp_gpt2_documents"


# ------------------------------------------------------------------ parallel

def read_tatoeba(lang):
    path = os.path.join(RAW, "tatoeba", f"{lang}_sentences.tsv.bz2")
    with bz2.open(path, "rt") as f:
        return {int(r[0]): norm_space(r[2]) for r in csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)}


def parallel_sources():
    tok = read_tatoeba("tok")
    # gold: human-linked Tatoeba pairs
    for lang, key in (("eng", "en"), ("cmn", "zh")):
        other = read_tatoeba(lang)
        with bz2.open(os.path.join(RAW, "tatoeba", f"tok-{lang}_links.tsv.bz2"), "rt") as f:
            for a, b in csv.reader(f, delimiter="\t"):
                a, b = int(a), int(b)
                if a in tok and b in other:
                    yield {"tok": tok[a], key: other[b], "source": "tatoeba", "quality": "gold", "unit": "sentence"}

    # silver: NetherQuartz tatoeba pivot (non-English sources were machine translated to en/zh)
    for split in ("train", "validation"):
        df = pd.read_parquet(os.path.join(RAW, "tatoeba-tokipona", f"{split}.parquet"))
        for r in df.itertuples():
            en_gold = r.source_lang == "English"
            yield {"tok": norm_space(r.tok), "en": norm_space(r.en), "zh": norm_space(r.zh),
                   "source": "tatoeba_pivot", "quality": "gold" if en_gold else "silver", "unit": "sentence",
                   "zh_quality": "gold" if r.source_lang == "Chinese" else "silver"}

    for split in ("train", "validation"):
        df = pd.read_parquet(os.path.join(RAW, "tokipona-wiki-parallel-mt", f"{split}.parquet"))
        for r in df.itertuples():
            yield {"tok": norm_space(r.tok), "en": norm_space(r.en), "zh": norm_space(r.zh),
                   "source": "wiki_parallel", "quality": "silver", "unit": "paragraph"}

    df = pd.read_parquet(os.path.join(RAW, "proper-names", "train.parquet"))
    for r in df.itertuples():
        yield {"tok": norm_space(r.tok), "en": norm_space(r.en), "zh": norm_space(r.zh),
               "source": "proper_names", "quality": "synthetic", "unit": "sentence"}

    # long-form: gospel of john + holy grail chapters (multi-sentence context)
    ch = pd.read_csv(os.path.join(TP_GPT2, "chapters.tsv"), sep="\t")
    for r in ch.itertuples():
        if isinstance(r.tok, str) and isinstance(r.eng, str):
            yield {"tok": r.tok.strip(), "en": r.eng.strip(), "zh": r.cmn.strip() if isinstance(r.cmn, str) else "",
                   "source": f"chapters_{r.content_type.replace(' ', '_')}", "quality": "gold", "unit": "document"}


def dictionary_entries(linku):
    for word, w in sorted(linku.items()):
        tr = w.get("translations", {})
        zh = tr.get("zh_hans", tr.get("zh", {})).get("definition", "")
        yield {"word": word, "usage": w.get("usage_category"),
               "en": tr.get("en", {}).get("definition", ""), "zh": zh,
               "ku_en": w.get("ku_data") or {}, "ucsur": w.get("representations", {}).get("ucsur")}


# ------------------------------------------------------------------ main

def main():
    random.seed(0)
    typst, linku = load_vocab()
    vocab = set(linku) | set(typst)
    stats = {"mono": Counter(), "mono_dropped": Counter(), "parallel": Counter(), "parallel_dropped": Counter()}

    seen = set()
    mono = []
    for text, source in mono_sources():
        score, n = tp_score(text, vocab)
        key = norm_key(text)
        if n < 1 or score < 0.85 or key in seen:
            stats["mono_dropped"][source] += 1
            continue
        seen.add(key)
        mono.append({"text": text, "source": source, "n_words": n})
        stats["mono"][source] += 1

    # merge parallel rows on the toki pona side so en/zh from different sources combine
    merged = {}
    rank = {"gold": 0, "silver": 1, "synthetic": 2}
    for row in parallel_sources():
        tok = row["tok"]
        score, n = tp_score(tok, vocab)
        if n < 1 or score < 0.85 or not (row.get("en") or row.get("zh")):
            stats["parallel_dropped"][row["source"]] += 1
            continue
        key = norm_key(tok)
        cur = merged.get(key)
        if cur is None:
            merged[key] = {"tok": tok, "en": row.get("en", ""), "zh": row.get("zh", ""), "source": row["source"],
                           "quality": row["quality"], "unit": row["unit"],
                           "en_quality": row.get("en_quality", row["quality"]) if row.get("en") else None,
                           "zh_quality": row.get("zh_quality", row["quality"]) if row.get("zh") else None}
            continue
        # prefer gold translations per target language
        for lang in ("en", "zh"):
            q, new_q = cur[f"{lang}_quality"], row.get(f"{lang}_quality", row["quality"])
            if row.get(lang) and (q is None or rank[new_q] < rank[q]):
                cur[lang], cur[f"{lang}_quality"] = row[lang], new_q
        if rank[row["quality"]] < rank[cur["quality"]]:
            cur["quality"], cur["source"] = row["quality"], row["source"]

    parallel = list(merged.values())
    for row in parallel:
        stats["parallel"][row["source"]] += 1

    # held-out dev/test from gold Tatoeba sentence pairs; every pair sharing a tok
    # sentence with a held-out pair is held out too, so nothing leaks into train
    gold = [r for r in parallel if r["en_quality"] == "gold" and r["unit"] == "sentence"]
    tok_keys = sorted({norm_key(r["tok"]) for r in gold})
    random.shuffle(tok_keys)
    test_keys, dev_keys = set(tok_keys[:1500]), set(tok_keys[1500:2250])
    zh_gold = sorted({norm_key(r["tok"]) for r in parallel if r["zh_quality"] == "gold" and r["unit"] == "sentence"} - test_keys - dev_keys)
    random.shuffle(zh_gold)
    test_keys |= set(zh_gold[:300])
    for row in parallel:
        k = norm_key(row["tok"])
        row["split"] = "test" if k in test_keys else "dev" if k in dev_keys else "train"
    held = {norm_key(r["tok"]) for r in parallel if r["split"] != "train"}
    mono = [m for m in mono if norm_key(m["text"]) not in held]

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "mono.jsonl"), "w") as f:
        for m in mono:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    with open(os.path.join(OUT, "parallel.jsonl"), "w") as f:
        for p in parallel:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    with open(os.path.join(OUT, "dictionary.jsonl"), "w") as f:
        for d in dictionary_entries(linku):
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    splits = Counter(p["split"] for p in parallel)
    summary = {
        "mono_passages": len(mono),
        "mono_words": sum(m["n_words"] for m in mono),
        "mono_by_source": dict(stats["mono"]),
        "mono_dropped": dict(stats["mono_dropped"]),
        "parallel_pairs": len(parallel),
        "parallel_by_source": dict(stats["parallel"]),
        "parallel_dropped": dict(stats["parallel_dropped"]),
        "parallel_splits": dict(splits),
        "test_en_gold": sum(1 for p in parallel if p["split"] == "test" and p["en_quality"] == "gold"),
        "test_zh_gold": sum(1 for p in parallel if p["split"] == "test" and p["zh_quality"] == "gold"),
    }
    with open(os.path.join(OUT, "stats.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
