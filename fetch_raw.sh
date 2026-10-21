#!/bin/bash
# Downloads every raw source used by build_corpus.py, embeddings/ and ../tp-mt/grammar into corpus/raw/.
set -euo pipefail
cd "$(dirname "$0")"
RAW=corpus/raw
HF=https://huggingface.co/datasets
mkdir -p "$RAW" && cd "$RAW"

# parallel + monolingual sets (NetherQuartz)
for d in tatoeba-tokipona tokipona-monolingual tokipona-wiki-parallel-mt tokipona-mined-pairs; do
	mkdir -p "$d"
	for s in train validation; do
		curl -sfL -o "$d/$s.parquet" "$HF/NetherQuartz/$d/resolve/main/data/$s-00000-of-00001.parquet"
	done
done
mkdir -p proper-names
curl -sfL -o proper-names/train.parquet "$HF/NetherQuartz/tokipona-proper-names-mt/resolve/main/data/train-00000-of-00001.parquet"

# tok.wikipedia plaintext dump
curl -sfL -o tokwiki.txt.gz "$HF/ffatty/plaintext-wikipedia-full-toki-pona/resolve/main/data/tokwiki_01_of_01.txt.gz"

# lipu Linku dictionary (all languages)
curl -sfL -o linku_words.json "https://api.linku.la/v1/words?lang=*"

# official Tatoeba exports (gold human links)
mkdir -p tatoeba
for l in tok eng cmn; do
	curl -sfL -o "tatoeba/${l}_sentences.tsv.bz2" "https://downloads.tatoeba.org/exports/per_language/$l/${l}_sentences.tsv.bz2"
done
# dated export: lets ../tp-mt/compare.py score only sentences newer than the baseline checkpoints
curl -sfL -o tatoeba/tok_sentences_detailed.tsv.bz2 "https://downloads.tatoeba.org/exports/per_language/tok/tok_sentences_detailed.tsv.bz2"
for l in eng cmn; do
	curl -sfL -o "tatoeba/tok-${l}_links.tsv.bz2" "https://downloads.tatoeba.org/exports/per_language/tok/tok-${l}_links.tsv.bz2"
done

# poki Lapo (incl. lipu tenpo)
[ -d poki ] || git clone -q --depth 1 https://github.com/kulupu-lapo/poki.git

# Huang & Wu (2026) word2vec models + categories; Huang & Joo (2025) nearley grammar
mkdir -p huang-wu
[ -d huang-wu/toki-pona-word2vec ] || git clone -q --depth 1 https://github.com/cubedhuang/toki-pona-word2vec.git huang-wu/toki-pona-word2vec
[ -d huang-wu/ilo-nasin-sin ] || git clone -q --depth 1 https://github.com/cubedhuang/ilo-nasin-sin.git huang-wu/ilo-nasin-sin

echo "raw sources ready in $(pwd)"
