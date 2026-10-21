"""Huang & Wu (2026) evaluation suite for any toki pona word embedding.

Metrics follow cubedhuang/toki-pona-word2vec exactly:
  retrieval   word->category-centroid cosine above mean + 1 std of its sims (accuracy.py)
  silhouette  complete-linkage cosine agglomerative clustering, best over thresholds 0.05..0.79 (cluster.py)
  RSA         27x27 category-centroid similarity vs Google News word2vec (matrix.py),
              summarized here as Spearman rho over the upper triangle

Embeddings are .npz files with `words` and `vectors`, or gensim .model files.

    uv run --project ../../tp-mt python evaluate.py w2v_full_d24.npz w2v_pure_d24.npz --umap
"""

import argparse
import json
import os

import numpy as np
from scipy.stats import spearmanr
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

from categories import ALL_CATEGORIZED_WORDS, CATEGORIES, CATEGORIES_EN

HERE = os.path.dirname(os.path.abspath(__file__))
CAT_NAMES = sorted(CATEGORIES)


def load_embedding(path):
    if path.endswith(".model"):
        from gensim.models import Word2Vec
        wv = Word2Vec.load(path).wv
        return {w: wv[w] for w in wv.index_to_key}
    data = np.load(path)
    return {str(w): v for w, v in zip(data["words"], data["vectors"])}


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n else v


def centroids(emb, categories):
    out = {}
    for name, words in categories.items():
        vecs = [unit(emb[w]) for w in words if w in emb]
        if vecs:
            out[name] = np.mean(vecs, axis=0)
    return out


def retrieval(emb):
    cents = centroids(emb, CATEGORIES)
    word_cats = {}
    for cat, words in CATEGORIES.items():
        for w in words:
            word_cats.setdefault(w, []).append(cat)
    hits = total = 0
    for word, cats in word_cats.items():
        if word not in emb:
            continue
        v = unit(emb[word])
        sims = {c: float(v @ cv / np.linalg.norm(cv)) if np.linalg.norm(cv) else 0.0 for c, cv in cents.items()}
        threshold = np.mean(list(sims.values())) + np.std(list(sims.values()))
        for c in cats:
            if c in sims:
                total += 1
                hits += sims[c] > threshold
    return hits, total


def silhouette(emb):
    words = sorted(w for w in ALL_CATEGORIZED_WORDS if w in emb)
    X = normalize(np.array([emb[w] for w in words]), norm="l2")
    best = (-1.0, None, None)
    for k in range(5, 80):
        labels = AgglomerativeClustering(n_clusters=None, distance_threshold=k / 100,
                                         metric="cosine", linkage="complete").fit_predict(X)
        if 1 < len(set(labels)) < len(words):
            s = silhouette_score(X, labels, metric="cosine")
            if s > best[0]:
                best = (float(s), k / 100, int(len(set(labels))))
    return best


def similarity_matrix(emb, categories):
    cents = centroids(emb, categories)
    return np.array([[float(cents[a] @ cents[b]) for b in CAT_NAMES] for a in CAT_NAMES])


def english_matrix():
    import gensim.downloader as api
    en = api.load("word2vec-google-news-300")
    return similarity_matrix({w: en[w] for ws in CATEGORIES_EN.values() for w in ws if w in en}, CATEGORIES_EN)


def rsa(tp_matrix, en_matrix):
    iu = np.triu_indices(len(CAT_NAMES), k=1)
    rho, p = spearmanr(tp_matrix[iu], en_matrix[iu])
    return float(rho), float(p)


def plot_umap(embs, out_path):
    import matplotlib.pyplot as plt
    import umap

    primary = {}
    for cat in CAT_NAMES:
        for w in CATEGORIES[cat]:
            primary.setdefault(w, cat)
    cmap = plt.get_cmap("tab20", len(CAT_NAMES))
    fig, axes = plt.subplots(1, len(embs), figsize=(7 * len(embs), 7), squeeze=False)
    for ax, (name, emb) in zip(axes[0], embs.items()):
        words = sorted(w for w in ALL_CATEGORIZED_WORDS if w in emb)
        X = normalize(np.array([emb[w] for w in words]))
        Y = umap.UMAP(n_neighbors=10, metric="cosine", random_state=0).fit_transform(X)
        for (x, y), w in zip(Y, words):
            c = cmap(CAT_NAMES.index(primary[w]))
            ax.scatter(x, y, color=c, s=20)
            ax.annotate(w, (x, y), fontsize=7, color=c)
        ax.set_title(name)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)


def plot_rsa(tp_matrix, en_matrix, name, out_path):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(16, 7.5))
    for ax, m, title in ((axes[0], tp_matrix, f"toki pona ({name})"), (axes[1], en_matrix, "English (Google News)")):
        im = ax.imshow(m, cmap="viridis")
        ax.set_xticks(range(len(CAT_NAMES)), CAT_NAMES, rotation=90, fontsize=7)
        ax.set_yticks(range(len(CAT_NAMES)), CAT_NAMES, fontsize=7)
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("embeddings", nargs="+")
    ap.add_argument("--no-rsa", action="store_true")
    ap.add_argument("--umap", action="store_true")
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    en = None if args.no_rsa else english_matrix()
    embs, report = {}, {}
    for path in args.embeddings:
        name = os.path.basename(path).rsplit(".", 1)[0]
        emb = load_embedding(path)
        embs[name] = emb
        hits, total = retrieval(emb)
        sil, thr, k = silhouette(emb)
        r = {"retrieval": f"{hits}/{total}", "retrieval_acc": hits / total,
             "silhouette": sil, "silhouette_threshold": thr, "n_clusters": k,
             "missing_words": sorted(ALL_CATEGORIZED_WORDS - set(emb))}
        if en is not None:
            m = similarity_matrix(emb, CATEGORIES)
            r["rsa_spearman"], r["rsa_p"] = rsa(m, en)
            plot_rsa(m, en, name, os.path.join(args.out, f"rsa_{name}.png"))
        report[name] = r
        print(f"{name:32s} retrieval {r['retrieval']:>8s}  silhouette {sil:.3f} (t={thr}, k={k})"
              + (f"  RSA rho {r['rsa_spearman']:.3f}" if en is not None else ""))
    if args.umap:
        plot_umap(embs, os.path.join(args.out, "umap.png"))
    with open(os.path.join(args.out, "report.json"), "w") as f:
        json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
