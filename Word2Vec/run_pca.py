"""
Ce script sert à visualiser uniquement la PCA globale du corpus COHA.
"""
import re
import os
import glob
from collections import Counter
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords


CORPUS_DIR  = "../COHA_sample/"
TOP_N_WORDS = 3500
WINDOW_SIZE = 5
N_CLUSTERS  = 5

STOPWORDS = set(stopwords.words("english"))

def remove_stopwords(tokens):
    return [w for w in tokens if w not in STOPWORDS]

def remove_punct(tokens):
    cleaned = []
    for w in tokens:
        w = re.sub(r"[^a-z0-9]", "", w)
        if w:
            cleaned.append(w)
    return cleaned


### Chargement
all_tokens = []
files = glob.glob(os.path.join(CORPUS_DIR, "**", "*.txt"), recursive=True)
print(f"{len(files)} fichiers trouvés.")

for fpath in files:
    try:
        with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read().lower()
        tokens = remove_punct(text.split())
        tokens = remove_stopwords(tokens)
        all_tokens.extend(tokens)
    except Exception:
        continue

print(f"Tokens totaux (après nettoyage) : {len(all_tokens):,}")

#Constrcution du vocabulaire
word_freq = Counter(all_tokens)
top_words = [w for w, _ in word_freq.most_common(TOP_N_WORDS)]
word_to_idx = {w: i for i, w in enumerate(top_words)}
vocab = top_words
size  = len(vocab)

print(f"Vocabulaire : {size} mots")

# Matrice de co-occurrence
print("Construction de la matrice de co-occurrence...")
cooc = np.zeros((size, size), dtype=np.float32)

for i, token in enumerate(all_tokens):
    if token not in word_to_idx:
        continue
    idx = word_to_idx[token]
    start = max(0, i - WINDOW_SIZE)
    end   = min(len(all_tokens), i + WINDOW_SIZE + 1)
    for j in range(start, end):
        if j == i:
            continue
        ctx = all_tokens[j]
        if ctx in word_to_idx:
            cooc[idx, word_to_idx[ctx]] += 1

# PPMI
print("Calcul PPMI...")
total    = cooc.sum()
word_sum = cooc.sum(axis=1, keepdims=True)   # P(w)
ctx_sum  = cooc.sum(axis=0, keepdims=True)   # P(c)
with np.errstate(divide="ignore", invalid="ignore"):
    ppmi = np.log2((cooc * total) / (word_sum * ctx_sum + 1e-9))
ppmi = np.maximum(ppmi, 0)                   # PMI positive

#PCA
print("PCA...")
reducer = PCA(n_components=2, random_state=42)
data_2d = reducer.fit_transform(ppmi)
print(f"Variance expliquée : PC1={reducer.explained_variance_ratio_[0]:.2%}, "
      f"PC2={reducer.explained_variance_ratio_[1]:.2%}")

# Clustering K-Means
kmeans   = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init="auto")
clusters = kmeans.fit_predict(ppmi)

score = silhouette_score(ppmi, clusters, metric="cosine")
print(f"Silhouette score : {score:.3f}")

# Visualisation
plt.figure(figsize=(12, 8))
palette = sns.color_palette("tab10", n_colors=N_CLUSTERS)

for i in range(N_CLUSTERS):
    mask = clusters == i
    plt.scatter(
        data_2d[mask, 0], data_2d[mask, 1],
        s=40, alpha=0.6, color=palette[i], label=f"Cluster {i}"
    )

# Annoter les N mots les plus fréquents de chaque cluster
N_LABELS = 20
for cluster_id in range(N_CLUSTERS):
    indices = np.where(clusters == cluster_id)[0]
    # trier par fréquence décroissante
    indices_sorted = sorted(indices, key=lambda i: word_freq[vocab[i]], reverse=True)
    for i in indices_sorted[:N_LABELS]:
        plt.annotate(
            vocab[i],
            (data_2d[i, 0], data_2d[i, 1]),
            fontsize=7, alpha=0.8,
            xytext=(4, 4), textcoords="offset points"
        )

plt.axhline(0, color="grey", lw=0.8, ls="--")
plt.axvline(0, color="grey", lw=0.8, ls="--")
plt.xlabel(f"PC1 ({reducer.explained_variance_ratio_[0]:.1%})")
plt.ylabel(f"PC2 ({reducer.explained_variance_ratio_[1]:.1%})")
plt.title(f"ACP – COHA | Top {TOP_N_WORDS} mots | {N_CLUSTERS} clusters")
plt.legend(loc="best", fontsize=8)
plt.tight_layout()
plt.savefig("pca_coha.jpeg", format="jpeg", dpi=300, bbox_inches="tight")
plt.show()

print("Graphe sauvegardé → pca_coha.jpeg")
