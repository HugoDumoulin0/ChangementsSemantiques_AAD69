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



# Visualisation
plt.figure(figsize=(12, 8))

# Tous les mots dans le même nuage
plt.scatter(
    data_2d[:, 0],
    data_2d[:, 1],
    s=15,
    alpha=0.5
)

# Annoter uniquement les mots les plus fréquents
N_LABELS = 100

for word, _ in word_freq.most_common(N_LABELS):
    if word in word_to_idx:
        idx = word_to_idx[word]
        plt.annotate(
            word,
            (data_2d[idx, 0], data_2d[idx, 1]),
            fontsize=8,
            alpha=0.8,
            xytext=(3, 3),
            textcoords="offset points"
        )

plt.axhline(0, color="grey", lw=0.8, ls="--")
plt.axvline(0, color="grey", lw=0.8, ls="--")

plt.xlabel(f"PC1 ({reducer.explained_variance_ratio_[0]:.1%})")
plt.ylabel(f"PC2 ({reducer.explained_variance_ratio_[1]:.1%})")
plt.title(f"PCA du corpus COHA – Top {TOP_N_WORDS} mots")
plt.tight_layout()

plt.savefig(
    "pca_coha_2d.jpeg",
    format="jpeg",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("Graphe sauvegardé → pca_coha_2d.jpeg")
