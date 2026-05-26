import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import FancyArrowPatch
from matplotlib.lines import Line2D
from collections import Counter, defaultdict
from sklearn.decomposition import PCA
import nltk; nltk.download("stopwords")
from nltk.corpus import stopwords

STOPWORDS = set(stopwords.words("english"))

def remove_stopwords(tokens: list[str]) -> list[str]:
    """Retire les stopwords anglais d'une liste de tokens."""
    return [w for w in tokens if w not in STOPWORDS]

def remove_punct(tokens: list[str]) -> list[str]:
    """Retire la ponctuation et les caractères spéciaux de chaque token.
    - Supprime tout caractère non alphanumérique (garde uniquement a-z et 0-9)
    - Écarte les tokens devenus vides après nettoyage
    """
    cleaned = []
    for w in tokens:
        w = re.sub(r"[^a-z0-9]", "", w)   # supprime tout sauf lettres/chiffres
        if w:                               # ignore les tokens vides
            cleaned.append(w)
    return cleaned

### CONFIG
CORPUS_DIR      = "../COHA_sample/"   # dossier contenant tous les .txt
TARGET_WORD     = "people"            # mot cible
TOP_N_WORDS     = 10000               # limitation du vocabulaire pour la mémoire
TOP_N_NEIGHBORS = 1                   # mots voisins affichés par décennie
WINDOW_SIZE     = 5                   # fenêtre de co-occurrence (en tokens)
MIN_COOC        = 2                   # co-occurrences min pour afficher un voisin

### GROUPEMENT DES FICHIERS PAR DÉCENNIE
FILENAME_RE = re.compile(r"^[^_]+_(\d{4})_\d+\.txt$")

files_by_decade = defaultdict(list)

for fname in os.listdir(CORPUS_DIR):
    m = FILENAME_RE.match(fname)
    if m :
        year   = int(m.group(1))
        decade = (year // 10) * 10
        files_by_decade[decade].append(os.path.join(CORPUS_DIR, fname))
decades_available = sorted(files_by_decade.keys())

### CHARGEMENT DU CORPUS PAR DÉCENNIE
corpus_by_decade = {}

for decade in decades_available:
    lines = []
    for fpath in files_by_decade[decade]:
        try:
            with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read().lower()
                tokens = text.split()
                tokens = remove_punct(tokens)
                tokens = remove_stopwords(tokens)

                lines.append(" ".join(tokens))
        except Exception:
            continue
    corpus_by_decade[decade] = lines
    print(f"  {decade}s : {len(lines)} fichiers chargés")

### VOCABULAIRE GLOBAL
print(f"\nConstruction du vocabulaire global (top {TOP_N_WORDS} mots)...")

word_freq = Counter()
for lines in corpus_by_decade.values():
    for line in lines:
        word_freq.update(line.split())

top_words   = [w for w, _ in word_freq.most_common(TOP_N_WORDS)]
vocab_index = {w: i for i, w in enumerate(top_words)}
size        = len(top_words)

print(f"  Vocabulaire limité à {size} mots les plus fréquents.")

if TARGET_WORD not in vocab_index:
    raise ValueError(f"'{TARGET_WORD}' absent du top-{TOP_N_WORDS} — "
                     f"augmenter TOP_N_WORDS ou changer de mot cible.")


### MATRICE DE CO-OCCURRENCE GLOBALE + PPMI
print("\nConstruction de la matrice de co-occurrence globale (fond PCA)...")

global_matrix = np.zeros((size, size), dtype=np.float32)

for lines in corpus_by_decade.values():
    for line in lines:
        words = line.split()
        for i, target in enumerate(words):
            if target not in vocab_index:
                continue
            start = max(i - WINDOW_SIZE, 0)
            end   = min(i + WINDOW_SIZE + 1, len(words))
            for j in range(start, end):
                if i != j:
                    context = words[j]
                    if context in vocab_index:
                        global_matrix[vocab_index[target], vocab_index[context]] += 1
print(f" Co-occurrences globales totales : {int(global_matrix.sum()):,}")

# Pondération PPMI
print(" Pondération PPMI...")
total_count  = global_matrix.sum()
term_sums    = global_matrix.sum(axis=1)[:, np.newaxis]
context_sums = global_matrix.sum(axis=0)[np.newaxis, :]

with np.errstate(divide="ignore", invalid="ignore"):
    P_tc = global_matrix / total_count
    P_t  = term_sums     / total_count
    P_c  = context_sums  / total_count
    PMI  = np.log2(P_tc / (P_t * P_c))
    PMI[np.isinf(PMI)] = 0
    PMI[np.isnan(PMI)] = 0

ppmi_global = np.maximum(PMI, 0)

print(f"\nMatrice DSM (PPMI) globale ({size}×{size}) :")
print(pd.DataFrame(ppmi_global, index=top_words, columns=top_words))


# PCA GLOBALE
print("\n  PCA globale sur la matrice PPMI...")
reducer = PCA(n_components=2, random_state=42)
data_2d = reducer.fit_transform(ppmi_global)   # shape (size, 2)


### CO-OCCURRENCE DU MOT CIBLE + TRAJECTOIRE PAR DÉCENNIE
print(f"\nConstruction des co-occurrences de « {TARGET_WORD} » par décennie...")

cooc_by_decade   = {}
coords_by_decade = {}

for decade in decades_available:
    corpus = corpus_by_decade[decade]

    cooc = np.zeros(size, dtype=np.float32)
    for line in corpus:
        words = line.split()
        for i, tok in enumerate(words):
            if tok != TARGET_WORD:
                continue
            start = max(i - WINDOW_SIZE, 0)
            end   = min(i + WINDOW_SIZE + 1, len(words))
            for j in range(start, end):
                if i != j:
                    ctx = words[j]
                    if ctx in vocab_index:
                        cooc[vocab_index[ctx]] += 1

    if cooc.sum() == 0:
        print(f"  !!! {decade}s : '{TARGET_WORD}' jamais rencontré, ignoré")
        continue

    cooc_by_decade[decade] = cooc

    # Pondération + projection
    weights      = cooc / (cooc.sum() + 1e-9)
    weighted_vec = (ppmi_global * weights[:, None]).sum(axis=0)   # shape (size,)
    pt           = reducer.transform(weighted_vec.reshape(1, -1))[0]
    coords_by_decade[decade] = pt

    print(f" ✓ {decade}s  ({len(corpus)} fichiers · {int(cooc.sum())} co-occ.)")

active_decades = sorted(coords_by_decade.keys())
if len(active_decades) < 2:
    raise ValueError("Moins de 2 décennies avec des données — trajectoire impossible.")

print(f"\n{len(active_decades)} décennies tracées : "
      f"{active_decades[0]}s --> {active_decades[-1]}s")


### TOP-N VOISINS PAR DÉCENNIE
neighbors = {}
for decade in active_decades:
    cooc   = cooc_by_decade[decade]
    ranked = np.argsort(cooc)[::-1]
    top    = []
    for idx in ranked:
        if cooc[idx] < MIN_COOC:
            break
        if top_words[idx] == TARGET_WORD:
            continue
        wx, wy = data_2d[idx]
        top.append((top_words[idx], wx, wy))
        if len(top) == TOP_N_NEIGHBORS:
            break
    neighbors[decade] = top

### VISUALISATION
fig, ax = plt.subplots(figsize=(14, 9))
ax.set_facecolor("#f8f9fb")
fig.patch.set_facecolor("#eef0f3")

n_dec  = len(active_decades)
cmap   = cm.get_cmap("turbo", n_dec)
colors = {d: cmap(i) for i, d in enumerate(active_decades)}

# Fond : nuage global
ax.scatter(data_2d[:, 0], data_2d[:, 1],
           s=8, c="lightgrey", alpha=0.2, zorder=1)

# Ligne de trajectoire (pointillé)
xs = [coords_by_decade[d][0] for d in active_decades]
ys = [coords_by_decade[d][1] for d in active_decades]
ax.plot(xs, ys, color="dimgrey", linewidth=1.2,
        linestyle="--", alpha=0.4, zorder=2)

# Flèches de la trajectoire
for i in range(n_dec - 1):
    d0, d1 = active_decades[i], active_decades[i + 1]
    x0, y0 = coords_by_decade[d0]
    x1, y1 = coords_by_decade[d1]
    arrow  = FancyArrowPatch(
        (x0, y0), (x1, y1),
        arrowstyle="-|>",
        mutation_scale=20,
        linewidth=2,
        color=colors[d0],
        alpha=0.85,
        zorder=4
    )
    ax.add_patch(arrow)

# Points et labels des décennies
for decade in active_decades:
    x, y = coords_by_decade[decade]
    ax.scatter(x, y, s=150, color=colors[decade],
               edgecolors="white", linewidth=1.8, zorder=5)
    ax.annotate(
        f"{decade}",
        (x, y),
        textcoords="offset points", xytext=(7, 6),
        fontsize=8, fontweight="bold",
        color=colors[decade], zorder=6
    )

# Voisins par décennie
for decade, neighs in neighbors.items():
    c      = colors[decade]
    tx, ty = coords_by_decade[decade]
    for (word, nx, ny) in neighs:
        ax.scatter(nx, ny, s=45, color=c,
                   alpha=0.65, marker="D", zorder=3)
        ax.annotate(
            word,
            (nx, ny),
            textcoords="offset points", xytext=(4, 3),
            fontsize=7, color=c, alpha=0.9, zorder=6
        )
        ax.plot([tx, nx], [ty, ny],
                linestyle=":", linewidth=0.9,
                color=c, alpha=0.35, zorder=2)

# Axes et titre
ax.axhline(0, color="grey", lw=0.7, alpha=0.4)
ax.axvline(0, color="grey", lw=0.7, alpha=0.4)
ax.set_xlabel("PC1", fontsize=11)
ax.set_ylabel("PC2", fontsize=11)
ax.set_title(
    f"Trajectoire sémantique de « {TARGET_WORD} » par décennie\n"
    f"{active_decades[0]}s – {active_decades[-1]}s  ·  "
    f"co-occurrence PPMI (fenêtre={WINDOW_SIZE}) | top-{TOP_N_NEIGHBORS} voisins",
    fontsize=10, fontweight="bold", pad=14
)

# Légende
legend_elements = [
    Line2D([0], [0], marker="o", color="w",
           markerfacecolor=colors[d], markeredgecolor="white",
           markersize=10, label=f"{d}")
    for d in active_decades
]
legend_elements.append(
    Line2D([0], [0], marker="D", color="w",
           markerfacecolor="grey", markersize=8,
           label=f"Voisins (top-{TOP_N_NEIGHBORS})")
)
ax.legend(handles=legend_elements, loc="upper left",
          framealpha=0.9, fontsize=9, title="Décennie")

plt.tight_layout()
output = f"trajectoire_{TARGET_WORD}.jpeg"
plt.savefig(output, format="jpeg", dpi=300, bbox_inches="tight")
print(f"\n✓ Sauvegardé : {output}")
plt.show()
