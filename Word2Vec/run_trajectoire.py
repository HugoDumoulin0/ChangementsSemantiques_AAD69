import os
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import FancyArrowPatch
from matplotlib.lines import Line2D
from collections import defaultdict
from gensim.models import Word2Vec
from sklearn.decomposition import PCA


### CONFIG
CORPUS_DIR      = "../COHA_sample/" # dossier contenant tous les .txt
TARGET_WORD     = "people" # mot cible
TOP_N_NEIGHBORS = 1 # mot voisins
WINDOW          = 5 # fenêtre de co-occurrence (en tokens)
MIN_COOC        = 2 # co-occurrences min pour afficher un voisin


### CHARGEMENT MODÈLE
print("Chargement du modèle...")
model       = Word2Vec.load("W2V.model")
vocab       = list(model.wv.key_to_index)
word_to_idx = {w: i for i, w in enumerate(vocab)}
data        = model.wv[vocab]

#PCA GLOBALE
reducer = PCA(n_components=2, random_state=42)
data_2d = reducer.fit_transform(data)

### GROUPEMENT DES FICHIERS PAR PERIODE (DECENNIE)
FILENAME_RE = re.compile(r"^[^_]+_(\d{4})_\d+\.txt$")

files_by_decade = defaultdict(list)

for fname in os.listdir(CORPUS_DIR):
    m = FILENAME_RE.match(fname)
    if m:
        year   = int(m.group(1))
        decade = (year // 10) * 10 #Arrondissement en décennie
        files_by_decade[decade].append(os.path.join(CORPUS_DIR, fname))

decades_available = sorted(files_by_decade.keys())

if TARGET_WORD not in word_to_idx:
    raise ValueError(f"'{TARGET_WORD}' introuvable dans le vocabulaire de Word2Vec.")


### CONSTRUCTION VECTEUR DE CO-OCCURRENCE
def build_cooc_vector(file_list, target, vocab_index, window):
    cooc = np.zeros(len(vocab_index), dtype=np.float32)
    for fpath in file_list:
        try:
            with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                tokens = f.read().lower().split()
        except Exception:
            continue
        for i, tok in enumerate(tokens):
            if tok != target:
                continue
            start = max(0, i - window)
            end   = min(len(tokens), i + window + 1)
            for j in range(start, end):
                if j == i:
                    continue
                ctx = tokens[j]
                if ctx in vocab_index:
                    cooc[vocab_index[ctx]] += 1
    return cooc
print(f"\nConstruction de co-occurrence pour « {TARGET_WORD} »...")

cooc_by_decade   = {}
coords_by_decade = {}

for decade in decades_available:
    files = files_by_decade[decade]
    cooc  = build_cooc_vector(files, TARGET_WORD, word_to_idx, WINDOW)

    if cooc.sum() == 0:
        print(f"  !!! {decade}s : '{TARGET_WORD}' jamais rencontré, ignoré")
        continue

    cooc_by_decade[decade] = cooc

    # Pondération --> projection PCA
    weights      = cooc / (cooc.sum() + 1e-9)
    weighted_vec = (data * weights[:, None]).sum(axis=0)
    pt = reducer.transform(weighted_vec.reshape(1, -1))[0]
    coords_by_decade[decade] = pt
    print(f" ✓ {decade}s  ({len(files)} fichiers · {int(cooc.sum())} co-occ.)")

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
        if vocab[idx] == TARGET_WORD:
            continue
        wx, wy = data_2d[idx]
        top.append((vocab[idx], wx, wy))
        if len(top) == TOP_N_NEIGHBORS:
            break
    neighbors[decade] = top


### VISUALISATION
fig, ax = plt.subplots(figsize=(14, 9))
ax.set_facecolor("#f8f9fb")
fig.patch.set_facecolor("#eef0f3")

n_dec  = len(active_decades)
cmap   = cm.get_cmap("inferno", n_dec)
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
    d0, d1   = active_decades[i], active_decades[i + 1]
    x0, yy0  = coords_by_decade[d0]
    x1, yy1  = coords_by_decade[d1]
    arrow = FancyArrowPatch(
        (x0, yy0), (x1, yy1),
        arrowstyle="-|>",
        mutation_scale=20,
        linewidth=2,
        color=colors[d0],
        alpha=0.85,
        zorder=4
    )
    ax.add_patch(arrow)

# Labels
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
ax.set_xlabel(f"PC1", fontsize=11)
ax.set_ylabel(f"PC2", fontsize=11)
ax.set_title(
    f"Trajectoire sémantique de « {TARGET_WORD} » par décennie\n"
    f"{active_decades[0]}s – {active_decades[-1]}s  ·  "
    f"co-occurrence (fenêtre={WINDOW}) | top-{TOP_N_NEIGHBORS} voisins",
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
print(f"\n Sauvegardé : {output}")
plt.show()
