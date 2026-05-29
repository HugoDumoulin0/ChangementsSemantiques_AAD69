import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from umap import UMAP
from matplotlib.patches import FancyArrowPatch
from matplotlib.lines import Line2D
from collections import Counter, defaultdict
from sklearn.decomposition import PCA
import nltk; nltk.download("stopwords")
from nltk.corpus import stopwords

STOPWORDS = set(stopwords.words("english"))

def remove_stopwords(tokens: list[str]) -> list[str]:
    return [w for w in tokens if w not in STOPWORDS]

def remove_punct(tokens: list[str]) -> list[str]:
    cleaned = []
    for w in tokens:
        w = re.sub(r"[^a-z0-9]", "", w)
        if w:
            cleaned.append(w)
    return cleaned

### CONFIG
CORPUS_DIR      = "../COHA_sample/"
TARGET_WORD     = "people"
TOP_N_WORDS     = 6000
TOP_N_NEIGHBORS = 3
TOP_N_COOC      = 50        # voisins par co-occurrence brute à afficher
WINDOW_SIZE     = 5
MIN_COOC        = 2

# Dossier pour sauvegarder/charger les matrices PPMI par décennie
MATRICES_DIR    = "ppmi_matrix/"
os.makedirs(MATRICES_DIR, exist_ok=True)
# Dossier pour sauvegarder les visualisations "cards"
PLOTS_DIR       = f"cards_{TARGET_WORD}/"
os.makedirs(PLOTS_DIR, exist_ok=True)


### GROUPEMENT DES FICHIERS PAR DÉCENNIE
FILENAME_RE = re.compile(r"^[^_]+_(\d{4})_\d+\.txt$")

files_by_decade = defaultdict(list)
for fname in os.listdir(CORPUS_DIR):
    m = FILENAME_RE.match(fname)
    if m:
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
                text   = f.read().lower()
                tokens = text.split()
                tokens = remove_punct(tokens)
                tokens = remove_stopwords(tokens)
                lines.append(" ".join(tokens))
        except Exception:
            continue
    corpus_by_decade[decade] = lines
    print(f" {decade} : {len(lines)} fichiers chargés")

### VOCABULAIRE GLOBAL
print(f"\nConstruction du vocabulaire global (top {TOP_N_WORDS} mots)...")

word_freq = Counter()
for lines in corpus_by_decade.values():
    for line in lines:
        word_freq.update(line.split())

top_words   = [w for w, _ in word_freq.most_common(TOP_N_WORDS)]
vocab_index = {w: i for i, w in enumerate(top_words)}
size        = len(top_words)
print(f"Vocabulaire limité à {size} mots les plus fréquents.")

if TARGET_WORD not in vocab_index:
    raise ValueError(f"'{TARGET_WORD}' est absent.")

target_idx = vocab_index[TARGET_WORD]

#Sauvegarde du vocabulaire
# vocab_path = os.path.join(MATRICES_DIR, "vocab.txt")
# if not os.path.exists(vocab_path):
#     with open(vocab_path, "w", encoding="utf-8") as f:
#         f.write("\n".join(top_words))
#     print(f"Vocabulaire sauvegardé : {vocab_path}")
# else:
#     print(f"Vocabulaire déjà présent : {vocab_path}")


### MATRICES DE CO-OCCURRENCE PPMI PAR DÉCENNIE
# Chaque matrice est sauvegardée en .npy, si elle existe déjà, on le charge directement
print("\nConstruction / chargement des matrices PPMI par décennie...")

ppmi_by_decade = {}
cooc_by_decade = {}

for decade in decades_available:
    matrix_path = os.path.join(MATRICES_DIR, f"ppmi_{decade}.npy")
    cooc_path   = os.path.join(MATRICES_DIR, f"cooc_{TARGET_WORD}_{decade}.npy")

    if os.path.exists(matrix_path) and os.path.exists(cooc_path):
        ppmi_by_decade[decade] = np.load(matrix_path)
        cooc_by_decade[decade] = np.load(cooc_path)
        print(f"  ✓ {decade}s : matrice + cooc chargées depuis le cache")
        continue

    cooc_matrix = np.zeros((size, size), dtype=np.float32)
    corpus      = corpus_by_decade[decade]

    for line in corpus:
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
                        cooc_matrix[vocab_index[target], vocab_index[context]] += 1

    print(f"  {decade}s : {int(cooc_matrix.sum()):,} co-occurrences brutes")

    target_cooc = cooc_matrix[target_idx].copy()
    cooc_by_decade[decade] = target_cooc
    np.save(cooc_path, target_cooc)

    total_count  = cooc_matrix.sum()
    term_sums    = cooc_matrix.sum(axis=1)[:, np.newaxis]
    context_sums = cooc_matrix.sum(axis=0)[np.newaxis, :]

    with np.errstate(divide="ignore", invalid="ignore"):
        P_tc = cooc_matrix / total_count
        P_t  = term_sums   / total_count
        P_c  = context_sums / total_count
        PMI  = np.log2(P_tc / (P_t * P_c))
        PMI[np.isinf(PMI)] = 0
        PMI[np.isnan(PMI)]  = 0

    ppmi = np.maximum(PMI, 0)
    ppmi_by_decade[decade] = ppmi

    np.save(matrix_path, ppmi)
    print(f" ✓ {decade} : matrice PPMI sauvegardée → {matrix_path}")

### MATRICE GLOBALE (somme des matrices par décennie)

if os.path.exists(global_matrix_path):
    ppmi_global = np.load(global_matrix_path)
    print(f"\n✓ Matrice globale chargée depuis {global_matrix_path}")
else:
    print("\nConcaténation des matrices par décennie → matrice globale")
    ppmi_global = sum(ppmi_by_decade[d] for d in decades_available)
    np.save(global_matrix_path, ppmi_global)
    print(f"✓ Matrice globale sauvegardée → {global_matrix_path}")

print(f"\nMatrice DSM pondérée globale ({size}×{size}) :")
print(pd.DataFrame(ppmi_global, index=top_words, columns=top_words))


### PCA GLOBALE
print("\n  PCA globale sur la matrice PPMI...")
reducer = PCA(n_components=2, random_state=42)
data_2d = reducer.fit_transform(ppmi_global)


### TRAJECTOIRE DU TARGET_WORD
print(f"\nProjection de « {TARGET_WORD} » par décennie (depuis les PPMI)...")

coords_by_decade = {}

for decade in decades_available:
    ppmi_dec = ppmi_by_decade[decade]

    # Vecteur du mot cible dans cette décennie
    target_vec = ppmi_dec[target_idx]

    if target_vec.sum() == 0:
        print(f" !! {decade} : Pas de co-occurrences trouvé")
        continue

    # Projection dans l'espace PCA global
    pt = reducer.transform(target_vec.reshape(1, -1))[0]
    coords_by_decade[decade] = pt
    print(f"  ✓ {decade}  →  PC1={pt[0]:.4f}  PC2={pt[1]:.4f}")

active_decades = sorted(coords_by_decade.keys())
if len(active_decades) < 2:
    raise ValueError("Moins de 2 décennies avec des données — trajectoire impossible.")

print(f"\n{len(active_decades)} décennies tracées : "
      f"{active_decades[0]}s --> {active_decades[-1]}s")


### TOP-N VOISINS PAR DÉCENNIE
# Distance euclidienne dans l'espace PCA entre le point du TARGET_WORD et chaque mot du vocabulaire.
neighbors = {}

for decade in active_decades:
    dec_pt  = coords_by_decade[decade]
    ppmi_dec = ppmi_by_decade[decade]
    decade_2d = reducer.transform(ppmi_dec) # Projection dans l'espace PCA global
    dists  = np.linalg.norm(decade_2d - dec_pt, axis=1 ) # Distance euclidienne
    ranked = np.argsort(dists)

    top = []
    for idx in ranked:
        if top_words[idx] == TARGET_WORD:
            continue
        cooc_val = int(ppmi_dec[target_idx, idx] > 0)
        top.append({
            "word": top_words[idx],
            "dist": float(dists[idx]),
            "pc1":  float(decade_2d[idx, 0]),
            "pc2":  float(decade_2d[idx, 1]),
        })
        if len(top) == TOP_N_NEIGHBORS:
            break
    neighbors[decade] = top

### VISUALISATION
def make_trajectory_plot(ax, active_decades, coords_by_decade,
                         neighbors, data_2d, colors, highlight=None):
    n_dec = len(active_decades)
    ax.set_facecolor("#f8f9fb")

    ax.scatter(data_2d[:, 0], data_2d[:, 1],
               s=8, c="grey", alpha=0.2, zorder=1)

    xs = [coords_by_decade[d][0] for d in active_decades]
    ys = [coords_by_decade[d][1] for d in active_decades]
    ax.plot(xs, ys, color="dimgrey", linewidth=1.2,
            linestyle="--", alpha=0.4, zorder=2)

    for i in range(n_dec - 1):
        d0, d1 = active_decades[i], active_decades[i + 1]
        x0, y0 = coords_by_decade[d0]
        x1, y1 = coords_by_decade[d1]
        alpha  = 0.85 if highlight is None else (0.9 if d0 == highlight else 0.2)
        arrow  = FancyArrowPatch(
            (x0, y0), (x1, y1),
            arrowstyle="-|>", mutation_scale=20,
            linewidth=2, color=colors[d0],
            alpha=alpha, zorder=4
        )
        ax.add_patch(arrow)

    for decade in active_decades:
        x, y   = coords_by_decade[decade]
        is_hl  = (highlight is None or decade == highlight)
        size_s = 200 if (decade == highlight) else 150
        alpha  = 1.0 if is_hl else 0.25
        ax.scatter(x, y, s=size_s, color=colors[decade],
                   edgecolors="white", linewidth=1.8,
                   alpha=alpha, zorder=5)
        ax.annotate(
            f"{decade}", (x, y),
            textcoords="offset points", xytext=(7, 6),
            fontsize=8, fontweight="bold",
            color=colors[decade], alpha=alpha, zorder=6
        )

    for decade in active_decades:
        if highlight is not None and decade != highlight:
            continue
        c      = colors[decade]
        tx, ty = coords_by_decade[decade]
        for entry in neighbors[decade]:
            nx, ny = entry["pc1"], entry["pc2"]
            ax.scatter(nx, ny, s=45, color=c,
                       alpha=0.65, marker="D", zorder=3)
            ax.annotate(entry["word"], (nx, ny),
                        textcoords="offset points", xytext=(4, 3),
                        fontsize=7, color=c, alpha=0.9, zorder=6)
            ax.plot([tx, nx], [ty, ny],
                    linestyle=":", linewidth=0.9,
                    color=c, alpha=0.35, zorder=2)

    ax.axhline(0, color="grey", lw=0.7, alpha=0.4)
    ax.axvline(0, color="grey", lw=0.7, alpha=0.4)
    ax.set_xlabel("PC1", fontsize=10)
    ax.set_ylabel("PC2", fontsize=10)


def make_umap_cooc_plot(ax, decade, top_cooc, color, target_word,
                        ppmi_dec, vocab_index, top_words, target_idx):
    """
    Visualisation UMAP des TOP_N_COOC voisins + TARGET_WORD.

    Entrées UMAP : vecteurs PPMI de la décennie pour chaque mot du sous-ensemble
    (TARGET_WORD + ses TOP_N_COOC voisins). UMAP projette ces vecteurs en 2D en
    préservant la topologie locale → les mots sémantiquement proches se regroupent.

    Encodages visuels :
    - Position        : coordonnées UMAP (espace sémantique local de la décennie)
    - Taille du point : co-occurrence brute avec TARGET_WORD
    - Épaisseur ligne : co-occurrence brute (TARGET_WORD → voisin)
    - Taille du label : co-occurrence brute
    """

    ax.set_facecolor("#f8f9fb")
    ax.axis("off")

    if not top_cooc:
        ax.text(0.5, 0.5, "Pas de données", ha="center", va="center",
                fontsize=10, transform=ax.transAxes)
        return

    # ── Sous-ensemble : TARGET_WORD (position 0) + voisins ──
    neighbor_indices = [vocab_index[e["word"]]
                        for e in top_cooc if e["word"] in vocab_index]
    all_indices  = [target_idx] + neighbor_indices
    sub_matrix   = ppmi_dec[all_indices]              # shape (1 + N, vocab_size)

    # ── Projection UMAP ──
    n_samples   = len(all_indices)
    n_neighbors = min(15, n_samples - 1)
    umap_model  = UMAP(n_components=2, n_neighbors=n_neighbors,
                       min_dist=0.3, random_state=42, verbose=False)
    coords_2d   = umap_model.fit_transform(sub_matrix)

    target_xy   = coords_2d[0]
    neighbor_xy = coords_2d[1:]

    # ── Normalisation des co-occurrences ──
    counts  = np.array([e["cooc"] for e in top_cooc], dtype=float)
    norm    = (counts - counts.min()) / (counts.max() - counts.min() + 1e-9)
    lw_vals = 0.4 + norm * 3.0     # épaisseur ligne  [0.4 – 3.4]
    fs_vals = 6.5 + norm * 4.5     # taille label     [6.5 – 11.0]
    dot_s   = 20  + norm * 120     # taille point     [20  – 140]

    # ── Lignes TARGET_WORD → voisins ──
    for i in range(len(top_cooc)):
        ax.plot([target_xy[0], neighbor_xy[i, 0]],
                [target_xy[1], neighbor_xy[i, 1]],
                color=color, linewidth=lw_vals[i], alpha=0.25, zorder=1)

    # ── Points + labels des voisins ──
    for i, entry in enumerate(top_cooc):
        nx, ny = neighbor_xy[i]
        ax.scatter(nx, ny, s=dot_s[i], color=color,
                   alpha=0.70, zorder=3, linewidths=0)

        ha       = "left" if nx >= target_xy[0] else "right"
        offset_x = 0.02 * (1 if ha == "left" else -1)
        ax.text(nx + offset_x, ny, entry["word"],
                fontsize=fs_vals[i], color=color,
                ha=ha, va="center",
                fontweight="bold" if i < 5 else "normal",
                alpha=0.92, zorder=4)

    # ── Mot central ──
    ax.scatter(*target_xy, s=320, color=color,
               zorder=6, linewidths=1.5, edgecolors="white")
    ax.text(target_xy[0], target_xy[1], target_word,
            ha="center", va="center",
            fontsize=11, fontweight="bold",
            color="black", zorder=7)

    # ── Marges ──
    all_x = np.append(neighbor_xy[:, 0], target_xy[0])
    all_y = np.append(neighbor_xy[:, 1], target_xy[1])
    pad_x = (all_x.max() - all_x.min()) * 0.18 + 0.3
    pad_y = (all_y.max() - all_y.min()) * 0.18 + 0.3
    ax.set_xlim(all_x.min() - pad_x, all_x.max() + pad_x)
    ax.set_ylim(all_y.min() - pad_y, all_y.max() + pad_y)


### VISUALISATION 1 : trajectoire globale
n_dec  = len(active_decades)
cmap   = cm.get_cmap("turbo", n_dec)
colors = {d: cmap(i) for i, d in enumerate(active_decades)}

fig, ax = plt.subplots(figsize=(14, 9))
fig.patch.set_facecolor("#eef0f3")
make_trajectory_plot(ax, active_decades, coords_by_decade,
                     neighbors, data_2d, colors)
ax.set_title(
    f"Trajectoire sémantique de « {TARGET_WORD} » — toutes décennies\n"
    f"{active_decades[0]}s – {active_decades[-1]}s  ·  "
    f"PPMI (fenêtre={WINDOW_SIZE}) | top-{TOP_N_NEIGHBORS} voisins PCA",
    fontsize=10, fontweight="bold", pad=14
)
legend_elements = [
    Line2D([0], [0], marker="o", color="w",
           markerfacecolor=colors[d], markeredgecolor="white",
           markersize=10, label=str(d))
    for d in active_decades
] + [Line2D([0], [0], marker="D", color="w",
            markerfacecolor="grey", markersize=8,
            label=f"Top-{TOP_N_NEIGHBORS} PCA")]
ax.legend(handles=legend_elements, loc="upper left",
          framealpha=0.9, fontsize=9, title="Décennie")
plt.tight_layout()
global_plot = f"trajectoire_cooc_{TARGET_WORD}.jpeg"
plt.savefig(global_plot, format="jpeg", dpi=300, bbox_inches="tight")
print(f"\n✓ Trajectoire globale sauvegardée : {global_plot}")
plt.show()



### TOP-N_COOC VOISINS PAR CO-OCCURRENCE

top_cooc_by_decade = {}

for decade in decades_available:
    cooc = cooc_by_decade.get(decade)
    if cooc is None or cooc.sum() == 0:
        continue

    ranked = np.argsort(cooc)[::-1]
    top = []
    for idx in ranked:
        if top_words[idx] == TARGET_WORD:
            continue
        if cooc[idx] < MIN_COOC:
            break
        top.append({"word": top_words[idx], "cooc": int(cooc[idx])})
        if len(top) == TOP_N_COOC:
            break
    top_cooc_by_decade[decade] = top

### VISUALISATION 2 : une figure par décennie
print(f"\nGénération des cards par décennie → {PLOTS_DIR}")

for decade in active_decades:
    fig, ax_radial = plt.subplots(figsize=(10, 10))
    fig.patch.set_facecolor("#eef0f3")

    top_cooc = top_cooc_by_decade.get(decade, [])
    make_umap_cooc_plot(ax_radial, decade, top_cooc,
                        color=colors[decade], target_word=TARGET_WORD,
                        ppmi_dec=ppmi_by_decade[decade],
                        vocab_index=vocab_index, top_words=top_words,
                        target_idx=target_idx)
    ax_radial.set_title(
        f"Top-{TOP_N_COOC} voisins co-occ · {decade} [UMAP]\n"
        f"taille du point / label ∝ co-occ  ·  position = espace PPMI local",
        fontsize=10, fontweight="bold", pad=10
    )

    plt.tight_layout()
    out_path = os.path.join(PLOTS_DIR, f"{decade}_{TARGET_WORD}.jpeg")
    plt.savefig(out_path, format="jpeg", dpi=200, bbox_inches="tight")
    print(f"  ✓ {out_path}")
    plt.show()

print(f"\n✓ Terminé — {len(active_decades)} plots sauvegardés dans {PLOTS_DIR}")
