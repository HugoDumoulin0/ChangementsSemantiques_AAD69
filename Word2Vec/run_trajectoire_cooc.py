import os
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import FancyArrowPatch
from matplotlib.lines import Line2D
from collections import defaultdict, Counter
from sklearn.decomposition import PCA
from scipy.sparse import lil_matrix, csr_matrix
from umap import UMAP
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
CORPUS_DIR      = "../COHA_sample/"    # dossier contenant tous les fichiers .txt
TARGET_WORD     = "people"             # mot cible
TOP_N_WORDS     = 6000                 # vocabulaire
TOP_N_NEIGHBORS = 1                    # mots voisins (PCA)
TOP_N_COOC      = 50                   # voisins par cooc
WINDOW_SIZE     = 5                    # fenêtre de cooc (en tokens)
MIN_COOC        = 2                    # co-occurrences min pour afficher un voisin

MATRICES_DIR    = "ppmi_matrix/"
os.makedirs(MATRICES_DIR, exist_ok=True)
PLOTS_DIR       = f"cards_{TARGET_WORD}/"
os.makedirs(PLOTS_DIR, exist_ok=True)


##### GROUPEMENT DES FICHIERS PAR DÉCENNIE
FILENAME_RE = re.compile(r"^[^_]+_(\d{4})_\d+\.txt$")
files_by_decade = defaultdict(list)

for fname in os.listdir(CORPUS_DIR):
    m = FILENAME_RE.match(fname)
    if m:
        year   = int(m.group(1))
        decade = (year // 10) * 10
        files_by_decade[decade].append(os.path.join(CORPUS_DIR, fname))

decades_available = sorted(files_by_decade.keys())
print(f"Décennies détectées : {decades_available}")


##### CONSTRUCTION DU VOCABULAIRE GLOBAL
print(f"\nConstruction du vocabulaire global...")

word_freq = Counter()
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
        word_freq.update(text.split())
        lines.append(text)
    corpus_by_decade[decade] = lines

top_words   = [w for w, _ in word_freq.most_common(TOP_N_WORDS)]
vocab_index = {w: i for i, w in enumerate(top_words)}
word_to_idx = vocab_index
size        = len(top_words)
print(f"Vocabulaire limité à {size} mots les plus fréquents.")

if TARGET_WORD not in vocab_index:
    raise ValueError(f"'{TARGET_WORD}' est absent du vocabulaire.")

target_idx = vocab_index[TARGET_WORD]


##### MATRICE DE CO-OCCURRENCE GLOBALE
def build_cooc_matrix(file_list_or_texts, vocab_index, window, is_texts=False):
    """
    Construit une matrice de co-occurrence à partir d'une liste de fichiers.
    Retourne une matrice.
    """
    v = len(vocab_index)
    mat = lil_matrix((v, v), dtype=np.float32)

    def process_tokens(tokens):
        ids = [vocab_index[t] for t in tokens if t in vocab_index]
        n   = len(ids)
        for i, wi in enumerate(ids):
            start = max(0, i - window)
            end   = min(n,  i + window + 1)
            for j in range(start, end):
                if j == i:
                    continue
                mat[wi, ids[j]] += 1

    if is_texts:
        for text in file_list_or_texts:
            process_tokens(text.split())
    else:
        for fpath in file_list_or_texts:
            try:
                with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                    process_tokens(f.read().lower().split())
            except Exception:
                continue

    return mat.toarray()


_path_global = os.path.join(MATRICES_DIR, "cooc_global.npy")
if os.path.exists(_path_global):
    print(f"\nChargement de la matrice globale depuis le cache → {_path_global}")
    cooc_global = np.load(_path_global)
    print(f"  Matrice globale : {cooc_global.shape})")
else:
    print("\nConstruction de la matrice de co-occurrence GLOBALE...")
    all_files   = [fp for fps in files_by_decade.values() for fp in fps]
    cooc_global = build_cooc_matrix(all_files, vocab_index, WINDOW_SIZE)
    np.save(_path_global, cooc_global)
    print(f"  Matrice globale : {cooc_global.shape})")
    print(f"  ✓ Sauvegardée → {_path_global}")


##### PCA GLOBALE sur la matrice de co-occurrence
print("\nPCA globale sur la matrice globale...")

# Normalisation L2 ligne par ligne pour stabiliser la PCA
norms = np.linalg.norm(cooc_global, axis=1, keepdims=True)
norms[norms == 0] = 1
data = cooc_global / norms

reducer = PCA(n_components=2, random_state=42)
data_2d = reducer.fit_transform(data)   # (vocab, 2) — utilisé dans make_trajectory_plot
#print(f"  Variance expliquée : PC1={reducer.explained_variance_ratio_[0]:.3f}, PC2={reducer.explained_variance_ratio_[1]:.3f}")


##### MATRICES DE CO-OCCURRENCE PAR DÉCENNIE + PPMI
def ppmi(cooc_mat):
    """
    Positive Pointwise Mutual Information à partir d'une matrice de co-occurrence.
    PPMI(w, c) = max(0,  log2( P(w,c) / (P(w)*P(c)) ))
    """
    total     = cooc_mat.sum()
    if total == 0:
        return cooc_mat.copy()
    p_wc      = cooc_mat / total                          # P(w,c)
    p_w       = cooc_mat.sum(axis=1, keepdims=True) / total  # P(w)
    p_c       = cooc_mat.sum(axis=0, keepdims=True) / total  # P(c)
    denom     = p_w * p_c
    with np.errstate(divide="ignore", invalid="ignore"):
        pmi   = np.where(denom > 0, np.log2(p_wc / denom), 0.0)
    return np.maximum(pmi, 0).astype(np.float32)


print("\nConstruction des matrices par décennie + PPMI...")
cooc_by_decade  = {}    # co-occurrence brute (vecteur ligne du mot cible)
ppmi_by_decade  = {}    # matrice PPMI complète par décennie

for decade in decades_available:
    _path_cooc = os.path.join(MATRICES_DIR, f"cooc_{decade}.npy")
    _path_ppmi = os.path.join(MATRICES_DIR, f"ppmi_{decade}.npy")

    # ── Cache : les deux fichiers existent → chargement direct ──
    if os.path.exists(_path_cooc) and os.path.exists(_path_ppmi):
        cooc_vec = np.load(_path_cooc)
        ppmi_by_decade[decade] = np.load(_path_ppmi)
        if cooc_vec.sum() == 0:
            print(f"  !!! {decade} : '{TARGET_WORD}' absent dans le cache, ignoré")
            continue
        cooc_by_decade[decade] = cooc_vec
        print(f" * {decade}  chargé depuis le cache  "
              f"({int(cooc_vec.sum())} co-occ · PPMI shape={ppmi_by_decade[decade].shape})")
        continue

    # ── Pas de cache → construction ──
    texts = corpus_by_decade.get(decade, [])
    if not texts:
        continue

    mat_dec  = build_cooc_matrix(texts, vocab_index, WINDOW_SIZE, is_texts=True)
    cooc_vec = mat_dec[target_idx]

    if cooc_vec.sum() == 0:
        print(f"  !!! {decade} : '{TARGET_WORD}' jamais rencontré, ignoré")
        continue

    cooc_by_decade[decade] = cooc_vec
    ppmi_by_decade[decade] = ppmi(mat_dec)

    np.save(_path_cooc, cooc_vec)
    np.save(_path_ppmi, ppmi_by_decade[decade])
    print(f"  ✓ {decade}  ({len(files_by_decade[decade])} fichiers · "
          f"{int(cooc_vec.sum())} co-occ · PPMI shape={ppmi_by_decade[decade].shape})"
          f"  → sauvegardé")


##### PROJECTION DU MOT CIBLE DANS L'ESPACE PCA PAR DÉCENNIE
# Pour chaque décennie, on représente TARGET_WORD par une moyenne pondérée des vecteurs PCA de ses contextes (voisins dans cooc_by_decade).
# Cela ancre la projection dans la PCA globale tout en reflétant la distribution contextuelle propre à chaque décennie.
print(f"\nRécupération des coords pour « {TARGET_WORD} »...")

coords_by_decade = {}
for decade in sorted(cooc_by_decade.keys()):
    cooc    = cooc_by_decade[decade]          # vecteur (vocab,)
    weights = cooc / (cooc.sum() + 1e-9)      # distribution de probabilité sur les contextes

    # Moyenne pondérée des lignes de data_2d
    weighted_vec = (data_2d * weights[:, None]).sum(axis=0)   # (2,)
    coords_by_decade[decade] = weighted_vec
    print(f"  ✓ {decade}  →  PC1={weighted_vec[0]:.4f}  PC2={weighted_vec[1]:.4f}")

active_decades = sorted(coords_by_decade.keys())

if len(active_decades) < 2:
    raise ValueError("Moins de 2 décennies avec des données — trajectoire impossible.")

print(f"\n{len(active_decades)} décennies tracées : {active_decades[0]} → {active_decades[-1]}")


##### TOP-N VOISINS PAR DÉCENNIE
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
        top.append({"word": top_words[idx], "pc1": wx, "pc2": wy})
        if len(top) == TOP_N_NEIGHBORS:
            break
    neighbors[decade] = top


##### TOP-N_COOC VOISINS PAR CO-OCCURRENCE
top_cooc_by_decade = {}
for decade in decades_available:
    cooc = cooc_by_decade.get(decade)
    if cooc is None or cooc.sum() == 0:
        continue
    ranked = np.argsort(cooc)[::-1]
    top    = []
    for idx in ranked:
        if top_words[idx] == TARGET_WORD:
            continue
        if cooc[idx] < MIN_COOC:
            break
        top.append({"word": top_words[idx], "cooc": int(cooc[idx])})
        if len(top) == TOP_N_COOC:
            break
    top_cooc_by_decade[decade] = top


##### VISUALISATION
def make_trajectory_plot(ax, active_decades, coords_by_decade, neighbors, data_2d, colors, highlight=None):
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
    ax.set_facecolor("#f8f9fb")
    ax.axis("off")

    if not top_cooc:
        ax.text(0.5, 0.5, "Pas de données", ha="center", va="center",
                fontsize=10, transform=ax.transAxes)
        return

    neighbor_indices = [vocab_index[e["word"]]
                        for e in top_cooc if e["word"] in vocab_index]
    all_indices  = [target_idx] + neighbor_indices
    sub_matrix   = ppmi_dec[all_indices]

    n_samples   = len(all_indices)
    n_neighbors = min(15, n_samples - 1)
    umap_model  = UMAP(n_components=2, n_neighbors=n_neighbors,
                       min_dist=0.3, random_state=42, verbose=False)
    coords_2d   = umap_model.fit_transform(sub_matrix)

    target_xy   = coords_2d[0]
    neighbor_xy = coords_2d[1:]

    counts  = np.array([e["cooc"] for e in top_cooc], dtype=float)
    norm    = (counts - counts.min()) / (counts.max() - counts.min() + 1e-9)
    lw_vals = 0.4 + norm * 3.0
    fs_vals = 6.5 + norm * 4.5
    dot_s   = 20  + norm * 120

    for i in range(len(top_cooc)):
        ax.plot([target_xy[0], neighbor_xy[i, 0]],
                [target_xy[1], neighbor_xy[i, 1]],
                color=color, linewidth=lw_vals[i], alpha=0.25, zorder=1)

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

    ax.scatter(*target_xy, s=320, color=color,
               zorder=6, linewidths=1.5, edgecolors="white")
    ax.text(target_xy[0], target_xy[1], target_word,
            ha="center", va="center",
            fontsize=11, fontweight="bold",
            color="black", zorder=7)

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

for decade in decades_available:
    _path_cooc = os.path.join(MATRICES_DIR, f"cooc_{decade}.npy")
    if os.path.exists(_path_cooc):
        os.remove(_path_cooc)
        #print(f"  Supprimé : {_path_cooc}")
