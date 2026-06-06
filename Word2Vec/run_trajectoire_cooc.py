"""
Ce script permet de générer les matrices de coocurrences
du corpus COHA ; calcule les trajectoires de tous les
mots par décennie ; génère différentes visualisations.

1. Construction d'un vocabulaire (limité à 3500 tokens)
2. Constructution des matrices (globale & matrices par décennie)
3. PPMI sur les matrices de décennie
4. Réduction de dimensions avec PCA sur la matrice globale
5. Projection d'un mot (TARGET_WORD) dans l'espace PCA
6. Visualisations de la trajectoire globale du TARGET_WORD

"""

import os
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import FancyArrowPatch
from matplotlib.lines import Line2D
from collections import defaultdict, Counter
from sklearn.decomposition import PCA
from scipy.sparse import lil_matrix
from umap import UMAP
import nltk; nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

STOPWORDS = set(stopwords.words("english"))

nltk.download("wordnet")
lemmatizer = WordNetLemmatizer()

def transform_tokens(tokens: list[str], choice: str) -> list[str]:
    if choice == "form":
        return tokens
    elif choice == "lemma":
        return [lemmatizer.lemmatize(w) for w in tokens]
    else:
        raise ValueError(f"choice doit être 'form' ou 'lemma', reçu : '{choice}'")

def remove_stopwords(tokens: list[str]) -> list[str]:
    return [w for w in tokens if w not in STOPWORDS]

def remove_punct(tokens: list[str]) -> list[str]:
    cleaned = []
    for w in tokens:
        w = re.sub(r"[^a-z0-9]", "", w)
        if w:
            cleaned.append(w)
    return cleaned

def build_cooc_matrix(file_list_or_texts, vocab_index, window, is_texts=False):
    """
    Construit une matrice de co-occurrence à partir d'une liste de fichiers.
    Retourne une matrice.
    """
    v   = len(vocab_index)
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

def ppmi(cooc_mat):
    """
    Positive Pointwise Mutual Information.
    PPMI(w, c) = max(0, log2( P(w,c) / (P(w) * P(c)) ))
    """
    total = cooc_mat.sum()
    if total == 0:
        return cooc_mat.copy()
    p_wc  = cooc_mat / total
    p_w   = cooc_mat.sum(axis=1, keepdims=True) / total
    p_c   = cooc_mat.sum(axis=0, keepdims=True) / total
    denom = p_w * p_c
    with np.errstate(divide="ignore", invalid="ignore"):
        pmi = np.where(denom > 0, np.log2(p_wc / denom), 0.0)
    return np.maximum(pmi, 0).astype(np.float32)

def compute_word_trajectory_length(word_idx, ppmi_by_decade, active_decades, data_2d):
    """
    Calcule la longueur totale de trajectoire d'un mot dans l'espace PCA, via ses vecteurs PPMI pondérés par décennie.
    Retourne (None, {}) si le mot est présent dans moins de 2 décennies.
    """
    #distance eucli entre vecteurs
    coords = {}
    for decade in active_decades:
        ppmi_dec = ppmi_by_decade.get(decade)
        if ppmi_dec is None:
            continue
        cooc_row = ppmi_dec[word_idx]
        if cooc_row.sum() == 0:
            continue
        weights      = cooc_row / (cooc_row.sum() + 1e-9)
        weighted_vec = (data_2d * weights[:, None]).sum(axis=0)
        coords[decade] = weighted_vec

    decades_present = sorted(coords.keys())
    if len(decades_present) < 2:
        return None, {}

    total_length = 0.0
    for i in range(len(decades_present) - 1):
        d0, d1 = decades_present[i], decades_present[i + 1]
        total_length += np.linalg.norm(coords[d1] - coords[d0])

    return total_length, coords

def make_trajectory_plot(ax, active_decades, coords_by_decade,neighbors, data_2d, colors, highlight=None):
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

def make_umap_cooc_plot(ax, decade, top_cooc, color, target_word, ppmi_dec, vocab_index, top_words, target_idx):
    ax.set_facecolor("#f8f9fb")
    ax.axis("off")

    if not top_cooc:
        ax.text(0.5, 0.5, "Pas de données", ha="center", va="center",
                fontsize=10, transform=ax.transAxes)
        return

    neighbor_indices = [vocab_index[e["word"]]
                        for e in top_cooc if e["word"] in vocab_index]
    all_indices = [target_idx] + neighbor_indices
    sub_matrix  = ppmi_dec[all_indices]

    n_samples   = len(all_indices)
    n_neighbors = max(2, min(15, n_samples - 2))   # n_neighbors doit être < n_samples
    if n_samples < 4:                               # trop peu de points pour UMAP
        ax.text(0.5, 0.5, "Données insuffisantes\npour UMAP", ha="center", va="center",
                fontsize=10, transform=ax.transAxes)
        return
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
        nx, ny   = neighbor_xy[i]
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

def compute_word_trajectory_area(word_idx, ppmi_by_decade, active_decades, data_2d):
    """
    Calcule l'aire de la trajectoire d'un mot dans l'espace PCA (aire du polygone convexe
    formé par les positions pondérées par décennie).
    Retourne (None, {}) si le mot est présent dans moins de 3 décennies.
    """
    coords = {}
    for decade in active_decades:
        ppmi_dec = ppmi_by_decade.get(decade)
        if ppmi_dec is None:
            continue
        cooc_row = ppmi_dec[word_idx]
        if cooc_row.sum() == 0:
            continue
        weights      = cooc_row / (cooc_row.sum() + 1e-9)
        weighted_vec = (data_2d * weights[:, None]).sum(axis=0)
        coords[decade] = weighted_vec

    decades_present = sorted(coords.keys())
    if len(decades_present) < 3:
        return None, {}

    pts = np.array([coords[d] for d in decades_present])
    n   = len(pts)
    area = 0.0
    for i in range(n):
        j     = (i + 1) % n
        area += pts[i, 0] * pts[j, 1]
        area -= pts[j, 0] * pts[i, 1]
    area = abs(area) / 2.0

    return area, coords

if __name__ == "__main__" :

##### CONFIGURATION
    CORPUS_DIR      = "../COHA_sample/"   # dossier contenant les fichiers .txt
    TARGET_WORD     = "law"            # mot cible
    TOP_N_WORDS     = 3500                # taille du vocabulaire
    TOP_N_NEIGHBORS = 3                   # mots voisins affichés dans PCA
    TOP_N_COOC      = 20                  # voisins par co-occurrence (UMAP)
    WINDOW_SIZE     = 5                   # fenêtre de co-occurrence (en tokens)
    MIN_COOC        = 2                   # co-occurrences min pour afficher un voisin

    choice = "lemma"

    MATRICES_DIR = "ppmi_matrix/"
    os.makedirs(MATRICES_DIR, exist_ok=True)
    PLOTS_DIR = f"{TARGET_WORD}_viz/cards_cooc_{TARGET_WORD}/"
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
    print("\nConstruction du vocabulaire global...")

    word_freq        = Counter()
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
                    tokens = transform_tokens(tokens, choice)
                    word_freq.update(tokens)
                    lines.append(" ".join(tokens))
            except Exception:
                continue
        corpus_by_decade[decade] = lines

    top_words   = [w for w, _ in word_freq.most_common(TOP_N_WORDS)]
    vocab_index = {w: i for i, w in enumerate(top_words)}
    word_to_idx = vocab_index
    size        = len(top_words)
    print(f"Vocabulaire limité à {size} mots les plus fréquents.")

    if TARGET_WORD not in vocab_index:
        raise ValueError(f"'{TARGET_WORD}' est absent du vocabulaire.")

    target_idx = vocab_index[TARGET_WORD]


##### MATRICE DE CO-OCCURRENCE
    _path_global = os.path.join(MATRICES_DIR, "cooc_global.npy")
    if os.path.exists(_path_global):
        print(f"\nChargement de la matrice globale depuis le cache → {_path_global}")
        cooc_global = np.load(_path_global)
        print(f"  Matrice globale : {cooc_global.shape}")
    else:
        print("\nConstruction de la matrice de co-occurrence GLOBALE...")
        all_files   = [fp for fps in files_by_decade.values() for fp in fps]
        cooc_global = build_cooc_matrix(all_files, vocab_index, WINDOW_SIZE)
        np.save(_path_global, cooc_global)
        print(f"  Matrice globale : {cooc_global.shape}")
        print(f"  ✓ Sauvegardée → {_path_global}")


##### PCA GLOBALE
    print("\nPCA globale sur la matrice globale...")

    norms         = np.linalg.norm(cooc_global, axis=1, keepdims=True)
    norms[norms == 0] = 1
    data          = cooc_global / norms

    reducer = PCA(n_components=2, random_state=42)
    data_2d = reducer.fit_transform(data)   # (vocab, 2)


##### MATRICES PAR DÉCENNIE + PPMI
    print("\nConstruction / chargement des matrices par décennie...")

    cooc_by_decade = {}
    ppmi_by_decade = {}

    for decade in decades_available:
        _path_mat  = os.path.join(MATRICES_DIR, f"cooc_matrix_{decade}.npy")
        _path_ppmi = os.path.join(MATRICES_DIR, f"ppmi_{decade}.npy")

        # Chargement du cache
        if os.path.exists(_path_mat):
            mat_dec = np.load(_path_mat)
            if os.path.exists(_path_ppmi):
                ppmi_mat = np.load(_path_ppmi)
            else:
                ppmi_mat = ppmi(mat_dec)
                np.save(_path_ppmi, ppmi_mat)

            print(f" * {decade} chargé depuis le cache (matrice {mat_dec.shape})")

        # Construction
        else:
            texts = corpus_by_decade.get(decade, [])
            if not texts:
                continue
            mat_dec = build_cooc_matrix(texts,vocab_index,WINDOW_SIZE,is_texts=True)
            np.save(_path_mat, mat_dec)

            ppmi_mat = ppmi(mat_dec)
            np.save(_path_ppmi, ppmi_mat)

            print(f" ✓ {decade} construit ({len(files_by_decade[decade])} fichiers)")

        # Extraction du mot cible
        cooc_vec = mat_dec[target_idx]
        if cooc_vec.sum() == 0:
            print(f" !!! {decade} : '{TARGET_WORD}' absent")
            continue
        cooc_by_decade[decade] = cooc_vec
        ppmi_by_decade[decade] = ppmi_mat
        print(f"    -> {TARGET_WORD} : {int(cooc_vec.sum())} co-occurrences")


##### PROJECTION DU MOT CIBLE
    print(f"\nRécupération des coords pour « {TARGET_WORD} »...")

    coords_by_decade = {}
    for decade in sorted(cooc_by_decade.keys()):
        cooc         = cooc_by_decade[decade]
        weights      = cooc / (cooc.sum() + 1e-9)
        weighted_vec = (data_2d * weights[:, None]).sum(axis=0)
        coords_by_decade[decade] = weighted_vec
        print(f"  ✓ {decade}  →  PC1={weighted_vec[0]:.4f}  PC2={weighted_vec[1]:.4f}")

    active_decades = sorted(coords_by_decade.keys())

    if len(active_decades) < 2:
        raise ValueError("Moins de 2 décennies avec des données — trajectoire impossible.")

    print(f"\n{len(active_decades)} décennies tracées : {active_decades[0]} → {active_decades[-1]}")


##### TRAJECTOIRE DE TOUS LES MOTS DU VOCABULAIRE
    print("\nCalcul des trajectoires pour tous les mots du vocabulaire...")

    all_trajectories      = {}
    all_trajectories_area = {}

    for idx, word in enumerate(top_words):
        length, coords = compute_word_trajectory_length(
            idx, ppmi_by_decade, active_decades, data_2d
        )
        if length is not None:
            all_trajectories[word] = (length, coords)

        area, coords_a = compute_word_trajectory_area(
            idx, ppmi_by_decade, active_decades, data_2d
        )
        if area is not None:
            all_trajectories_area[word] = (area, coords_a)

    ranked_trajectories = sorted(
        all_trajectories.items(), key=lambda x: x[1][0], reverse=True
    )
    ranked_trajectories_area = sorted(
        all_trajectories_area.items(), key=lambda x: x[1][0], reverse=True
    )

##### SAUVEGARDE DES TRAJECTOIRES
    traj_txt_path = f"{TARGET_WORD}_viz/trajectoires_{TARGET_WORD}.txt"
    with open(traj_txt_path, "w", encoding="utf-8") as f_out:

        # Top global : tous les mots classés
        f_out.write(f"=== TRAJECTOIRES SÉMANTIQUES — mot cible : {TARGET_WORD} ===\n")
        f_out.write(f"Décennies : {active_decades[0]}s – {active_decades[-1]}s  |  "
                    f"PPMI (fenêtre={WINDOW_SIZE})  |  vocabulaire top-{TOP_N_WORDS}\n")
        f_out.write("\n")
        f_out.write(f"{'Rang':<6} {'Mot':<22} {'Longueur':>10}\n")
        f_out.write("-" * 42 + "\n")
        for rank, (word, (length, _)) in enumerate(ranked_trajectories[:20], 1):
            f_out.write(f"{rank:<6} {word:<22} {length:>10.4f}\n")

        # Top-10 mots par décennie : mots dont le vecteur pondéré bouge le plus entre la décennie courante et la suivante
        f_out.write("\n\n=== TOP-10 MOTS PAR DÉPLACEMENT DÉCENNAL ===\n")
        f_out.write("(distance euclidienne entre la position pondérée d'une décennie\n"
                    " et celle de la décennie suivante, pour chaque mot du vocabulaire)\n\n")

        f_out.write("\n\n=== TOP-20 MOTS PAR AIRE DE TRAJECTOIRE ===\n")
        f_out.write("(aire du polygone formé par les positions pondérées par décennie)\n\n")
        f_out.write(f"{'Rang':<6} {'Mot':<22} {'Aire':>10}\n")
        f_out.write("-" * 42 + "\n")
        for rank, (word, (area, _)) in enumerate(ranked_trajectories_area[:20], 1):
            f_out.write(f"{rank:<6} {word:<22} {area:>10.4f}\n")

        for i in range(len(active_decades) - 1):
            d0, d1 = active_decades[i], active_decades[i + 1]
            decade_moves = []
            for idx, word in enumerate(top_words):
                _, coords_w = compute_word_trajectory_length(
                    idx, ppmi_by_decade, [d0, d1], data_2d
                )
                if d0 in coords_w and d1 in coords_w:
                    dist = np.linalg.norm(coords_w[d1] - coords_w[d0])
                    decade_moves.append((word, dist))

            decade_moves.sort(key=lambda x: x[1], reverse=True)

            f_out.write(f"--- {d0}s → {d1}s ---\n")
            f_out.write(f"{'Rang':<6} {'Mot':<22} {'Déplacement':>12}\n")
            f_out.write("-" * 42 + "\n")
            for rank, (word, dist) in enumerate(decade_moves[:10], 1):
                f_out.write(f"{rank:<6} {word:<22} {dist:>12.4f}\n")
            f_out.write("\n")

    print(f"\n✓ Trajectoires sauvegardées → {traj_txt_path}")


##### TOP-N VOISINS PAR DÉCENNIE (pour le plot trajectoire)
    neighbors = {}
    for decade in active_decades:
        cooc          = cooc_by_decade[decade]
        target_coords = coords_by_decade[decade] # position PCA du mot cible

        candidates = []
        for idx, word in enumerate(top_words):
            if word == TARGET_WORD:
                continue
            if cooc[idx] < MIN_COOC : # filtre co-occurrence minimale
                continue
            wx, wy = data_2d[idx]
            dist   = np.linalg.norm(np.array([wx, wy]) - target_coords)
            candidates.append({"word": word, "pc1": wx, "pc2": wy, "dist": dist})

        candidates.sort(key=lambda x: x["dist"]) # tri par proximité PCA
        neighbors[decade] = candidates[:TOP_N_NEIGHBORS]


##### TOP-N_COOC VOISINS PAR CO-OCCURRENCE (pour le plot UMAP)
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


##### VISUALISATIONS

# VISUALISATION 1 : trajectoire globale
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


# VISUALISATION 2 : une carte UMAP par décennie
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
        #plt.show()

    print(f"\n✓ Terminé — {len(active_decades)} plots sauvegardés dans {PLOTS_DIR}")


# VISUALISATION 3 : top-K trajectoires les plus longues — un subplot par mot
    TOP_K_TRAJ = 5

    cmap_traj   = cm.get_cmap("tab10", TOP_K_TRAJ)
    traj_colors = [cmap_traj(i) for i in range(TOP_K_TRAJ)]

    fig, axes = plt.subplots(1, TOP_K_TRAJ, figsize=(6 * TOP_K_TRAJ, 8))
    fig.patch.set_facecolor("#eef0f3")

    for rank, (word, (length, coords)) in enumerate(ranked_trajectories[:TOP_K_TRAJ]):
        ax   = axes[rank]
        c    = traj_colors[rank]
        dec_present = sorted(coords.keys())

        ax.set_facecolor("#f8f9fb")
        ax.scatter(data_2d[:, 0], data_2d[:, 1],
                   s=5, c="grey", alpha=0.12, zorder=1)

        xs = [coords[d][0] for d in dec_present]
        ys = [coords[d][1] for d in dec_present]
        ax.plot(xs, ys, color=c, linewidth=1.2, linestyle="--", alpha=0.45, zorder=2)

        for i in range(len(dec_present) - 1):
            d0, d1 = dec_present[i], dec_present[i + 1]
            x0, y0 = coords[d0]
            x1, y1 = coords[d1]
            arrow   = FancyArrowPatch(
                (x0, y0), (x1, y1),
                arrowstyle="-|>", mutation_scale=14,
                linewidth=1.8, color=c, alpha=0.80, zorder=4
            )
            ax.add_patch(arrow)

        ax.scatter(xs, ys, s=70, color=c,
                   edgecolors="white", linewidth=1.3, alpha=0.90, zorder=5)
        for d in dec_present:
            x, y = coords[d]
            ax.annotate(str(d), (x, y),
                        textcoords="offset points", xytext=(5, 5),
                        fontsize=7, fontweight="bold", color=c, zorder=6)

        ax.axhline(0, color="grey", lw=0.6, alpha=0.35)
        ax.axvline(0, color="grey", lw=0.6, alpha=0.35)
        ax.set_xlabel("PC1", fontsize=9)
        ax.set_ylabel("PC2", fontsize=9)
        ax.set_title(f"{rank + 1}. « {word} »\nlong. = {length:.2f}",
                     fontsize=10, fontweight="bold", color=c, pad=8)

    fig.suptitle(
        f"Top-{TOP_K_TRAJ} mots à plus grande trajectoire sémantique\n"
        f"{active_decades[0]}s – {active_decades[-1]}s  ·  PPMI (fenêtre={WINDOW_SIZE}) | PCA",
        fontsize=11, fontweight="bold", y=1.01
    )
    plt.tight_layout()
    top_traj_plot = f"{TARGET_WORD}_viz/top_trajectoires_{TARGET_WORD}_vocab.jpeg"
    plt.savefig(top_traj_plot, format="jpeg", dpi=300, bbox_inches="tight")
    print(f"\n✓ Plot top trajectoires sauvegardé : {top_traj_plot}")
    plt.show()

# VISUALISATION 4 : top-K trajectoires par aire
    TOP_K_AREA = 5

    cmap_area   = cm.get_cmap("tab10", TOP_K_AREA)
    area_colors = [cmap_area(i) for i in range(TOP_K_AREA)]

    fig, axes = plt.subplots(1, TOP_K_AREA, figsize=(6 * TOP_K_AREA, 8))
    fig.patch.set_facecolor("#eef0f3")

    for rank, (word, (area, coords)) in enumerate(ranked_trajectories_area[:TOP_K_AREA]):
        ax          = axes[rank]
        c           = area_colors[rank]
        dec_present = sorted(coords.keys())

        ax.set_facecolor("#f8f9fb")
        ax.scatter(data_2d[:, 0], data_2d[:, 1],
                   s=5, c="grey", alpha=0.12, zorder=1)

        xs  = [coords[d][0] for d in dec_present]
        ys  = [coords[d][1] for d in dec_present]
        xs_closed = xs + [xs[0]]
        ys_closed = ys + [ys[0]]

        ax.fill(xs, ys, color=c, alpha=0.12, zorder=2)
        ax.plot(xs_closed, ys_closed, color=c, linewidth=1.2,
                linestyle="--", alpha=0.45, zorder=3)

        for i in range(len(dec_present) - 1):
            d0, d1 = dec_present[i], dec_present[i + 1]
            x0, y0 = coords[d0]
            x1, y1 = coords[d1]
            arrow   = FancyArrowPatch(
                (x0, y0), (x1, y1),
                arrowstyle="-|>", mutation_scale=14,
                linewidth=1.8, color=c, alpha=0.80, zorder=4
            )
            ax.add_patch(arrow)

        ax.scatter(xs, ys, s=70, color=c,
                   edgecolors="white", linewidth=1.3, alpha=0.90, zorder=5)
        for d in dec_present:
            x, y = coords[d]
            ax.annotate(str(d), (x, y),
                        textcoords="offset points", xytext=(5, 5),
                        fontsize=7, fontweight="bold", color=c, zorder=6)

        ax.axhline(0, color="grey", lw=0.6, alpha=0.35)
        ax.axvline(0, color="grey", lw=0.6, alpha=0.35)
        ax.set_xlabel("PC1", fontsize=9)
        ax.set_ylabel("PC2", fontsize=9)
        ax.set_title(f"{rank + 1}. « {word} »\naire = {area:.4f}",
                     fontsize=10, fontweight="bold", color=c, pad=8)

    fig.suptitle(
        f"Top-{TOP_K_AREA} mots à plus grande aire de trajectoire sémantique\n"
        f"{active_decades[0]}s – {active_decades[-1]}s  ·  PPMI (fenêtre={WINDOW_SIZE}) | PCA",
        fontsize=11, fontweight="bold", y=1.01
    )
    plt.tight_layout()
    area_traj_plot = f"{TARGET_WORD}_viz/top_aires_{TARGET_WORD}_vocab.jpeg"
    plt.savefig(area_traj_plot, format="jpeg", dpi=300, bbox_inches="tight")
    print(f"\n✓ Plot top aires sauvegardé : {area_traj_plot}")
    plt.show()
