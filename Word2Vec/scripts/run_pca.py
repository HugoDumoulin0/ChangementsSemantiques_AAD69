"""
Ce script sert a visualiser uniquement la PCA globale du corpus COHA.
"""

import argparse
import glob
import os
import re
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from nltk.corpus import stopwords
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nltk_setup import ensure_project_nltk_data


DEFAULT_CORPUS_DIR = "../COHA_sample/"
DEFAULT_TOP_N_WORDS = 3500
DEFAULT_WINDOW_SIZE = 5
DEFAULT_OUTPUT_PATH = "pca_coha_2d.jpeg"
DEFAULT_N_LABELS = 100

STOPWORDS = set()


def load_stopwords():
    try:
        return set(stopwords.words("english"))
    except LookupError:
        print("Ressource NLTK 'stopwords' absente, utilisation du fallback sklearn.")
        return set(ENGLISH_STOP_WORDS)


def parse_args():
    parser = argparse.ArgumentParser(description="Genere la PCA globale du corpus COHA.")
    parser.add_argument("--corpus-dir", default=DEFAULT_CORPUS_DIR)
    parser.add_argument("--top-n-words", type=int, default=DEFAULT_TOP_N_WORDS)
    parser.add_argument("--window-size", type=int, default=DEFAULT_WINDOW_SIZE)
    parser.add_argument("--output-path", default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--n-labels", type=int, default=DEFAULT_N_LABELS)
    parser.add_argument(
        "--show",
        dest="show",
        action="store_true",
        default=True,
        help="Affiche la figure matplotlib a la fin.",
    )
    parser.add_argument(
        "--no-show",
        dest="show",
        action="store_false",
        help="N'affiche pas la figure, utile pour l'interface.",
    )
    return parser.parse_args()


def remove_stopwords(tokens):
    return [word for word in tokens if word not in STOPWORDS]


def remove_punct(tokens):
    cleaned = []
    for word in tokens:
        word = re.sub(r"[^a-z0-9]", "", word)
        if word:
            cleaned.append(word)
    return cleaned


def run_pca(
    corpus_dir=DEFAULT_CORPUS_DIR,
    top_n_words=DEFAULT_TOP_N_WORDS,
    window_size=DEFAULT_WINDOW_SIZE,
    output_path=DEFAULT_OUTPUT_PATH,
    n_labels=DEFAULT_N_LABELS,
    show=True,
):
    all_tokens = []
    files = glob.glob(os.path.join(corpus_dir, "**", "*.txt"), recursive=True)
    print(f"{len(files)} fichiers trouves.")

    for fpath in files:
        try:
            with open(fpath, "r", encoding="utf-8", errors="ignore") as file_obj:
                text = file_obj.read().lower()
            tokens = remove_punct(text.split())
            tokens = remove_stopwords(tokens)
            all_tokens.extend(tokens)
        except Exception:
            continue

    print(f"Tokens totaux (apres nettoyage) : {len(all_tokens):,}")

    word_freq = Counter(all_tokens)
    top_words = [word for word, _ in word_freq.most_common(top_n_words)]
    word_to_idx = {word: idx for idx, word in enumerate(top_words)}
    size = len(top_words)

    print(f"Vocabulaire : {size} mots")
    print("Construction de la matrice de co-occurrence...")

    cooc = np.zeros((size, size), dtype=np.float32)

    for i, token in enumerate(all_tokens):
        if token not in word_to_idx:
            continue
        idx = word_to_idx[token]
        start = max(0, i - window_size)
        end = min(len(all_tokens), i + window_size + 1)
        for j in range(start, end):
            if j == i:
                continue
            ctx = all_tokens[j]
            if ctx in word_to_idx:
                cooc[idx, word_to_idx[ctx]] += 1

    print("Calcul PPMI...")
    total = cooc.sum()
    word_sum = cooc.sum(axis=1, keepdims=True)
    ctx_sum = cooc.sum(axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        ppmi = np.log2((cooc * total) / (word_sum * ctx_sum + 1e-9))
    ppmi = np.maximum(ppmi, 0)

    print("PCA...")
    reducer = PCA(n_components=2, random_state=42)
    data_2d = reducer.fit_transform(ppmi)
    print(
        f"Variance expliquee : PC1={reducer.explained_variance_ratio_[0]:.2%}, "
        f"PC2={reducer.explained_variance_ratio_[1]:.2%}"
    )

    plt.figure(figsize=(12, 8))
    plt.scatter(data_2d[:, 0], data_2d[:, 1], s=15, alpha=0.5)

    for word, _ in word_freq.most_common(n_labels):
        if word in word_to_idx:
            idx = word_to_idx[word]
            plt.annotate(
                word,
                (data_2d[idx, 0], data_2d[idx, 1]),
                fontsize=8,
                alpha=0.8,
                xytext=(3, 3),
                textcoords="offset points",
            )

    plt.axhline(0, color="grey", lw=0.8, ls="--")
    plt.axvline(0, color="grey", lw=0.8, ls="--")
    plt.xlabel(f"PC1 ({reducer.explained_variance_ratio_[0]:.1%})")
    plt.ylabel(f"PC2 ({reducer.explained_variance_ratio_[1]:.1%})")
    plt.title(f"PCA du corpus COHA - Top {top_n_words} mots")
    plt.tight_layout()

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    plt.savefig(output_path, format="jpeg", dpi=300, bbox_inches="tight")
    print(f"Graphe sauvegarde -> {output_path}")

    if show:
        plt.show()
    else:
        plt.close()

    return {
        "output_path": output_path,
        "n_files": len(files),
        "n_tokens": len(all_tokens),
        "vocab_size": size,
    }


def main():
    ensure_project_nltk_data(download_missing=True)
    args = parse_args()
    global STOPWORDS
    STOPWORDS = load_stopwords()
    run_pca(
        corpus_dir=args.corpus_dir,
        top_n_words=args.top_n_words,
        window_size=args.window_size,
        output_path=args.output_path,
        n_labels=args.n_labels,
        show=args.show,
    )


if __name__ == "__main__":
    main()
