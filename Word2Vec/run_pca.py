import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import importlib.util
from gensim.models import Word2Vec

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

corpus = "DSM"

if corpus == "DSM" :
    ##### CONSTRUCTION DU VOCABULAIRE GLOBAL
    print(f"\nConstruction du vocabulaire global...")

    word_freq = Counter()
    corpus_by_decade = {}   # {decade: [tokens_str, ...]}  (une entrée = un fichier)

    for decade in decades_available:
        lines = []
        for fpath in files_by_decade[decade]:
            try:
                with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read().lower()
                    tokens = text.split()
                    tokens = remove_punct(tokens) #nettoyage
                    tokens = remove_stopwords(tokens)
                    lines.append(" ".join(tokens))
            except Exception:
                continue
            word_freq.update(text.split())
            lines.append(text)
        corpus_by_decade[decade] = lines

    top_words   = [w for w, _ in word_freq.most_common(TOP_N_WORDS)]
    vocab_index = {w: i for i, w in enumerate(top_words)}   # alias word_to_idx
    word_to_idx = vocab_index                                # nom utilisé plus bas
    size        = len(top_words)
    print(f"Vocabulaire limité à {size} mots les plus fréquents.")

    target_idx = vocab_index[TARGET_WORD]

if corpus == "W2V" :
    sns.set(style="whitegrid", palette="muted")
    model = Word2Vec.load("W2V.model")
    vocab = list(model.wv.key_to_index)

    vocab_freq = {word: model.wv.get_vecattr(word, "count") for word in model.wv.index_to_key}
    sorted_vocab = sorted(vocab_freq.items(), key=lambda item: item[1], reverse=True)
    N = 400
    most_frequent_words = [word for word, freq in sorted_vocab[:N]]
    labels = np.array(most_frequent_words)

    labels = np.array(vocab)
    data = model.wv[vocab]

reducer = PCA(n_components=2, random_state=42)
data_2d = reducer.fit_transform(data)

# Clustering K-Means
n_clusters = 2
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
clusters = kmeans.fit_predict(data_2d)

# Évaluation du clustering
score = silhouette_score(data, clusters, metric="cosine")
print(f"Silhouette score : {score:.3f}")

class_names = None

# Visualisation
plt.figure(figsize=(10, 7))
palette = sns.color_palette("tab10", n_colors=n_clusters)

for i in range(n_clusters):
    plt.scatter(
        data_2d[clusters == i, 0],
        data_2d[clusters == i, 1],
        s=50,
        alpha=0.7,
        color=palette[i]
    )

# Labels
for cluster_id in range(n_clusters):
    indices_cluster = np.where(clusters == cluster_id)[0][:100]
    for i in indices_cluster:
        plt.annotate(
            vocab[i],
            (data_2d[i, 0], data_2d[i, 1]),
            fontsize=7,
            alpha=0.75,
            xytext=(4, 4),
            textcoords="offset points"
        )

plt.axhline(0, color="grey", lw=1)
plt.axvline(0, color="grey", lw=1)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("ACP – individus")
plt.savefig(f"graphe.jpeg", format="jpeg", dpi=300, bbox_inches="tight")
plt.show()
