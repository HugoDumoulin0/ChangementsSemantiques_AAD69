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
n_clusters = 5
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
