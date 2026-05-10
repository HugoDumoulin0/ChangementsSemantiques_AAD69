import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from gensim.models import Word2Vec
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid", palette="muted")

# Chargement du model
model = Word2Vec.load("W2V.model")
vocab = list(model.wv.index_to_key)

# Réduction dimensionnelle (PCA avec nomalisation)
# scaler = StandardScaler()
# data_normalized = scaler.fit_transform(model.wv[vocab])
reducer = PCA(n_components=2, random_state=42)
data_2d = reducer.fit_transform(model.wv[vocab])

# Clustering K-Means
n_clusters = 3
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
clusters = kmeans.fit_predict(data_2d)

# Visualisation avec labels
plt.figure(figsize=(14, 10))
palette = sns.color_palette("tab10", n_colors=n_clusters)
for i in range(n_clusters):
    mask = clusters == i
    plt.scatter(
        data_2d[mask, 0],
        data_2d[mask, 1],
        label=f"Groupe {i+1}",
        s=50,
        alpha=0.7,
        color=palette[i]
    )

# Labels — 50 premiers par cluster
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

plt.title(f"Espace sémantique réduit en 2D — K-Means k={n_clusters}")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.legend(title="Clusters", loc="best")
plt.tight_layout()
plt.savefig("visualisation_w2v.png", dpi=150)
plt.show()
print("Figure sauvegardée → visualisation_w2v.png")
