import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity


MATRICES_DIR = "ppmi_matrix/"
os.makedirs(MATRICES_DIR, exist_ok=True)

_path_global = os.path.join(MATRICES_DIR, "cooc_global.npy")
print(f"\nChargement de la matrice globale → {_path_global}")
cooc_global = np.load(_path_global)

X = cooc_global
X_used = X

vocab = np.arange(X_used.shape[0]).astype(str)


#Courbe d'inertie (Elbow method)
inerties = []
K_range = range(1, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_used)
    inerties.append(kmeans.inertia_)


plt.figure(figsize=(8, 5))
plt.plot(K_range, inerties, marker='o')
plt.title("Méthode du coude (Inertie)")
plt.xlabel("Nombre de clusters (K)")
plt.ylabel("Inertie (variance intra-cluster)")
plt.xticks(K_range)
plt.grid(True)
plt.tight_layout()
plt.show()


##################################

def plot_dendrogram(model, labels=None, **kwargs):
    """
    Cette fontion est retrouvable sur sklearn.
    """

    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)

    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    dendrogram(linkage_matrix, labels=labels, **kwargs)


#Clustering hiérarchique (Ward)
model = AgglomerativeClustering(
    distance_threshold=0,
    n_clusters=None,
    linkage="ward",
)

model = model.fit(X_used)


plt.figure(figsize=(24, 12))

plt.title("Dendrogramme - Clustering Hiérarchique (Ward)")

plot_dendrogram(
    model,
    labels=vocab,
    leaf_rotation=90,
    leaf_font_size=6
)

plt.xlabel("Observations")
plt.ylabel("Distance")
plt.tight_layout()
plt.show()


##############################################################################

n_clusters = 2

model_ward = AgglomerativeClustering(
    n_clusters=n_clusters,
    linkage="ward",
)

cluster_labels = model_ward.fit_predict(X_used)


pca_2d = PCA(n_components=2, random_state=42)
X_2d = pca_2d.fit_transform(X_used)

plt.figure(figsize=(12, 8))

scatter = plt.scatter(
    X_2d[:, 0],
    X_2d[:, 1],
    c=cluster_labels,
    cmap='tab10',
    s=30,
    alpha=0.8
)

plt.title("Clusters Ward visualisés avec PCA (2D)")
plt.xlabel("PC1")
plt.ylabel("PC2")

plt.colorbar(scatter, label="Cluster ID")
plt.grid(True)
plt.tight_layout()
plt.show()
