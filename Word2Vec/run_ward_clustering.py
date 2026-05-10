import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity



model = Word2Vec.load("W2V.model")
vocab = list(model.wv.index_to_key)

# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(model.wv[vocab])

X_scaled = model.wv[vocab]

#Courbe d'inertie (Elbow method)
inerties = []
K_range = range(1, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inerties.append(kmeans.inertia_)


#Affichage de la courbe d'inertie
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

#Fonction pour dendrogramme (qu'on peut retrouver sur le site dsklearn)
def plot_dendrogram(model, labels=None, **kwargs):
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
model = model.fit(X_scaled)


#Affichage du dendrogramme
plt.figure(figsize=(24, 12))

plt.title("Dendrogramme - Clustering Hiérarchique (Ward)")
plot_dendrogram(
    model,
    labels=vocab,
    leaf_rotation=90,
    leaf_font_size=8
)

plt.xlabel("Observations")
plt.ylabel("Distance (variance intra-cluster)")
plt.tight_layout()
plt.show()


##############################################################################
#Script 07 Ward Visualisation

n_clusters = 2
model_ward = AgglomerativeClustering(
    distance_threshold= None,
    n_clusters = n_clusters,
    linkage="ward",
)
cluster_labels = model_ward.fit_predict(X_scaled)


# Réduction à 2D avec PCA
pca_2d = PCA(n_components=2, random_state=42)
X_2d = pca_2d.fit_transform(X_scaled)

#print(f"Variance expliquée par PC1 et PC2 : {pca_2d.explained_variance_ratio_}")

plt.figure(figsize=(12, 8))

scatter = plt.scatter(
    X_2d[:, 0],
    X_2d[:, 1],
    c=cluster_labels,
    cmap='tab10',
    s=30,
    alpha=0.8
)

plt.title("Clusters Ward visualisés avec PCA (2D)", fontsize=1)
plt.xlabel("PC1")
plt.ylabel("PC2")

plt.colorbar(scatter, label="Cluster ID")
plt.grid(True)
plt.tight_layout()
plt.show()

###########################################################################

# top_k = 20
# print("Top 20 mots par cluster")
#
# for k in range(n_clusters):
#     cluster_idx = np.where(cluster_labels == k)[0]
#     cluster_vectors = X_scaled[cluster_idx]
#
#     centroid = cluster_vectors.mean(axis=0).reshape(1, -1)
#
#     # Similarité cosinus au centroïde
#     similarities = cosine_similarity(cluster_vectors, centroid).flatten()
#
#     # Indices triés par similarité décroissante
#     top_indices = cluster_idx[np.argsort(similarities)[::-1][:top_k]]
#
#     top_words = vocab[top_indices]
#
#     print(f"\nCluster {k} ({len(cluster_idx)} mots) :")
#     print(", ".join(top_words))
