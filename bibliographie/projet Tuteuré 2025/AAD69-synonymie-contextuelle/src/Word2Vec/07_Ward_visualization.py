import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from google.colab import files

# Charger les données
X_reduced = np.load("X_reduced.npy")
with open("words.pkl", "rb") as f:
    words = pickle.load(f)

# Clustering Ward
n_clusters = 10
ward = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
labels = ward.fit_predict(X_reduced)

# Réduction à 2D avec PCA
pca_2d = PCA(n_components=2, random_state=42)
X_2d = pca_2d.fit_transform(X_reduced)

# Visualisation
plt.figure(figsize=(12, 8))
scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap='tab10', s=20)
plt.title("Clusters Ward visualisés avec PCA (2D)", fontsize=16)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.colorbar(scatter, label="Cluster ID")
plt.grid(True)

# Ajouter quelques mots (facultatif)
for i, word in enumerate(words[:200]):
    plt.annotate(word, (X_2d[i, 0], X_2d[i, 1]), fontsize=8, alpha=0.6)

# Sauvegarde et téléchargement
plt.tight_layout()
plt.savefig("ward_pca_plot.png", dpi=300)
plt.show()
files.download("ward_pca_plot.png")
