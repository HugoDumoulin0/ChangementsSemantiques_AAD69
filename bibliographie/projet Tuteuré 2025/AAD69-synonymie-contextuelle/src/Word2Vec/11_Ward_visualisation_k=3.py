import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering

# Charger les données
X_reduced = np.load("X_reduced.npy")
with open("words.pkl", "rb") as f:
    words = pickle.load(f)

# Clustering Ward
n_clusters = 3
labels = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward').fit_predict(X_reduced)

# Réduction en 2D avec PCA
X_2d = PCA(n_components=2).fit_transform(X_reduced)

# Création du DataFrame
df_clusters = pd.DataFrame({
    "word": words,
    "x": X_2d[:, 0],
    "y": X_2d[:, 1],
    "cluster": labels
})

# Visualisation
plt.figure(figsize=(15, 10))
sns.scatterplot(data=df_clusters, x="x", y="y", hue="cluster", palette="tab10", legend="full", s=50)
plt.title("Visualisation des clusters de mots (Ward + PCA)", fontsize=18)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.grid(True)
plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

# ✅ Sauvegarde en PNG
plt.savefig("clusters_ward_pca_5.png", dpi=300)

# Affichage à l’écran
plt.show()
