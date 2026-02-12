# 06_cluster_ward.py
import numpy as np
import pickle
from sklearn.cluster import AgglomerativeClustering

# Charger les données réduites et les mots
X_reduced = np.load("X_reduced.npy")
with open("words.pkl", "rb") as f:
    words = pickle.load(f)

# Clustering avec méthode Ward
clusterer = AgglomerativeClustering(n_clusters=10, linkage='ward')
labels = clusterer.fit_predict(X_reduced)

# Associer les mots à leur cluster
clusters = {i: [] for i in range(10)}
for word, label in zip(words, labels):
    clusters[label].append(word)

# Afficher les clusters
for cluster_id, word_list in clusters.items():
    print(f"\n🔹 Cluster {cluster_id} ({len(word_list)} mots):")
    print(", ".join(word_list[:20]))  # afficher les 20 premiers mots
