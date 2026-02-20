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

sns.set(style="whitegrid", palette="muted")

# Choix de DSM Term-term ou DSM term-term avec word2vec
corpus = "DSM"

if corpus == "DSM" :
    spec = importlib.util.spec_from_file_location("DSM_module", "./DSM_term-context_surface.py")
    DSM_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(DSM_module)

    ppmi_matrix = DSM_module.ppmi_matrix  #la matrice PPMI

    data = ppmi_matrix.values


elif corpus == "DSM_w2v" :
    data_path = "word2vec_similarity_matrix.csv"
    data = pd.read_csv(data_path, index_col=0)  # index_col=0 pour utiliser les mots comme index
    print(f"Matrice chargée : {data.shape}")

    # Convertir en numpy array
    X = data.values
    vocab = data.index.values
else:
    raise ValueError("Corpus invalide. Choisir 'DSM' ou 'DSM_w2v'")


#Choix de la visualisation
method = 'PCA'  #PCA ou MDS
if method == 'PCA':
    scaler = StandardScaler() #normalistion "centrer et réduire"
    data_normalized = scaler.fit_transform(data)

    reducer = PCA(n_components=2, random_state=42)
    data_2d = reducer.fit_transform(data_normalized)

elif method == 'MDS':
    reducer = MDS(n_components=2, random_state=42)
    data_2d = reducer.fit_transform(data)
else:
    raise ValueError("Méthode invalide. Choisir 'PCA' ou 'MDS'")



# Clustering K-Means
n_clusters = 5  # nombre de groupes sémantiques
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
clusters = kmeans.fit_predict(data_2d)

# Évaluation du clustering
score = silhouette_score(data, clusters, metric="cosine")
print(f"Silhouette score : {score:.3f}")

class_names= None

labels = ppmi_matrix.index

# Visualisation
plt.figure(figsize=(10, 7))
palette = sns.color_palette("tab10", n_colors=n_clusters)

# eigvals = reducer.explained_variance_
# contrib = pd.DataFrame(
#     data_2d**2 / eigvals,
#     columns=[f"PC{i+1}" for i in range(data_2d.shape[1])]
# )
# contrib["Plan_12"] = contrib["PC1"] + contrib["PC2"]
# contrib.sort_values("Plan_12", ascending=False).head(10)
# top = contrib.sort_values("Plan_12", ascending=False).head(10).index

for i in range(n_clusters):
    # label_name = class_names[i] if class_names is not None else f"Groupe {i+1}"
    plt.scatter(
        data_2d[clusters == i, 0],
        data_2d[clusters == i, 1],
        # label=label_name,
        s=50,
        alpha=0.7,
        color=palette[i]
    )


for i, lab in enumerate(labels):
    # if lab in top:
        plt.annotate(lab, (data_2d[i,0], data_2d[i,1]), fontsize=7)

plt.axhline(0, color="grey", lw=1)
plt.axvline(0, color="grey", lw=1)
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.title("ACP – individus")
plt.show()




top_k = 10
print("\n=== Mots centraux par cluster ===")

for k in range(n_clusters):
    cluster_idx = np.where(clusters == k)[0]
    cluster_sim = X[cluster_idx][:, cluster_idx]

    centroid_sim = cluster_sim.mean(axis=0)
    top_idx = np.argsort(centroid_sim)[::-1][:top_k]

    central_words = vocab[cluster_idx[top_idx]]

    print(f"\nCluster {k} :")
    print(", ".join(central_words))

