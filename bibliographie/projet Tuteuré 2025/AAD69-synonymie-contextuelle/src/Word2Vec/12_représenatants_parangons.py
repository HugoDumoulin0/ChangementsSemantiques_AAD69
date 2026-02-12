from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
import numpy as np
import pandas as pd

# Charger les clusters et les mots
df_clusters = pd.read_csv("mots_clusters_ward.csv")  # Doit contenir 'word' et 'cluster'
w2v_model = Word2Vec.load("word2vec.model")

# S'assurer qu'on travaille bien sur les clusters 1, 2 et 3
df_clusters = df_clusters[df_clusters["cluster"].isin([0, 1, 2,3])]

# Filtrer les mots présents dans le modèle Word2Vec
df_clusters = df_clusters[df_clusters["word"].isin(w2v_model.wv.key_to_index)].copy()

# Calcul des centroïdes des clusters
centroides = {
    i: np.mean([w2v_model.wv[word] for word in df_clusters[df_clusters["cluster"] == i]["word"]], axis=0)
    for i in sorted(df_clusters["cluster"].unique())
}

# Stocker les résultats séparément
rep_data = []
par_data = []

for i in sorted(centroides.keys()):
    mots = df_clusters[df_clusters["cluster"] == i]["word"].tolist()
    vecteurs = np.array([w2v_model.wv[word] for word in mots])

    # Représentants spécifiques (proches du centroïde)
    sim_rep = cosine_similarity(vecteurs, [centroides[i]])[:, 0]
    top_representants = [(mots[idx], sim_rep[idx]) for idx in np.argsort(sim_rep)[-5:][::-1]]
    for mot, score in top_representants:
        rep_data.append({"Cluster": i, "Mot": mot, "Similarité au centroïde": score})

    # Parangons discriminants (éloignés des autres centroïdes)
    autres = np.mean([centroides[j] for j in centroides if j != i], axis=0).reshape(1, -1)
    sim_par = cosine_similarity(vecteurs, autres)[:, 0]
    top_parangons = [(mots[idx], sim_par[idx]) for idx in np.argsort(sim_par)[:5]]
    for mot, score in top_parangons:
        par_data.append({"Cluster": i, "Mot": mot, "Similarité autres centroïdes": score})

# Sauvegarder dans deux fichiers CSV
df_rep = pd.DataFrame(rep_data)
df_par = pd.DataFrame(par_data)

df_rep.to_csv("representants_k3.csv", index=False)
df_par.to_csv("parangons_k3.csv", index=False)

print("✅ Fichiers générés :")
print("- representants_k3.csv")
print("- parangons_k3.csv")
