import numpy as np
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
import pickle

# Charger le modèle Word2Vec
model = Word2Vec.load("word2vec.model")
words = list(model.wv.index_to_key)
X = np.array([model.wv[word] for word in words])

# Réduction de dimension à 20 avec PCA
pca = PCA(n_components=20, random_state=42)
X_reduced = pca.fit_transform(X)

# Sauvegarde
np.save("X_reduced.npy", X_reduced)
with open("words.pkl", "wb") as f:
    pickle.dump(words, f)

print(f"✅ Réduction terminée. Vecteurs réduits sauvegardés. ({X_reduced.shape})")
import numpy as np

# S'assurer que le modèle est déjà chargé ou entraîné
# Exemple : w2v_model = Word2Vec.load("model/word2vec.model")

print(f"✅ {len(words)} mots extraits pour le clustering.")
print("🔍 Exemples de mots :", words[:5])
