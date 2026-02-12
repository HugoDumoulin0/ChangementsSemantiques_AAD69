import pandas as pd
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

df = pd.read_csv("simpsons_cleaned.csv")
text_column = "spoken_words"
corpus = df[text_column].dropna().tolist()

# Tokenisation simple : convertir chaque ligne en liste de mots
sentences = [[w for w in line.lower().split()] for line in corpus]


model = Word2Vec(
    sentences,
    vector_size=100,    # dimension des vecteurs
    window=3,           # fenêtre contextuelle
    min_count=5,        # ignorer les mots rares
    workers=4,
    sg=1                # skip-gram
)


vocab = list(model.wv.index_to_key)  # mots du modèle
vectors = np.array([model.wv[word] for word in vocab])

print(f"Vocabulaire Word2Vec : {len(vocab)} mots")

similarity_matrix = cosine_similarity(vectors)

# Convertir en DataFrame pour une lecture plus simple
similarity_df = pd.DataFrame(similarity_matrix, index=vocab, columns=vocab)


similarity_df.to_csv("word2vec_similarity_matrix.csv")
print("\nMatrice de similarité Word2Vec sauvegardée")
