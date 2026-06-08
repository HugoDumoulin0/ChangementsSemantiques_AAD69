import pandas as pd
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import time

df = pd.read_csv("simpsons_cleaned.csv")
text_column = "spoken_words"
corpus = df[text_column].dropna().tolist()

# Tokenisation simple : convertir chaque ligne en liste de mots
sentences = [[w for w in line.lower().split()] for line in corpus]


model = Word2Vec(
    sentences,
    vector_size=300,    # dimension des vecteurs
    window=3,           # fenêtre contextuelle
    min_count=5,        # ignorer les mots rares
    workers=4,
    sg=1                # skip-gram
)

epochs=400
start_time = time.time()
# Boucle d'entraînement
for epoch in range(epochs):
    model.train(sentences, 
          total_examples=model.corpus_count, 
          epochs=1, report_delay=1, compute_loss=True)
    loss = model.get_latest_training_loss()
    print(loss)
    # Calculer le temps écoulé
    elapsed_time = time.time() - start_time
    estimated_time = elapsed_time / (epoch + 1) * (epochs - (epoch + 1))
    print(f"Epoch {epoch + 1}/{epochs} - Temps écoulé: {elapsed_time:.2f} secondes, Temps estimé restant: {estimated_time:.2f} secondes")


vocab = list(model.wv.index_to_key)  # mots du modèle
vectors = np.array([model.wv[word] for word in vocab])

model.save("simpson.model")

print(f"Vocabulaire Word2Vec : {len(vocab)} mots")

# similarity_matrix = cosine_similarity(vectors)

# # Convertir en DataFrame pour une lecture plus simple
# similarity_df = pd.DataFrame(similarity_matrix, index=vocab, columns=vocab)


# similarity_df.to_csv("word2vec_similarity_matrix.csv")
# print("\nMatrice de similarité Word2Vec sauvegardée")
