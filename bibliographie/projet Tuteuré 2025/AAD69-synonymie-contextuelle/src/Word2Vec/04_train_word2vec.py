import pandas as pd
import multiprocessing
from gensim.models import Word2Vec
import ast

df = pd.read_csv("documents_bigrams.csv")
df["tokens_bigrams"] = df["tokens_bigrams"].apply(ast.literal_eval)

sentences = df["tokens_bigrams"].tolist()

w2v_model = Word2Vec(
    sentences,
    vector_size=100,
    window=5,
    min_count=5,
    workers=multiprocessing.cpu_count(),
    sg=1
)

w2v_model.train(sentences, total_examples=len(sentences), epochs=10)
w2v_model.save("word2vec.model")
print("✅ Modèle Word2Vec entraîné et sauvegardé.")
