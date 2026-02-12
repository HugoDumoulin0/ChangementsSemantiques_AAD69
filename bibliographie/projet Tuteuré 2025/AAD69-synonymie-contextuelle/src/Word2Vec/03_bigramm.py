import pandas as pd
from gensim.models import Phrases
from gensim.models.phrases import Phraser
import ast

df = pd.read_csv("documents_nettoyes.csv")
df["tokens"] = df["tokens"].apply(ast.literal_eval)

phrases = Phrases(df["tokens"], min_count=5, threshold=10)
bigram = Phraser(phrases)

df["tokens_bigrams"] = df["tokens"].apply(lambda x: bigram[x])
df.to_csv("documents_bigrams.csv", index=False)
print("✅ Bigrams créés et sauvegardés.")
