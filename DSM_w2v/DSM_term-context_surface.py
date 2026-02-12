import pandas as pd
import numpy as np
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity


df = pd.read_csv("simpsons_cleaned.csv")
text_column = "spoken_words"
corpus = df[text_column].dropna().tolist()

window_size = 3
top_n_words = 5000  # Limitation du vocabulaire pour la mémoire


# Construire le vocabulaire limité
word_freq = Counter()
for line in corpus:
    word_freq.update(line.lower().split())

top_words = [w for w, _ in word_freq.most_common(top_n_words)]
vocab_index = {w: i for i, w in enumerate(top_words)}
size = len(top_words)

print(f"Vocabulaire limité à {size} mots les plus fréquents.")


# Calcul des co-occurrences
matrix = np.zeros((size, size), dtype=np.float32)

for line in corpus:
    words = line.lower().split()
    for i, target in enumerate(words):
        if target not in vocab_index:
            continue
        start = max(i - window_size, 0)
        end = min(i + window_size + 1, len(words))
        for j in range(start, end):
            if i != j:
                context = words[j]
                if context in vocab_index:
                    matrix[vocab_index[target], vocab_index[context]] += 1

# 4. Calcul PPMI
total_count = matrix.sum()
term_sums = matrix.sum(axis=1)[:, np.newaxis]
context_sums = matrix.sum(axis=0)[np.newaxis, :]

with np.errstate(divide='ignore', invalid='ignore'):
    P_tc = matrix / total_count
    P_t = term_sums / total_count
    P_c = context_sums / total_count
    PMI = np.log2(P_tc / (P_t * P_c))
    PMI[np.isinf(PMI)] = 0
    PMI[np.isnan(PMI)] = 0

PPMI = np.maximum(PMI, 0)
ppmi_matrix = pd.DataFrame(PPMI, index=top_words, columns=top_words)

similarity = cosine_similarity(ppmi_matrix)
np.fill_diagonal(similarity, 0)

"""top_n = 10
sum_sim = similarity.sum(axis=1)
top_indices = np.argsort(sum_sim)[::-1][:top_n]
selected_words = [top_words[i] for i in top_indices]
ppmi_top10 = ppmi_matrix.loc[selected_words, selected_words]
"""

print("\nMatrice DSM (PPMI)")
print(ppmi_matrix)
