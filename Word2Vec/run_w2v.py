import os
import glob
import numpy as np
import pandas as pd
import re
from gensim.models import Word2Vec
import nltk; nltk.download("stopwords")
from nltk.corpus import stopwords

STOPWORDS = set(stopwords.words("english"))
def remove_stopwords(tokens: list[str]) -> list[str]:
    """Retire les stopwords anglais d'une liste de tokens."""
    return [w for w in tokens if w not in STOPWORDS]

def remove_punct(tokens: list[str]) -> list[str]:
    """Retire la ponctuation et les caractères spéciaux de chaque token.
    - Supprime tout caractère non alphanumérique (garde uniquement a-z et 0-9)
    - Écarte les tokens devenus vides après nettoyage
    """
    cleaned = []
    for w in tokens:
        w = re.sub(r"[^a-z0-9]", "", w)   # supprime tout sauf lettres/chiffres
        if w:                               # ignore les tokens vides
            cleaned.append(w)
    return cleaned

# Chargement du corpus COHA
TXT_FOLDER = "../COHA_sample/"

txt_files = glob.glob(os.path.join(TXT_FOLDER, "*.txt")) #Recup tous les .txt
if not txt_files:
    raise FileNotFoundError(f"Aucun fichier .txt trouvé dans : {TXT_FOLDER}")

# print(f"{len(txt_files)} fichier(s) .txt trouvé(s) :")
# for f in txt_files:
#     print(f"  • {os.path.basename(f)}")


sentences = []
total_tokens_before = 0
total_tokens_after  = 0

for path in txt_files:
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            line = line.strip()
            if line:
                tokens = line.lower().split()
                total_tokens_before += len(tokens)

                tokens = remove_punct(tokens)
                tokens = remove_stopwords(tokens)
                total_tokens_after += len(tokens)

                if tokens:               # ignorer les phrases devenues vides
                    sentences.append(tokens)

# print(f"\n{len(sentences)} phrases chargées au total.")
# print(f"Tokens avant filtrage : {total_tokens_before:,}")
# print(f"Tokens après filtrage : {total_tokens_after:,}  "
#       f"({100*(1-total_tokens_after/total_tokens_before):.1f}% retirés)")

model = Word2Vec(
    sentences,
    vector_size=100,   # dimension des vecteurs
    window=3,          # fenêtre contextuelle
    min_count=5,       # ignorer les mots rares
    workers=4,
    sg=1               # skip-gram
)

vocab   = list(model.wv.index_to_key)
vectors = np.array([model.wv[word] for word in vocab])
print(f"\nVocabulaire Word2Vec : {len(vocab)} mots")

model.save("W2V.model") #Sauvegarde du model
