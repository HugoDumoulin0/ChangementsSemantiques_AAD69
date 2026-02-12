import pandas as pd
import re
import spacy
import ast

nlp = spacy.load("fr_core_news_sm", disable=["ner", "parser"])

def nettoyage_texte(texte):
    texte = re.sub(r"[^a-zA-ZÀ-ÿ']+", ' ', texte)
    doc = nlp(texte)
    return [token.lemma_ for token in doc if not token.is_stop and len(token) > 2]

if __name__ == "__main__":
    df = pd.read_csv("documents_bruts.csv")
    df["tokens"] = df["text"].apply(nettoyage_texte)
    df.to_csv("documents_nettoyes.csv", index=False)
    print("✅ Textes nettoyés et sauvegardés.")
