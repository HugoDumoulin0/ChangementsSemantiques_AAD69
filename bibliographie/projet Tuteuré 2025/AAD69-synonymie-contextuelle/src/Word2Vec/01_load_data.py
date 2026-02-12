# 01_load_data.py

import os
import pandas as pd

def load_parlamint_data(data_dir):
    documents = []
    labels = []

    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".txt"):
                path = os.path.join(root, file)
                with open(path, 'r', encoding='utf-8') as f:
                    text = f.read().strip()
                    documents.append(text)
                    labels.append(os.path.basename(root))  # Dossier comme étiquette

    df_texts = pd.DataFrame({
        "label": labels,
        "text": documents
    })

    print(f"✅ {len(df_texts)} documents chargés.")
    return df_texts


if __name__ == "__main__":
    DATA_DIR = "../data/Parlamint2018_raw"  # Remplacer si besoin
    df = load_parlamint_data(DATA_DIR)
    df.to_csv("documents_bruts.csv", index=False)
    print("✅ Fichier 'documents_bruts.csv' sauvegardé.")
