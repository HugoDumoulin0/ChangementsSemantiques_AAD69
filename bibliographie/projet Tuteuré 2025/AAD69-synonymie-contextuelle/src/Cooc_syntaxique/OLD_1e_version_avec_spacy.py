import torch
import torch.serialization
from numpy.core.multiarray import _reconstruct
from spacy_conll import init_parser
import pandas as pd

def patch_torch_load():
    """
    Patch la fonction torch.load pour éviter les erreurs de chargement de modèle.
    Je ne me souviens pas précisément quelle est l'errzeur mais la fonction de 
    génération du conll nécessite de patcher torch.load pour éviter les erreurs. 
    Je ne sais pas si c'est lié à la version de torch ou à un autre problème.
    :return: None  
    """
    torch.serialization.add_safe_globals([_reconstruct])

    # Problème sur la fonction torch.load / weights_only=False
    original_torch_load = torch.load

    def patched_torch_load(*args, **kwargs):
        if 'weights_only' not in kwargs:
            kwargs['weights_only'] = False
        return original_torch_load(*args, **kwargs)
    
    torch.load = patched_torch_load

def generate_conll(text, filename):
    """
    Génère un fichier CoNLL à partir d'un texte donné en utilisant Stanza.
    :param text: Texte à analyser
    :param filename: Nom du fichier de sortie (sans extension)
    """
    nlp = init_parser("fr",
                    "stanza",
                    parser_opts={"use_gpu": True, "verbose": False},
                    include_headers=True)
    doc = nlp(text)

    # sauvegarder au format .conll
    conll = doc._.conll_str
    with open(f"{filename}.conll", "w", encoding="utf-8") as f:
        f.write(conll)

    conll_df = doc._.conll_pd
    conll_df.to_csv(f"{filename}.csv", sep="\t", index=False)


def extraire_heads_et_core_arguments(fichier_csv):
    """
    Extrait les prédicats principaux et leurs core arguments à partir d'un fichier CoNLL CSV.
    Chaque entrée aura un ID unique et contiendra la forme du HEAD.

    :param fichier_csv: Chemin du fichier CSV
    :return: Liste des prédicats avec leurs core arguments sous forme de dictionnaire
    """
    # Charger le fichier CSV
    df = pd.read_csv(fichier_csv, sep=None, engine='python')  # Auto-détecter le séparateur

    # Vérifier si les colonnes nécessaires sont présentes
    if not {"HEAD", "DEPREL", "UPOS", "FORM", "ID"}.issubset(df.columns):
        print("Erreur : Colonnes nécessaires ('HEAD', 'DEPREL', 'UPOS', 'FORM', 'ID') manquantes.")
        return []

    # Étape 1 : Identification des prédicats principaux
    # Liste des conditions pour identifier les prédicats principaux
    conditions_heads_predicats = (
        (df["UPOS"] == "VERB") |  # Tous les VERB doivent être pris
        ((df["UPOS"].isin(["ADJ", "NOUN"])) & (df["DEPREL"].isin(["root", "conj", "ccomp", "xcomp", "advcl", "acl:relcl"])))  # Adjectifs/Noms avec copule
    )

    # Liste des ID qui sont des prédicats principaux
    # Vérification des conditions
    heads_predicats = df.loc[conditions_heads_predicats, "ID"].dropna().astype(int).unique().tolist()
    print(heads_predicats)
    
    # Étape 2 : Extraction des core arguments liés aux prédicats
    core_arguments = {}
    id_counter = 1  # Initialisation de l'ID unique

    for _, row in df.iterrows():
        head_id = row["HEAD"]  # Numéro du gouverneur
        dep = row["DEPREL"]  # Relation syntaxique
        form = row["FORM"]  # Mot correspondant

        # Vérifier si le HEAD fait partie des prédicats principaux
        if head_id in heads_predicats:
            # Récupérer la forme du HEAD en utilisant son ID
            head_form = df.loc[df["ID"] == head_id, "FORM"].values
            head_form = head_form[0] if len(head_form) > 0 else None  # Éviter les erreurs si HEAD est absent

            if head_id not in core_arguments:
                core_arguments[head_id] = {
                    "id": id_counter,  # Associer un ID unique
                    "head": head_form,  # Ajouter la **forme du HEAD** dans le dictionnaire
                    "nsubj": [],
                    "obj": [],
                    "iobj": [],
                    "cop": []
                }
                id_counter += 1  # Incrémenter l'ID

            # Ajouter les arguments selon la relation
            if dep == "nsubj":
                core_arguments[head_id]["nsubj"].append(form)
            elif dep == "obj":
                core_arguments[head_id]["obj"].append(form)
            elif dep == "iobj":
                core_arguments[head_id]["iobj"].append(form)
            elif dep == "cop":
                core_arguments[head_id]["cop"].append(form)
            # Ajouter d'autres relations si nécessaire

    return heads_predicats, core_arguments

def main():
    # patch_torch_load()

    text = "Le sol est mouillé parce qu’il a plu cette nuit."
    chemin = "../data/"
    filename = "pluie"
    filename = chemin + filename
    generate_conll(text, filename)

    print(f"Fichiers {filename}.conll et {filename}.csv générés avec succès !")

    fichier_csv = f"{filename}.csv"  # Mets ici le bon chemin de ton fichier
    heads_predicats, core_args = extraire_heads_et_core_arguments(fichier_csv)

    # Affichage des résultats
    print("\n🔹 **Liste des HEADs correspondant aux prédicats principaux :**")
    print(heads_predicats)

    print("\n🔹 **Core Arguments extraits avec ID unique et FORME du HEAD :**")
    for head, args in core_args.items():
        print(f"HEAD '{args['head']}': {args}")


if __name__ == "__main__":
    main()
