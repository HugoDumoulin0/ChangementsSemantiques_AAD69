#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sur la base d'un script créé par @hugodumoulin on Wed Apr 23 14:21:23 
Modifié le 06/04/2025 by Lucile Bessac

Script pour extraire les énoncés élémentaires d'un corpus ParlaMint 2018
Pour un corpus textuel donné, ce script génère un fichier CoNLL
et indexe les énoncés élémentaires (EE) en fonction d'un motif syntaxique spécifique à l'aide de GrewPy.
Pour un Corpus (object GrewPy), une request et un parametre donné, donne les mots qu'il y a derrière.
Grâce au numéro de node, on peut récupérer le mot correspondant dans le corpus.
Grâce au module features.

@author: hugodumoulin
Modified on 06/06/2025 by Lucile Bessac
"""

import time
import grewpy
from pathlib import Path
import torch
import torch.serialization
from numpy.core.multiarray import _reconstruct
from spacy_conll import init_parser

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


def indexe_enonces_elem(corpus, liste_match, param):
    start_indexation = time.time()
    print("Setup config SUD")

    grewpy.set_config("sud")
    print(f"Config SUD setup en {time.time() - start_indexation:.2f} secondes")

    # print("LISTE DES MATCHS\n")
    # for element in liste_match:
    #     print(element)

    liste_des_enonces_elem = []
    n = 0

    def get_feature(node_label): # pour éviter les erreurs si le label n'existe pas
        if node_label in nodes:
            node_id = str(nodes[node_label])
            if node_id in sent_features:
                return sent_features[node_id][param]
            else:
                print(f"⚠️ No feature for node_id '{node_id}' in sentence '{sent_id}'")
        return ""

    for match in liste_match:
        sent_id = match["sent_id"]
        formes_EE = {}
        nodes = match["matching"]["nodes"]
        sent_features = corpus[sent_id].features

        formes_EE["D1"] = get_feature("Z")
        formes_EE["N1"] = get_feature("Y")
        formes_EE["AUX"] = get_feature("W")
        formes_EE["V"] = get_feature("X")

        dico_un_enonce_elem = {
            "id_EE": n + 1,
            "id_sent": sent_id,
            "formes_EE": formes_EE,
        }
        liste_des_enonces_elem.append(dico_un_enonce_elem)
        n += 1

    print(f"Indexation des énoncés terminée en {time.time() - start_indexation:.2f} secondes")
    return liste_des_enonces_elem



def supprimer_doublons_semantiques(enonces):
    uniques = {}

    for e in enonces:
        cle = (e["id_sent"], e["formes_EE"]["N1"], e["formes_EE"]["V"])
        score = sum(1 for v in e["formes_EE"].values() if v != "")

        if cle not in uniques or score > uniques[cle][0]:
            uniques[cle] = (score, e)

    return [val[1] for val in uniques.values()]


def generate_conll(text, filename):
    """
    Génère un fichier CoNLL à partir d'un texte donné en utilisant Stanza.
    :param text: Texte à analyser
    :param filename: Nom du fichier de sortie (sans extension)
    """
    start = time.time()
    print("Analyse pour génération du conll en cours...")
    doc = nlp(text)
    print(f"Analyse terminée en {time.time() - start:.2f} secondes")

    # sauvegarder au format .conll
    conll = doc._.conll_str
    with open(f"../../data/ParlaMint-FR_2018-conll/{filename}.conll", "w", encoding="utf-8") as f:
        f.write(conll)
    print(f"Conll généré et stocké dans ../../data/ParlaMint-FR_2018-conll/{filename}.conll")



if __name__ == "__main__":
    # patch_torch_load()

    # Initialisation pipeline NLP
    print("Initialisation de la pipeline NLP...")
    nlp = init_parser(
    "fr",
    "stanza",
    parser_opts={"use_gpu": False, "verbose": True},  # use_gpu=False pour éviter les problèmes de mémoire
    include_headers=True
    )

    ## GÉNÉRATION DES FICHIERS CONLL

    ## Corpus Parlamint2018_raw disponible sur le dépôt 
    dossier_parent = Path("../../data/Parlamint2018_raw")

    # Lire tous les fichiers contenus dans les sous-dossiers
    fichiers = [fichier for fichier in dossier_parent.iterdir() if fichier.is_dir() for fichier in fichier.iterdir() if fichier.is_file()]

    for fichier in fichiers:
        print(fichier.stem)
        with open(fichier, "r", encoding="utf-8") as f:
            contenu = f.read()

            destination_filename = fichier.stem  # Utiliser le nom de fichier sans extension

            # Appel unique, nlp déjà initialisé
            generate_conll(contenu, destination_filename)

    ## FIN GÉNÉRATION DES FICHIERS CONLL
    ## ANALYSE DES FICHIERS CONLL POUR EN EXTRAIRE LES ÉNONCÉS ÉLÉMENTAIRES
    
    # Chronométrage du chargement du corpus
    start_corpus = time.time()
    print("Chargement du corpus...")
    treebank_folder_path = "../data/ParlaMint_100"
    corpus = grewpy.Corpus(treebank_folder_path)
    print(f"Corpus chargé en {time.time() - start_corpus:.2f} secondes\n")

    param = "lemma"

    # Chronométrage de la recherche de motifs
    print("Recherche de motifs...")
    start_patterns = time.time()

    # Initialisation de la liste pour stocker tous les matches
    all_matches = []

    pattern = """pattern {
            X-[nsubj|obj|iobj|nsubj:pass|cop]->Y;
            Y-[det]->Z;
            X-[aux:pass|aux:tense]->W;
        }"""
    
    req = grewpy.Request(pattern)
    matches = corpus.search(req)
    all_matches.extend(matches)
    print(f"Motifs trouvés en {time.time() - start_patterns:.2f} secondes\n")

    ## INDEXATION DES ÉNONCÉS ÉLÉMENTAIRES DANS UNE LISTE DE DICTIONNAIRES
    # Chronométrage de l’indexation
    print("Indexation des énoncés élémentaires...")
    start_indexation = time.time()
    liste_enonces_elem = indexe_enonces_elem(corpus, all_matches, param)
    print(f"Indexation terminée en {time.time() - start_indexation:.2f} secondes\n")

    # Affichage final
    nb_EE = len(liste_enonces_elem)
    print(f"Nombre d'énoncés élémentaires indexés : {nb_EE}\n")
    # print("\n\nLISTE DES ÉNONCÉS ÉLÉMENTAIRES :\n")
    # for element in liste_enonces_elem:
    #     print(element)

    ## ENREGISTREMENT DES ÉNONCÉS ÉLÉMENTAIRES DANS UN FICHIER TEXTE RÉUTILISABLE POUR L'ANALYSE SUIVANTE AVEC R
    
    # Enregistrement dans un fichier texte
    print("Écriture des énoncés dans un fichier texte...")

    output_path = "../../data/enonces_elementaires.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        for ee in liste_enonces_elem:
            formes = ee["formes_EE"]
            phrase = " ".join([formes["D1"], formes["N1"], formes["AUX"], formes["V"]])
            f.write(phrase.strip() + ".\n")

    print(f"Fichier texte généré : {output_path}")
