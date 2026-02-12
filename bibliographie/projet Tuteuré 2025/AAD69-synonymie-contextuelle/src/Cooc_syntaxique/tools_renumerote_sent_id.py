import os

"""
Renumérote les sent_id dans tous les fichiers d'un dossier donné.
Chaque sent_id est remplacé par un numéro global incrémenté à partir de 1.
Le script crée un nouveau dossier pour les fichiers renumérotés.
"""

def renumerote_sent_id(dossier):
    sent_id_global = 1  # compteur global

    for nom_fichier in sorted(os.listdir(dossier)):
        chemin = os.path.join(dossier, nom_fichier)
        if not os.path.isfile(chemin):
            continue

        with open(chemin, "r", encoding="utf-8") as f:
            lignes = f.readlines()

        nouvelles_lignes = []
        for ligne in lignes:
            if ligne.startswith("# sent_id ="):
                nouvelle_ligne = f"# sent_id = {sent_id_global}\n"
                nouvelles_lignes.append(nouvelle_ligne)
                sent_id_global += 1
            else:
                nouvelles_lignes.append(ligne)

        # # Option 1 : écrase le fichier original
        # with open(chemin, "w", encoding="utf-8") as f:
        #     f.writelines(nouvelles_lignes)

        # Option 2 (sécurité) : écrit dans un dossier "sortie"
        desti = f"{dossier}_renumerotes"
        os.makedirs(desti, exist_ok=True)
        with open(os.path.join(desti, nom_fichier), "w", encoding="utf-8") as f:
            f.writelines(nouvelles_lignes)

    print(f"Tous les sent_id ont été renumérotés dans le dossier : {desti}")

# Exemple d'utilisation :
renumerote_sent_id("../../data/ParlaMint-FR_2018-conll_non-renumerotes")