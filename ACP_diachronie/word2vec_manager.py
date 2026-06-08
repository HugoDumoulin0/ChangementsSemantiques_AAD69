
import os
import json
from gensim.models import Word2Vec



# Modèle global
def entrainer_modele_global(
        sentences,
        vector_size=200,
        window=5,
        min_count=5,
        epochs=20,
        negative=10,
        sg=1,
        workers=4
):
    """
    Entraîne le modèle global.
    """

    print()

    print("Entraînement modèle global...")

    model = Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        sg=sg,
        negative=negative,
        epochs=epochs
    )

    print(
        f"Vocabulaire global : "
        f"{len(model.wv)} mots"
    )

    return model



# Modèles décennie
def entrainer_modeles_decennies(
        corpus_decennies,
        vector_size=200,
        window=5,
        min_count=5,
        epochs=20,
        negative=10,
        sg=1,
        workers=4,
        min_documents=5
):
    """
    Entraîne un modèle par décennie.
    """

    modeles = {}

    for decennie in sorted(corpus_decennies):

        sentences = corpus_decennies[
            decennie
        ]

        if len(sentences) < min_documents:

            print(
                f"Décennie {decennie} ignorée "
                f"(pas assez de documents)"
            )

            continue

        print()

        print(
            f"Entraînement décennie "
            f"{decennie}"
        )

        print(
            f"Documents : "
            f"{len(sentences)}"
        )

        model = Word2Vec(
            sentences=sentences,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            workers=workers,
            sg=sg,
            negative=negative,
            epochs=epochs
        )

        print(
            f"Vocabulaire : "
            f"{len(model.wv)} mots"
        )

        modeles[decennie] = model

    return modeles


# Sauvegarde
def sauvegarder_modeles(
        modele_global,
        modeles_decennies,
        dossier="models",
        metadata=None
):
    """
    Sauvegarde tous les modèles.
    """

    os.makedirs(
        dossier,
        exist_ok=True
    )

    modele_global.save(
        os.path.join(
            dossier,
            "global.model"
        )
    )

    for decennie, modele in modeles_decennies.items():

        modele.save(
            os.path.join(
                dossier,
                f"{decennie}.model"
            )
        )

    if metadata is not None:

        with open(
            os.path.join(
                dossier,
                "metadata.json"
            ),
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                metadata,
                f,
                ensure_ascii=True,
                indent=2
            )

    print()

    print(
        f"Modèles sauvegardés "
        f"dans : {dossier}"
    )



# Chargement
def charger_modele(
        chemin_modele
):
    """
    Recharge un modèle sauvegardé.
    """

    return Word2Vec.load(
        chemin_modele
    )


def charger_modeles(
        dossier="models"
):
    """
    Recharge tous les modèles sauvegardés
    ainsi que les métadonnées éventuelles.
    """

    chemin_global = os.path.join(
        dossier,
        "global.model"
    )

    if not os.path.exists(chemin_global):
        raise FileNotFoundError(
            f"Modèle global introuvable : {chemin_global}"
        )

    modele_global = charger_modele(
        chemin_global
    )

    modeles_decennies = {}

    for nom_fichier in os.listdir(dossier):

        if not nom_fichier.endswith(".model"):
            continue

        if nom_fichier == "global.model":
            continue

        decennie_str = nom_fichier.replace(
            ".model",
            ""
        )

        try:
            decennie = int(decennie_str)
        except ValueError:
            continue

        modeles_decennies[decennie] = charger_modele(
            os.path.join(
                dossier,
                nom_fichier
            )
        )

    metadata = {}

    chemin_metadata = os.path.join(
        dossier,
        "metadata.json"
    )

    if os.path.exists(chemin_metadata):

        with open(
            chemin_metadata,
            "r",
            encoding="utf-8"
        ) as f:

            metadata = json.load(f)

    return (
        modele_global,
        modeles_decennies,
        metadata
    )



# Informations
def informations_modele(
        modele
):
    """
    Retourne quelques statistiques.
    """

    return {

        "vocabulaire":
            len(modele.wv),

        "dimensions":
            modele.vector_size
    }
