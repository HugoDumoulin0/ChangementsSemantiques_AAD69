
import os
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
        dossier="models"
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