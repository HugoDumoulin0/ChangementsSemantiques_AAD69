
import numpy as np

from scipy.linalg import orthogonal_procrustes

# Paramètres
MIN_COMMON_WORDS = 500


# Normalisation
def normaliser_matrice(X):
    """
    Normalise chaque vecteur ligne.
    """

    normes = np.linalg.norm(
        X,
        axis=1,
        keepdims=True
    )

    normes[normes == 0] = 1

    return X / normes


# Alignement Procrustes
def aligner_modele(
        modele_reference,
        modele_local,
        min_common_words=MIN_COMMON_WORDS
):
    """
    Calcule la matrice de rotation
    permettant d'aligner modele_local
    sur modele_reference.
    """

    vocab_global = set(
        modele_reference.wv.index_to_key
    )

    vocab_local = set(
        modele_local.wv.index_to_key
    )

    communs = sorted(
        vocab_global &
        vocab_local
    )

    if len(communs) < min_common_words:
        return None

    X = np.array([
        modele_reference.wv[mot]
        for mot in communs
    ])

    Y = np.array([
        modele_local.wv[mot]
        for mot in communs
    ])

    X = normaliser_matrice(X)
    Y = normaliser_matrice(Y)

    R, _ = orthogonal_procrustes(
        Y,
        X
    )

    return R


# Construction des rotations
def construire_rotations(
        modele_global,
        modeles_decennies,
        min_common_words=MIN_COMMON_WORDS
):
    """
    Pré-calcule toutes les matrices
    de rotation.
    """

    rotations = {}

    for decennie, modele in modeles_decennies.items():

        rotation = aligner_modele(
            modele_global,
            modele,
            min_common_words=min_common_words
        )

        if rotation is not None:

            rotations[decennie] = rotation

    return rotations



# Projection d'un vecteur
def vecteur_aligne(
        modele,
        mot,
        matrice_rotation
):
    """
    Retourne le vecteur aligné.
    """

    vecteur = modele.wv[mot]

    return vecteur @ matrice_rotation


# Construction trajectoire temporelle
def construire_trajectoire(
        mot,
        modele_global,
        modeles_decennies,
        rotations
):
    """
    Construit la trajectoire du mot.
    """

    vecteurs = []

    labels = []

    for decennie in sorted(modeles_decennies):

        if decennie not in rotations:
            continue

        modele = modeles_decennies[decennie]

        if mot not in modele.wv:
            continue

        vecteur = vecteur_aligne(
            modele,
            mot,
            rotations[decennie]
        )

        vecteurs.append(vecteur)

        labels.append(str(decennie))

    if mot in modele_global.wv:

        vecteurs.append(
            modele_global.wv[mot]
        )

        labels.append("GLOBAL")

    return vecteurs, labels



# Voisins sémantiques
def most_similar_par_decennie(
        mot,
        modele_global,
        modeles_decennies,
        rotations,
        topn=10
):
    """
    Voisins sémantiques du mot
    dans chaque décennie.
    """

    resultats = {}

    for decennie in sorted(modeles_decennies):

        if decennie not in rotations:
            continue

        modele = modeles_decennies[decennie]

        if mot not in modele.wv:
            continue

        vecteur = vecteur_aligne(
            modele,
            mot,
            rotations[decennie]
        )

        similaires = modele_global.wv.similar_by_vector(
            vecteur,
            topn=topn + 5
        )

        similaires = [
            x
            for x in similaires
            if x[0] != mot
        ]

        resultats[decennie] = similaires[:topn]

    return resultats



# Distance au modèle global
def distance_au_global(
        mot,
        modele_global,
        modeles_decennies,
        rotations
):
    """
    Distance entre chaque décennie
    et le modèle global.
    """

    distances = {}

    if mot not in modele_global.wv:
        return distances

    vecteur_global = modele_global.wv[mot]

    for decennie in sorted(modeles_decennies):

        if decennie not in rotations:
            continue

        modele = modeles_decennies[decennie]

        if mot not in modele.wv:
            continue

        vecteur_local = vecteur_aligne(
            modele,
            mot,
            rotations[decennie]
        )

        distance = np.linalg.norm(
            vecteur_local - vecteur_global
        )

        distances[decennie] = float(distance)

    return distances


# Distance entre décennies successives
def distance_temporelle(
        mot,
        modeles_decennies,
        rotations
):
    """
    Mesure l'évolution du mot
    entre décennies successives.
    """

    vecteurs = []
    labels = []

    for decennie in sorted(modeles_decennies):

        if decennie not in rotations:
            continue

        modele = modeles_decennies[decennie]

        if mot not in modele.wv:
            continue

        vecteur = vecteur_aligne(
            modele,
            mot,
            rotations[decennie]
        )

        vecteurs.append(vecteur)

        labels.append(decennie)

    distances = {}

    for i in range(len(vecteurs) - 1):

        d = np.linalg.norm(
            vecteurs[i + 1] -
            vecteurs[i]
        )

        cle = (
            labels[i],
            labels[i + 1]
        )

        distances[cle] = float(d)

    return distances
