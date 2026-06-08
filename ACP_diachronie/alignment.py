
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
        topn=10,
        min_count_voisin=0
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
            topn=max(topn + 20, topn * 5)
        )

        similaires = [
            x
            for x in similaires
            if (
                x[0] != mot
                and x[0] in modele.wv
                and modele.wv.get_vecattr(
                    x[0],
                    "count"
                ) >= min_count_voisin
            )
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


def _vecteurs_par_decennie_pour_mot(
        mot,
        modeles_decennies,
        rotations
):
    """
    Retourne les vecteurs alignés d'un mot
    par décennie disponible.
    """

    vecteurs = []

    for decennie in sorted(modeles_decennies):

        if decennie not in rotations:
            continue

        modele = modeles_decennies[decennie]

        if mot not in modele.wv:
            continue

        vecteurs.append(
            (
                decennie,
                vecteur_aligne(
                    modele,
                    mot,
                    rotations[decennie]
                )
            )
        )

    return vecteurs


def trajectoire_extremes_vocabulaire(
        modeles_decennies,
        rotations
):
    """
    Distance directe entre la première
    et la dernière décennie observées
    pour chaque mot.
    """

    decennies_valides = sorted(
        decennie
        for decennie in modeles_decennies
        if decennie in rotations
    )

    if len(decennies_valides) < 2:
        return {}

    premiere = decennies_valides[0]
    derniere = decennies_valides[-1]

    mots_premiere = set(
        modeles_decennies[premiere].wv.index_to_key
    )

    mots_derniere = set(
        modeles_decennies[derniere].wv.index_to_key
    )

    communs = sorted(
        mots_premiere & mots_derniere
    )

    trajectoires = {}

    for mot in communs:

        vecteur_premiere = vecteur_aligne(
            modeles_decennies[premiere],
            mot,
            rotations[premiere]
        )

        vecteur_derniere = vecteur_aligne(
            modeles_decennies[derniere],
            mot,
            rotations[derniere]
        )

        trajectoires[mot] = {
            "premiere_decennie": premiere,
            "derniere_decennie": derniere,
            "distance": float(
                np.linalg.norm(
                    vecteur_derniere -
                    vecteur_premiere
                )
            )
        }

    return trajectoires


def trajectoire_cumulee_vocabulaire(
        modeles_decennies,
        rotations
):
    """
    Somme des distances successives
    pour chaque mot sur les décennies
    où il est observé.
    """

    vocabulaire = set()

    for decennie, modele in modeles_decennies.items():

        if decennie not in rotations:
            continue

        vocabulaire.update(
            modele.wv.index_to_key
        )

    trajectoires = {}

    for mot in sorted(vocabulaire):

        vecteurs = _vecteurs_par_decennie_pour_mot(
            mot,
            modeles_decennies,
            rotations
        )

        if len(vecteurs) < 2:
            continue

        somme = 0.0
        transitions = []

        for i in range(len(vecteurs) - 1):

            decennie_a, vecteur_a = vecteurs[i]
            decennie_b, vecteur_b = vecteurs[i + 1]

            distance = float(
                np.linalg.norm(
                    vecteur_b - vecteur_a
                )
            )

            somme += distance

            transitions.append(
                {
                    "debut": decennie_a,
                    "fin": decennie_b,
                    "distance": distance
                }
            )

        trajectoires[mot] = {
            "premiere_decennie": vecteurs[0][0],
            "derniere_decennie": vecteurs[-1][0],
            "distance": somme,
            "transitions": transitions
        }

    return trajectoires
