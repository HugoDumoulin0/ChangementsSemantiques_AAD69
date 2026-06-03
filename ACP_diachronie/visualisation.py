
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Backend compatible avec Tkinter
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


# PCA
def projeter_pca(vecteurs):
    """
    Projection PCA 2D.
    """

    if len(vecteurs) < 2:
        return None

    pca = PCA(
        n_components=2
    )

    coords = pca.fit_transform(
        np.array(vecteurs)
    )

    return coords



# Trajectoire diachronique
def afficher_trajectoire(
        mot,
        vecteurs,
        labels,
        sauvegarder=False,
        fichier_sortie=None
):
    """
    Affiche la trajectoire du mot.
    """

    if len(vecteurs) < 2:

        print(
            "Pas assez de points à afficher."
        )

        return

    # Vérification
    global_present = (
        labels[-1] == "GLOBAL"
    )

    if global_present:

        vecteurs_decennies = (
            vecteurs[:-1]
        )

        vecteur_global = (
            vecteurs[-1]
        )

    else:

        vecteurs_decennies = (
            vecteurs
        )

        vecteur_global = None

    # PCA sur les décennies
    pca = PCA(
        n_components=2
    )

    coords_decennies = (
        pca.fit_transform(
            np.array(
                vecteurs_decennies
            )
        )
    )

    # Projection du global
    coord_global = None

    if vecteur_global is not None:

        coord_global = pca.transform(
            vecteur_global.reshape(1, -1)
        )

    # Figure
    plt.figure(
        figsize=(14, 10)
    )

    xs = coords_decennies[:, 0]
    ys = coords_decennies[:, 1]

    # trajectoire

    plt.plot(
        xs,
        ys,
        marker="o"
    )

    # Décennies
    for i, label in enumerate(
            labels[:-1]
            if global_present
            else labels
    ):

        plt.annotate(
            label,
            (
                xs[i],
                ys[i]
            ),
            xytext=(5, 5),
            textcoords="offset points"
        )


    # Global
    if coord_global is not None:

        plt.scatter(
            coord_global[0, 0],
            coord_global[0, 1],
            s=250,
            marker="X"
        )

        plt.annotate(
            "GLOBAL",
            (
                coord_global[0, 0],
                coord_global[0, 1]
            ),
            xytext=(10, 10),
            textcoords="offset points"
        )

    plt.title(
        f"Trajectoire diachronique : {mot}"
    )

    plt.xlabel(
        "Composante principale 1"
    )

    plt.ylabel(
        "Composante principale 2"
    )

    plt.grid(True)

    plt.tight_layout()


    # Export PNG
    if sauvegarder and fichier_sortie:

        plt.savefig(
            fichier_sortie,
            dpi=300,
            bbox_inches="tight"
        )

    plt.show()

# Distances au global
def afficher_distances_global(
        mot,
        distances
):
    """
    Graphique de déplacement
    sémantique par rapport au global.
    """

    if not distances:
        return

    decennies = list(
        distances.keys()
    )

    valeurs = list(
        distances.values()
    )

    plt.figure(
        figsize=(12, 6)
    )

    plt.plot(
        decennies,
        valeurs,
        marker="o"
    )

    plt.title(
        f"Distance au modèle global : {mot}"
    )

    plt.xlabel(
        "Décennie"
    )

    plt.ylabel(
        "Distance euclidienne"
    )

    plt.grid(True)

    plt.tight_layout()

    plt.show()



# Distances décennie -> décennie
def afficher_distances_temporelles(
        mot,
        distances
):
    """
    Evolution entre décennies.
    """

    if not distances:
        return

    labels = []

    valeurs = []

    for periode, distance in distances.items():

        debut, fin = periode

        labels.append(
            f"{debut}-{fin}"
        )

        valeurs.append(
            distance
        )

    plt.figure(
        figsize=(12, 6)
    )

    plt.bar(
        labels,
        valeurs
    )

    plt.title(
        f"Variation sémantique : {mot}"
    )

    plt.xlabel(
        "Transition"
    )

    plt.ylabel(
        "Distance"
    )

    plt.xticks(
        rotation=45
    )

    plt.tight_layout()

    plt.show()