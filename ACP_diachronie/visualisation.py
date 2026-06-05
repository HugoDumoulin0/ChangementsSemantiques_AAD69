import numpy as np
from matplotlib.figure import Figure
from sklearn.decomposition import PCA

COULEUR_TRAJECTOIRE = "#1f3a5f"
COULEUR_GLOBAL = "#d1495b"
COULEUR_VOISINS = "#2a9d8f"
COULEUR_LIAISON = "#9aa5b1"


def creer_figure_vide(message):
    figure = Figure(figsize=(8, 5), dpi=100)
    axe = figure.add_subplot(111)
    axe.text(
        0.5,
        0.5,
        message,
        ha="center",
        va="center",
        fontsize=12
    )
    axe.set_axis_off()
    figure.tight_layout()
    return figure


def creer_figure_trajectoire(
        mot,
        vecteurs,
        labels,
        voisins=None,
        nombre_voisins=0
):
    """
    Crée la figure de trajectoire diachronique.
    """

    if len(vecteurs) < 2:
        return creer_figure_vide(
            "Pas assez de points à afficher."
        )

    global_present = (
        labels[-1] == "GLOBAL"
    )

    if global_present:
        vecteurs_decennies = vecteurs[:-1]
        vecteur_global = vecteurs[-1]
    else:
        vecteurs_decennies = vecteurs
        vecteur_global = None

    if len(vecteurs_decennies) < 2:
        figure = Figure(
            figsize=(8, 5),
            dpi=100
        )

        axe = figure.add_subplot(111)
        etiquettes = labels[:-1] if global_present else labels

        axe.scatter(
            [0],
            [0],
            s=110,
            color=COULEUR_TRAJECTOIRE,
            label="Mot cible"
        )

        axe.annotate(
            etiquettes[0],
            (0, 0),
            xytext=(5, 5),
            textcoords="offset points"
        )

        if nombre_voisins > 0 and voisins:
            try:
                decennie = int(etiquettes[0])
            except ValueError:
                decennie = None

            if decennie in voisins:
                voisins_decennie = voisins[decennie][
                    :nombre_voisins
                ]

                if voisins_decennie:
                    xs_voisins = [
                        0.2 + (0.12 * i)
                        for i, _ in enumerate(voisins_decennie)
                    ]
                    ys_voisins = [
                        -0.08 - (0.05 * i)
                        for i, _ in enumerate(voisins_decennie)
                    ]

                    axe.scatter(
                        xs_voisins,
                        ys_voisins,
                        s=45,
                        marker="s",
                        alpha=0.55,
                        color=COULEUR_VOISINS,
                        label="Voisins"
                    )

                    for i, (voisin, _vecteur) in enumerate(voisins_decennie):
                        axe.annotate(
                            voisin,
                            (xs_voisins[i], ys_voisins[i]),
                            xytext=(5, 3),
                            textcoords="offset points",
                            fontsize=8
                        )

        if vecteur_global is not None:
            axe.scatter(
                [1],
                [0],
                s=250,
                marker="X",
                color=COULEUR_GLOBAL,
                label="GLOBAL"
            )

            axe.annotate(
                "GLOBAL",
                (1, 0),
                xytext=(10, 10),
                textcoords="offset points"
            )

            axe.plot(
                [0, 1],
                [0, 0],
                linestyle="--",
                color=COULEUR_LIAISON,
                alpha=0.8
            )

        axe.set_title(
            f"Trajectoire diachronique : {mot}"
        )
        axe.set_xlabel(
            "Projection simplifiée"
        )
        axe.set_ylabel(
            "Valeur relative"
        )
        axe.grid(True)
        axe.legend(loc="best")

        figure.tight_layout()
        return figure

    pca = PCA(
        n_components=2
    )

    coords_decennies = pca.fit_transform(
        np.array(vecteurs_decennies)
    )

    coord_global = None

    if vecteur_global is not None:
        coord_global = pca.transform(
            vecteur_global.reshape(1, -1)
        )

    figure = Figure(
        figsize=(8, 5),
        dpi=100
    )

    axe = figure.add_subplot(111)

    xs = coords_decennies[:, 0]
    ys = coords_decennies[:, 1]

    axe.plot(
        xs,
        ys,
        marker="o",
        color=COULEUR_TRAJECTOIRE,
        linewidth=2.2,
        markersize=6,
        label="Mot cible"
    )

    etiquettes = labels[:-1] if global_present else labels

    for i, label in enumerate(etiquettes):
        axe.annotate(
            label,
            (xs[i], ys[i]),
            xytext=(5, 5),
            textcoords="offset points"
        )

        if nombre_voisins > 0 and voisins:
            try:
                decennie = int(label)
            except ValueError:
                decennie = None

            if decennie in voisins:
                voisins_decennie = voisins[decennie][
                    :nombre_voisins
                ]

                if voisins_decennie:
                    coords_voisins = pca.transform(
                        np.array([
                            vecteur
                            for _voisin, vecteur in voisins_decennie
                        ])
                    )

                    axe.scatter(
                        coords_voisins[:, 0],
                        coords_voisins[:, 1],
                        s=45,
                        marker="s",
                        alpha=0.55,
                        color=COULEUR_VOISINS,
                        label="Voisins"
                    )

                    for coord_x, coord_y in coords_voisins:
                        axe.plot(
                            [xs[i], coord_x],
                            [ys[i], coord_y],
                            color=COULEUR_LIAISON,
                            linewidth=0.8,
                            alpha=0.35
                        )

                    for j, (voisin, _vecteur) in enumerate(voisins_decennie):
                        axe.annotate(
                            voisin,
                            (
                                coords_voisins[j, 0],
                                coords_voisins[j, 1]
                            ),
                            xytext=(5, 3),
                            textcoords="offset points",
                            fontsize=8
                        )

    if coord_global is not None:
        axe.scatter(
            coord_global[0, 0],
            coord_global[0, 1],
            s=250,
            marker="X",
            color=COULEUR_GLOBAL,
            label="GLOBAL"
        )

        axe.annotate(
            "GLOBAL",
            (
                coord_global[0, 0],
                coord_global[0, 1]
            ),
            xytext=(10, 10),
            textcoords="offset points"
        )

    axe.set_title(
        f"Trajectoire diachronique : {mot}"
    )
    axe.set_xlabel(
        "Composante principale 1"
    )
    axe.set_ylabel(
        "Composante principale 2"
    )
    axe.grid(True, alpha=0.25)

    handles, legend_labels = axe.get_legend_handles_labels()
    uniques = []
    deja_vus = set()

    for handle, legend_label in zip(handles, legend_labels):
        if legend_label in deja_vus:
            continue
        deja_vus.add(legend_label)
        uniques.append((handle, legend_label))

    if uniques:
        axe.legend(
            [handle for handle, _label in uniques],
            [legend_label for _handle, legend_label in uniques],
            loc="best"
        )

    figure.tight_layout()
    return figure


def creer_figure_distances_global(
        mot,
        distances
):
    """
    Crée la figure des distances au global.
    """

    if not distances:
        return creer_figure_vide(
            "Aucune distance au modèle global à afficher."
        )

    figure = Figure(
        figsize=(8, 5),
        dpi=100
    )

    axe = figure.add_subplot(111)

    decennies = list(
        distances.keys()
    )

    valeurs = list(
        distances.values()
    )

    axe.plot(
        decennies,
        valeurs,
        marker="o"
    )

    axe.set_title(
        f"Distance au modèle global : {mot}"
    )
    axe.set_xlabel(
        "Décennie"
    )
    axe.set_ylabel(
        "Distance euclidienne"
    )
    axe.grid(True)

    figure.tight_layout()
    return figure


def creer_figure_distances_temporelles(
        mot,
        distances
):
    """
    Crée la figure des distances entre décennies.
    """

    if not distances:
        return creer_figure_vide(
            "Aucune distance temporelle à afficher."
        )

    figure = Figure(
        figsize=(8, 5),
        dpi=100
    )

    axe = figure.add_subplot(111)

    labels = []
    valeurs = []

    for periode, distance in distances.items():
        debut, fin = periode
        labels.append(
            f"{debut}-{fin}"
        )
        valeurs.append(distance)

    positions = np.arange(
        len(labels)
    )

    axe.bar(
        positions,
        valeurs
    )

    axe.set_title(
        f"Distances temporelles : {mot}"
    )
    axe.set_xlabel(
        "Transition"
    )
    axe.set_ylabel(
        "Distance euclidienne"
    )
    axe.set_xticks(
        positions
    )
    axe.set_xticklabels(
        labels,
        rotation=45,
        ha="right"
    )
    axe.grid(
        True,
        axis="y"
    )

    figure.tight_layout()
    return figure
