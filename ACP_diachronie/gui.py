from corpus import (
    charger_corpus,
    indexer_occurrences_corpus,
    normaliser_mot_recherche
)

from word2vec_manager import (
    charger_modeles,
    entrainer_modele_global,
    entrainer_modeles_decennies,
    sauvegarder_modeles
)

from alignment import (
    construire_rotations,
    construire_trajectoire,
    most_similar_par_decennie,
    distance_au_global,
    distance_temporelle
)

from visualisation import (
    creer_figure_trajectoire,
    creer_figure_distances_global,
    creer_figure_distances_temporelles,
    creer_figure_vide
)




import tkinter as tk

from tkinter import ttk
from tkinter import filedialog
from tkinter import messagebox

import csv
import hashlib
import json
import os
import shutil
import threading
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg,
    NavigationToolbar2Tk
)


class Application(tk.Tk):

    def __init__(self):
        super().__init__()

        self.title(
            "Analyse Diachronique Word2Vec"
        )

        self.geometry(
            "1200x700"
        )

        self.corpus_cache = None
        self.occurrences_cache = None
        self.modeles_cache = None
        self.analyse_en_cours = False
        self.verification_en_cours = False
        self.graph_canvases = {}
        self.fenetre_graphes = None
        self.dernier_resultat = None

        self.creer_interface()
        self.creer_fenetre_graphes()

    
    # Interface
    def creer_interface(self):
        frame = ttk.Frame(
            self,
            padding=10
        )

        frame.pack(
            fill="both",
            expand=True
        )

        frame.columnconfigure(
            0,
            weight=1
        )

        bloc_corpus = ttk.LabelFrame(
            frame,
            text="Corpus",
            padding=10
        )

        bloc_corpus.grid(
            row=0,
            column=0,
            sticky="ew",
            pady=(0, 10)
        )

        bloc_corpus.columnconfigure(
            1,
            weight=1
        )

        ttk.Label(
            bloc_corpus,
            text="Dossier Corpus"
        ).grid(
            row=0,
            column=0,
            sticky="w"
        )

        self.var_dossier = tk.StringVar()

        ttk.Entry(
            bloc_corpus,
            textvariable=self.var_dossier,
            width=60
        ).grid(
            row=0,
            column=1,
            padx=6,
            pady=4,
            sticky="ew"
        )

        ttk.Button(
            bloc_corpus,
            text="Parcourir",
            command=self.choisir_dossier
        ).grid(
            row=0,
            column=2,
            padx=(0, 4)
        )

        ttk.Label(
            bloc_corpus,
            text="Mot recherché"
        ).grid(
            row=1,
            column=0,
            sticky="w"
        )

        self.var_mot = tk.StringVar()

        ttk.Entry(
            bloc_corpus,
            textvariable=self.var_mot,
            width=30
        ).grid(
            row=1,
            column=1,
            padx=6,
            pady=4,
            sticky="w"
        )

        ttk.Button(
            bloc_corpus,
            text="Vérifier mot",
            command=self.verifier_mot_thread
        ).grid(
            row=1,
            column=2,
            padx=(0, 4)
        )

        self.var_assistant_mot = tk.StringVar(
            value="Assistant corpus : en attente."
        )

        ttk.Label(
            bloc_corpus,
            textvariable=self.var_assistant_mot,
            wraplength=720
        ).grid(
            row=2,
            column=1,
            columnspan=2,
            sticky="w",
            pady=(4, 0)
        )

        bloc_parametres = ttk.LabelFrame(
            frame,
            text="Paramètres modèle",
            padding=10
        )

        bloc_parametres.grid(
            row=1,
            column=0,
            sticky="ew",
            pady=(0, 10)
        )

        for colonne in range(4):
            bloc_parametres.columnconfigure(
                colonne,
                weight=1 if colonne in (1, 3) else 0
            )

        ttk.Label(
            bloc_parametres,
            text="Taille vecteur"
        ).grid(
            row=0,
            column=0,
            sticky="w",
            padx=(0, 8),
            pady=4
        )

        self.var_vector_size = tk.IntVar(
            value=200
        )

        ttk.Entry(
            bloc_parametres,
            textvariable=self.var_vector_size,
            width=12
        ).grid(
            row=0,
            column=1,
            sticky="w",
            pady=4
        )

        ttk.Label(
            bloc_parametres,
            text="Min Count"
        ).grid(
            row=0,
            column=2,
            sticky="w",
            padx=(16, 8),
            pady=4
        )

        self.var_min_count = tk.IntVar(
            value=5
        )

        ttk.Entry(
            bloc_parametres,
            textvariable=self.var_min_count,
            width=12
        ).grid(
            row=0,
            column=3,
            sticky="w",
            pady=4
        )

        ttk.Label(
            bloc_parametres,
            text="Fenêtre"
        ).grid(
            row=1,
            column=0,
            sticky="w",
            padx=(0, 8),
            pady=4
        )

        self.var_window = tk.IntVar(
            value=5
        )

        ttk.Entry(
            bloc_parametres,
            textvariable=self.var_window,
            width=12
        ).grid(
            row=1,
            column=1,
            sticky="w",
            pady=4
        )

        ttk.Label(
            bloc_parametres,
            text="Min Docs / décennie"
        ).grid(
            row=1,
            column=2,
            sticky="w",
            padx=(16, 8),
            pady=4
        )

        self.var_min_documents = tk.IntVar(
            value=5
        )

        ttk.Entry(
            bloc_parametres,
            textvariable=self.var_min_documents,
            width=12
        ).grid(
            row=1,
            column=3,
            sticky="w",
            pady=4
        )

        ttk.Label(
            bloc_parametres,
            text="Epochs"
        ).grid(
            row=2,
            column=0,
            sticky="w",
            padx=(0, 8),
            pady=4
        )

        self.var_epochs = tk.IntVar(
            value=20
        )

        ttk.Entry(
            bloc_parametres,
            textvariable=self.var_epochs,
            width=12
        ).grid(
            row=2,
            column=1,
            sticky="w",
            pady=4
        )

        ttk.Label(
            bloc_parametres,
            text="Mots communs min"
        ).grid(
            row=2,
            column=2,
            sticky="w",
            padx=(16, 8),
            pady=4
        )

        self.var_min_common_words = tk.IntVar(
            value=500
        )

        ttk.Entry(
            bloc_parametres,
            textvariable=self.var_min_common_words,
            width=12
        ).grid(
            row=2,
            column=3,
            sticky="w",
            pady=4
        )

        bloc_options = ttk.LabelFrame(
            frame,
            text="Options d'analyse",
            padding=10
        )

        bloc_options.grid(
            row=2,
            column=0,
            sticky="ew",
            pady=(0, 10)
        )

        for colonne in range(4):
            bloc_options.columnconfigure(
                colonne,
                weight=1 if colonne == 3 else 0
            )

        self.var_stopwords = tk.BooleanVar(
            value=True
        )

        ttk.Checkbutton(
            bloc_options,
            text="Supprimer les stopwords",
            variable=self.var_stopwords
        ).grid(
            row=0,
            column=0,
            padx=(0, 16),
            pady=4,
            sticky="w"
        )

        self.var_lemmatisation = tk.BooleanVar(
            value=True
        )

        ttk.Checkbutton(
            bloc_options,
            text="Lemmatisation",
            variable=self.var_lemmatisation
        ).grid(
            row=0,
            column=1,
            padx=(0, 16),
            pady=4,
            sticky="w"
        )

        ttk.Label(
            bloc_options,
            text="Nombre de voisins"
        ).grid(
            row=0,
            column=2,
            padx=(0, 8),
            pady=4,
            sticky="w"
        )

        self.var_topn = tk.IntVar(
            value=10
        )

        ttk.Entry(
            bloc_options,
            textvariable=self.var_topn,
            width=12
        ).grid(
            row=0,
            column=3,
            sticky="w",
            pady=4
        )

        self.var_mode_voisins_graphe = tk.StringVar(
            value="Aucun"
        )

        ttk.Label(
            bloc_options,
            text="Voisins sur le graphique"
        ).grid(
            row=1,
            column=0,
            padx=(0, 8),
            pady=(6, 0),
            sticky="w"
        )

        selecteur_voisins = ttk.Combobox(
            bloc_options,
            textvariable=self.var_mode_voisins_graphe,
            values=["Aucun", "Top 3", "Top 5"],
            state="readonly",
            width=12
        )

        selecteur_voisins.grid(
            row=1,
            column=1,
            pady=(6, 0),
            sticky="w"
        )

        self.var_mode_voisins_graphe.trace_add(
            "write",
            self._sur_changement_voisins_graphe
        )

        self.progress = ttk.Progressbar(
            frame,
            mode="indeterminate"
        )

        self.progress.grid(
            row=3,
            column=0,
            sticky="ew",
            pady=10
        )

        barre_actions = ttk.Frame(
            frame
        )

        barre_actions.grid(
            row=4,
            column=0,
            sticky="w",
            pady=10
        )

        ttk.Button(
            barre_actions,
            text="Lancer Analyse",
            command=self.lancer_analyse_thread
        ).pack(
            side="left",
            padx=(0, 8)
        )

        ttk.Button(
            barre_actions,
            text="Ouvrir graphiques",
            command=self.creer_fenetre_graphes
        ).pack(
            side="left",
            padx=(0, 8)
        )

        ttk.Button(
            barre_actions,
            text="Exporter CSV",
            command=self.exporter_resultats_csv
        ).pack(
            side="left",
            padx=(0, 8)
        )

        ttk.Button(
            barre_actions,
            text="Effacer cache modèles",
            command=self.effacer_cache_modeles
        ).pack(
            side="left",
            padx=(0, 8)
        )

        ttk.Button(
            barre_actions,
            text="Quitter",
            command=self.destroy
        ).pack(
            side="left"
        )

        conteneur_resultats = ttk.LabelFrame(
            frame,
            text="Résultats",
            padding=10
        )

        conteneur_resultats.grid(
            row=5,
            column=0,
            sticky="nsew"
        )

        self.zone_resultats = tk.Text(
            conteneur_resultats,
            height=14
        )

        self.zone_resultats.pack(
            fill="both",
            expand=True
        )

        frame.rowconfigure(
            5,
            weight=1
        )

    def creer_fenetre_graphes(self):
        if (
            self.fenetre_graphes is not None
            and self.fenetre_graphes.winfo_exists()
        ):
            self.fenetre_graphes.lift()
            return

        self.fenetre_graphes = tk.Toplevel(self)
        self.fenetre_graphes.title(
            "Graphiques"
        )
        self.fenetre_graphes.geometry(
            "1100x780"
        )
        self.fenetre_graphes.protocol(
            "WM_DELETE_WINDOW",
            self.fermer_fenetre_graphes
        )

        conteneur_graphes = ttk.Frame(
            self.fenetre_graphes,
            padding=10
        )
        conteneur_graphes.pack(
            fill="both",
            expand=True
        )

        self.onglets_graphes = ttk.Notebook(
            conteneur_graphes
        )
        self.onglets_graphes.pack(
            fill="both",
            expand=True
        )

        self.frames_graphes = {}
        self.graph_canvases = {}

        for identifiant, titre in [
            ("trajectoire", "Trajectoire"),
            ("global", "Distance globale"),
            ("temporel", "Distance temporelle")
        ]:
            onglet = ttk.Frame(
                self.onglets_graphes
            )
            onglet.columnconfigure(
                0,
                weight=1
            )
            onglet.rowconfigure(
                1,
                weight=1
            )
            self.onglets_graphes.add(
                onglet,
                text=titre
            )
            barre = ttk.Frame(
                onglet
            )
            barre.grid(
                row=0,
                column=0,
                sticky="ew",
                pady=(0, 6)
            )

            zone_canvas = ttk.Frame(
                onglet
            )
            zone_canvas.grid(
                row=1,
                column=0,
                sticky="nsew"
            )

            self.frames_graphes[identifiant] = {
                "onglet": onglet,
                "barre": barre,
                "canvas": zone_canvas
            }

        self._initialiser_graphes_vides()

    def fermer_fenetre_graphes(self):
        if (
            self.fenetre_graphes is not None
            and self.fenetre_graphes.winfo_exists()
        ):
            self.fenetre_graphes.destroy()
        self.fenetre_graphes = None
        self.frames_graphes = {}
        self.graph_canvases = {}

    # Choix dossier
    def choisir_dossier(self):
        dossier = filedialog.askdirectory()

        if dossier:

            self.var_dossier.set(
                dossier
            )

    # Thread
    def lancer_analyse_thread(self):
        if self.analyse_en_cours:
            messagebox.showinfo(
                "Analyse en cours",
                "Patientez jusqu'à la fin de l'analyse actuelle."
            )
            return

        self.analyse_en_cours = True
        self.progress.start()
        self.zone_resultats.delete("1.0", tk.END)

        thread = threading.Thread(
            target=self.executer_analyse
        )

        thread.daemon = True

        thread.start()

    # Méthode pour insérer du texte thread-safe
    def _inserer_texte(self, texte):
        """Insère du texte dans la zone de résultats de manière thread-safe."""
        self.zone_resultats.insert(tk.END, texte)
        self.zone_resultats.see(tk.END)

    def _afficher_figure(self, identifiant, figure):
        if (
            self.fenetre_graphes is None
            or not self.fenetre_graphes.winfo_exists()
        ):
            self.creer_fenetre_graphes()

        elements = self.frames_graphes[identifiant]
        frame_canvas = elements["canvas"]
        frame_barre = elements["barre"]

        ancien_canvas = self.graph_canvases.get(
            identifiant
        )

        if ancien_canvas is not None:
            ancien_canvas.get_tk_widget().destroy()

        for enfant in frame_barre.winfo_children():
            enfant.destroy()

        canvas = FigureCanvasTkAgg(
            figure,
            master=frame_canvas
        )

        widget = canvas.get_tk_widget()
        widget.pack(
            fill="both",
            expand=True
        )

        toolbar = NavigationToolbar2Tk(
            canvas,
            frame_barre,
            pack_toolbar=False
        )
        toolbar.update()
        toolbar.pack(
            side="left"
        )

        ttk.Button(
            frame_barre,
            text="Réinitialiser la vue",
            command=toolbar.home
        ).pack(
            side="left",
            padx=(8, 0)
        )

        canvas.draw()
        self.graph_canvases[identifiant] = canvas

    def _initialiser_graphes_vides(self):
        for identifiant in self.frames_graphes:
            self._afficher_figure(
                identifiant,
                creer_figure_vide(
                    "Lancez une analyse pour afficher un graphique."
                )
            )

    def _terminer_analyse(self):
        self.analyse_en_cours = False
        self.progress.stop()

    def _terminer_verification(self):
        self.verification_en_cours = False

    def _nombre_voisins_graphe(self):
        mode = self.var_mode_voisins_graphe.get()

        if mode == "Top 3":
            return 3

        if mode == "Top 5":
            return 5

        return 0

    def _sur_changement_voisins_graphe(self, *_args):
        if self.dernier_resultat is None:
            return

        self._mettre_a_jour_graphique_trajectoire()

    def _mettre_a_jour_graphique_trajectoire(self):
        if self.dernier_resultat is None:
            return

        resultat = self.dernier_resultat
        nb_voisins = self._nombre_voisins_graphe()

        figure_trajectoire = creer_figure_trajectoire(
            resultat["mot"],
            resultat["vecteurs"],
            resultat["labels"],
            voisins=resultat.get("voisins_vecteurs"),
            nombre_voisins=nb_voisins
        )

        self._afficher_figure(
            "trajectoire",
            figure_trajectoire
        )

    def _mettre_a_jour_graphiques(self):
        if self.dernier_resultat is None:
            return

        resultat = self.dernier_resultat

        self._mettre_a_jour_graphique_trajectoire()

        figure_global = creer_figure_distances_global(
            resultat["mot"],
            resultat["distances"]
        )

        figure_temporelle = creer_figure_distances_temporelles(
            resultat["mot"],
            resultat["distances_temp"]
        )

        self._afficher_figure(
            "global",
            figure_global
        )

        self._afficher_figure(
            "temporel",
            figure_temporelle
        )

    def _repertoire_modeles(self):
        base = os.path.join(
            os.getcwd(),
            "models"
        )
        os.makedirs(
            base,
            exist_ok=True
        )
        return base

    def effacer_cache_modeles(self):
        if self.analyse_en_cours:
            messagebox.showinfo(
                "Cache modèles",
                "Attendez la fin de l'analyse avant d'effacer le cache."
            )
            return

        confirmer = messagebox.askyesno(
            "Effacer le cache",
            "Supprimer tous les modèles sauvegardés dans le dossier 'models' ?"
        )

        if not confirmer:
            return

        dossier_modeles = self._repertoire_modeles()

        if os.path.exists(dossier_modeles):
            for nom in os.listdir(dossier_modeles):
                chemin = os.path.join(
                    dossier_modeles,
                    nom
                )

                if os.path.isdir(chemin):
                    shutil.rmtree(chemin)
                else:
                    os.remove(chemin)

        self.modeles_cache = None
        self.dernier_resultat = None
        self._inserer_texte(
            "Cache des modèles effacé.\n"
        )

        messagebox.showinfo(
            "Cache modèles",
            "Le cache des modèles a été effacé."
        )

    def _signature_modeles_dict(self, dossier):
        return {
            "dossier": os.path.abspath(dossier),
            "stopwords": self.var_stopwords.get(),
            "lemmatisation": self.var_lemmatisation.get(),
            "vector_size": self.var_vector_size.get(),
            "window": self.var_window.get(),
            "min_count": self.var_min_count.get(),
            "epochs": self.var_epochs.get(),
            "min_documents": self.var_min_documents.get(),
            "min_common_words": self.var_min_common_words.get()
        }

    def _dossier_cache_modeles(self, dossier):
        signature = self._signature_modeles_dict(
            dossier
        )
        signature_json = json.dumps(
            signature,
            sort_keys=True
        )
        cle = hashlib.sha1(
            signature_json.encode("utf-8")
        ).hexdigest()[:12]
        return os.path.join(
            self._repertoire_modeles(),
            f"cache_{cle}"
        )

    def _signature_corpus(self, dossier):
        return (
            dossier,
            self.var_stopwords.get(),
            self.var_lemmatisation.get()
        )

    def _signature_occurrences(self, dossier):
        return dossier

    def _signature_modeles(self, dossier):
        return (
            dossier,
            self.var_stopwords.get(),
            self.var_lemmatisation.get(),
            self.var_vector_size.get(),
            self.var_window.get(),
            self.var_min_count.get(),
            self.var_epochs.get(),
            self.var_min_documents.get(),
            self.var_min_common_words.get()
        )

    def _finaliser_analyse(self, resultat):
        mot = resultat["mot"]
        vecteurs = resultat["vecteurs"]
        labels = resultat["labels"]
        voisins = resultat["voisins"]
        distances = resultat["distances"]
        distances_temp = resultat["distances_temp"]
        message = resultat.get("message_cache")
        self.dernier_resultat = resultat

        self._terminer_analyse()

        if message:
            self._inserer_texte(f"{message}\n")

        self._inserer_texte(
            "Mise à jour des graphiques...\n"
        )

        self._mettre_a_jour_graphiques()

        self._inserer_texte("\nVoisins par décennie:\n")

        for decennie in sorted(voisins):
            self._inserer_texte(f"\n{decennie}:\n")
            for voisin, score in voisins[decennie]:
                self._inserer_texte(f"  {voisin}: {score:.3f}\n")

        self._inserer_texte("\nDistances au modèle global:\n")

        for decennie in sorted(distances):
            self._inserer_texte(
                f"{decennie}: {distances[decennie]:.3f}\n"
            )

        self._inserer_texte("\nDistances temporelles:\n")

        for periode, distance in distances_temp.items():
            debut, fin = periode
            self._inserer_texte(
                f"{debut}-{fin}: {distance:.3f}\n"
            )

        self._inserer_texte(
            f"\nAnalyse terminée pour : {mot}\n"
        )

    def exporter_resultats_csv(self):
        if self.dernier_resultat is None:
            messagebox.showinfo(
                "Export CSV",
                "Lancez d'abord une analyse."
            )
            return

        chemin = filedialog.asksaveasfilename(
            title="Exporter les résultats",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv")]
        )

        if not chemin:
            return

        resultat = self.dernier_resultat

        with open(
            chemin,
            "w",
            encoding="utf-8",
            newline=""
        ) as f:

            writer = csv.writer(f)
            writer.writerow([
                "section",
                "periode",
                "valeur_1",
                "valeur_2"
            ])

            for decennie, voisins in sorted(
                resultat["voisins"].items()
            ):
                for voisin, score in voisins:
                    writer.writerow([
                        "voisins",
                        decennie,
                        voisin,
                        f"{score:.6f}"
                    ])

            for decennie, distance in sorted(
                resultat["distances"].items()
            ):
                writer.writerow([
                    "distance_global",
                    decennie,
                    f"{distance:.6f}",
                    ""
                ])

            for periode, distance in resultat[
                "distances_temp"
            ].items():
                debut, fin = periode
                writer.writerow([
                    "distance_temporelle",
                    f"{debut}-{fin}",
                    f"{distance:.6f}",
                    ""
                ])

        messagebox.showinfo(
            "Export CSV",
            f"Résultats exportés dans :\n{chemin}"
        )

    def verifier_mot_thread(self):
        if self.verification_en_cours:
            messagebox.showinfo(
                "Vérification en cours",
                "Patientez jusqu'à la fin de la vérification actuelle."
            )
            return

        dossier = self.var_dossier.get()
        mot = self.var_mot.get().strip()

        if not dossier:
            messagebox.showerror(
                "Erreur",
                "Choisissez un dossier."
            )
            return

        if not mot:
            messagebox.showerror(
                "Erreur",
                "Saisissez un mot à vérifier."
            )
            return

        self.verification_en_cours = True
        self.var_assistant_mot.set(
            "Assistant corpus : vérification en cours..."
        )

        thread = threading.Thread(
            target=self.verifier_mot_corpus
        )

        thread.daemon = True
        thread.start()

    def verifier_mot_corpus(self):
        dossier = self.var_dossier.get()
        mot = self.var_mot.get().strip()
        mot_normalise = normaliser_mot_recherche(mot)

        if mot_normalise is None:
            self.after(
                0,
                lambda: self.var_assistant_mot.set(
                    "Assistant corpus : saisissez un seul mot valide."
                )
            )
            self.after(0, self._terminer_verification)
            return

        try:
            signature_occurrences = self._signature_occurrences(
                dossier
            )

            if (
                self.occurrences_cache is not None
                and self.occurrences_cache["signature"] == signature_occurrences
            ):
                index_occurrences = self.occurrences_cache["index"]
            else:
                index_occurrences = indexer_occurrences_corpus(
                    dossier
                )

                self.occurrences_cache = {
                    "signature": signature_occurrences,
                    "index": index_occurrences
                }

            compteur = index_occurrences["compteur"]
            documents_par_mot = index_occurrences["documents_par_mot"]
            decennies_par_mot = index_occurrences["decennies_par_mot"]
            occurrences_par_decennie = index_occurrences["occurrences_par_decennie"]
            statistiques = index_occurrences["statistiques"]

            occurrences = compteur.get(
                mot_normalise,
                0
            )

            nb_documents = documents_par_mot.get(
                mot_normalise,
                0
            )

            decennies = decennies_par_mot.get(
                mot_normalise,
                []
            )

            detail_occurrences = occurrences_par_decennie.get(
                mot_normalise,
                {}
            )

            if occurrences:
                texte_decennies = ", ".join(
                    str(decennie)
                    for decennie in decennies
                )

                texte_details = ", ".join(
                    f"{decennie}: {nb}"
                    for decennie, nb in sorted(
                        detail_occurrences.items()
                    )
                )

                message = (
                    f"Assistant corpus : '{mot_normalise}' est present "
                    f"{occurrences} fois dans {nb_documents} document(s). "
                    f"Decennies : {texte_decennies or 'non detectees'}. "
                    f"Detail : {texte_details or 'non disponible'}."
                )
            else:
                message = (
                    f"Assistant corpus : '{mot_normalise}' est absent du corpus "
                    f"indexe ({statistiques['documents']} document(s) scannes)."
                )

            self.after(
                0,
                lambda m=message: self.var_assistant_mot.set(m)
            )

        except Exception as e:
            self.after(
                0,
                lambda err=str(e): messagebox.showerror(
                    "Erreur",
                    err
                )
            )
            self.after(
                0,
                lambda: self.var_assistant_mot.set(
                    "Assistant corpus : verification impossible."
                )
            )

        finally:
            self.after(0, self._terminer_verification)

    # Analyse
    def executer_analyse(self):
        dossier = self.var_dossier.get()
        mot = self.var_mot.get().strip()

        if not dossier:
            self.after(0, lambda: messagebox.showerror("Erreur", "Choisissez un dossier."))
            self.after(0, self._terminer_analyse)
            return

        if not mot:
            self.after(0, lambda: messagebox.showerror("Erreur", "Saisissez un mot."))
            self.after(0, self._terminer_analyse)
            return

        try:
            # Configurer les options NLP
            import corpus
            corpus.SUPPRIMER_STOPWORDS = self.var_stopwords.get()
            corpus.UTILISER_LEMMATISATION = self.var_lemmatisation.get()
            signature_corpus = self._signature_corpus(dossier)
            signature_modeles = self._signature_modeles(dossier)
            signature_modeles_dict = self._signature_modeles_dict(dossier)
            message_cache = None

            if (
                self.corpus_cache is not None
                and self.corpus_cache["signature"] == signature_corpus
            ):
                corpus_global = self.corpus_cache["corpus_global"]
                corpus_decennies = self.corpus_cache["corpus_decennies"]
                self.after(
                    0,
                    lambda: self._inserer_texte(
                        "Corpus déjà chargé, réutilisation...\n"
                    )
                )
            else:
                self.after(0, lambda: self._inserer_texte("Chargement du corpus...\n"))

                corpus_global, corpus_decennies = charger_corpus(dossier)

                self.corpus_cache = {
                    "signature": signature_corpus,
                    "corpus_global": corpus_global,
                    "corpus_decennies": corpus_decennies
                }
            
            if not corpus_global:
                self.after(0, lambda: messagebox.showerror("Erreur", "Le corpus est vide."))
                self.after(0, self._terminer_analyse)
                return

            if (
                self.modeles_cache is not None
                and self.modeles_cache["signature"] == signature_modeles
            ):
                modele_global = self.modeles_cache["modele_global"]
                modeles_decennies = self.modeles_cache["modeles_decennies"]
                rotations = self.modeles_cache["rotations"]
                message_cache = (
                    "Modèles déjà entraînés, réutilisation pour ce nouveau mot."
                )
                self.after(
                    0,
                    lambda: self._inserer_texte(
                        "Modèles déjà entraînés, réutilisation...\n"
                    )
                )
            else:
                dossier_modeles = self._dossier_cache_modeles(
                    dossier
                )
                metadata_valide = False

                if os.path.exists(
                    os.path.join(
                        dossier_modeles,
                        "global.model"
                    )
                ):
                    self.after(
                        0,
                        lambda: self._inserer_texte(
                            "Chargement des modèles sauvegardés...\n"
                        )
                    )

                    (
                        modele_global,
                        modeles_decennies,
                        metadata
                    ) = charger_modeles(
                        dossier_modeles
                    )

                    metadata_valide = (
                        metadata.get("signature")
                        == signature_modeles_dict
                    )

                if metadata_valide:
                    self.after(
                        0,
                        lambda: self._inserer_texte(
                            "Construction des rotations depuis les modèles chargés...\n"
                        )
                    )

                    rotations = construire_rotations(
                        modele_global,
                        modeles_decennies,
                        min_common_words=self.var_min_common_words.get()
                    )

                    message_cache = (
                        "Modèles rechargés depuis le disque pour ce corpus."
                    )
                else:
                    if os.path.exists(
                        os.path.join(
                            dossier_modeles,
                            "global.model"
                        )
                    ):
                        self.after(
                            0,
                            lambda: self._inserer_texte(
                                "Modèles sauvegardés incompatibles avec les paramètres actuels, nouvel entraînement...\n"
                            )
                        )

                    # Entraîner le modèle global
                    self.after(0, lambda: self._inserer_texte("Entraînement du modèle global...\n"))
                    
                    modele_global = entrainer_modele_global(
                        corpus_global,
                        vector_size=self.var_vector_size.get(),
                        window=self.var_window.get(),
                        min_count=self.var_min_count.get(),
                        epochs=self.var_epochs.get()
                    )
                
                    # Entraîner les modèles par décennie
                    self.after(0, lambda: self._inserer_texte("Entraînement des modèles par décennie...\n"))
                    
                    modeles_decennies = entrainer_modeles_decennies(
                        corpus_decennies,
                        vector_size=self.var_vector_size.get(),
                        window=self.var_window.get(),
                        min_count=self.var_min_count.get(),
                        epochs=self.var_epochs.get(),
                        min_documents=self.var_min_documents.get()
                    )
                    
                    # Construire les rotations
                    self.after(0, lambda: self._inserer_texte("Construction des rotations...\n"))
                    
                    rotations = construire_rotations(
                        modele_global,
                        modeles_decennies,
                        min_common_words=self.var_min_common_words.get()
                    )

                    sauvegarder_modeles(
                        modele_global,
                        modeles_decennies,
                        dossier=dossier_modeles,
                        metadata={
                            "signature": signature_modeles_dict
                        }
                    )

                    self.after(
                        0,
                        lambda: self._inserer_texte(
                            "Modèles sauvegardés pour réutilisation future...\n"
                        )
                    )

                self.modeles_cache = {
                    "signature": signature_modeles,
                    "modele_global": modele_global,
                    "modeles_decennies": modeles_decennies,
                    "rotations": rotations
                }

            # Vérifier que le mot existe
            if mot not in modele_global.wv:
                self.after(0, lambda m=mot: messagebox.showerror(
                    "Erreur",
                    f"Le mot '{m}' n'existe pas dans le vocabulaire."
                ))
                self.after(0, self._terminer_analyse)
                return
            
            if not rotations:
                self.after(0, lambda: messagebox.showwarning(
                    "Avertissement",
                    "Aucune rotation n'a pu être construite. Pas assez de mots communs entre les décennies et le modèle global."
                ))
                self.after(0, self._terminer_analyse)
                return
            
            # Construire la trajectoire
            self.after(0, lambda: self._inserer_texte("Construction de la trajectoire...\n"))
            
            vecteurs, labels = construire_trajectoire(
                mot,
                modele_global,
                modeles_decennies,
                rotations
            )

            voisins = most_similar_par_decennie(
                mot,
                modele_global,
                modeles_decennies,
                rotations,
                topn=self.var_topn.get()
            )

            voisins_vecteurs = {
                decennie: [
                    (
                        voisin,
                        modele_global.wv[voisin]
                    )
                    for voisin, _score in liste_voisins
                    if voisin in modele_global.wv
                ]
                for decennie, liste_voisins in voisins.items()
            }

            distances = distance_au_global(
                mot,
                modele_global,
                modeles_decennies,
                rotations
            )

            distances_temp = distance_temporelle(
                mot,
                modeles_decennies,
                rotations
            )

            resultat = {
                "mot": mot,
                "vecteurs": vecteurs,
                "labels": labels,
                "voisins": voisins,
                "voisins_vecteurs": voisins_vecteurs,
                "distances": distances,
                "distances_temp": distances_temp,
                "message_cache": message_cache
            }

            self.after(
                0,
                lambda r=resultat: self._finaliser_analyse(r)
            )

        except Exception as e:
            self.after(0, lambda err=str(e): messagebox.showerror("Erreur", err))
            self.after(0, self._terminer_analyse)
            self.after(0, self.progress.stop)
