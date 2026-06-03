from corpus import charger_corpus

from word2vec_manager import (
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
    afficher_trajectoire,
    afficher_distances_global,
    afficher_distances_temporelles
)




import tkinter as tk

from tkinter import ttk
from tkinter import filedialog
from tkinter import messagebox

import threading


class Application(tk.Tk):

    def __init__(self):
        super().__init__()

        self.title(
            "Analyse Diachronique Word2Vec"
        )

        self.geometry(
            "900x700"
        )

        self.creer_interface()

    
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


        # Corpus
        ttk.Label(
            frame,
            text="Dossier Corpus"
        ).grid(
            row=0,
            column=0,
            sticky="w"
        )

        self.var_dossier = tk.StringVar()

        ttk.Entry(
            frame,
            textvariable=self.var_dossier,
            width=60
        ).grid(
            row=0,
            column=1,
            padx=5,
            pady=5
        )

        ttk.Button(
            frame,
            text="Parcourir",
            command=self.choisir_dossier
        ).grid(
            row=0,
            column=2
        )

        # Mot recherché
        ttk.Label(
            frame,
            text="Mot recherché"
        ).grid(
            row=1,
            column=0,
            sticky="w"
        )

        self.var_mot = tk.StringVar()

        ttk.Entry(
            frame,
            textvariable=self.var_mot,
            width=30
        ).grid(
            row=1,
            column=1,
            sticky="w"
        )

        # Paramètres Word2Vec
        ttk.Label(
            frame,
            text="Taille vecteur"
        ).grid(
            row=2,
            column=0,
            sticky="w"
        )

        self.var_vector_size = tk.IntVar(
            value=200
        )

        ttk.Entry(
            frame,
            textvariable=self.var_vector_size
        ).grid(
            row=2,
            column=1,
            sticky="w"
        )

        ttk.Label(
            frame,
            text="Fenêtre"
        ).grid(
            row=3,
            column=0,
            sticky="w"
        )

        self.var_window = tk.IntVar(
            value=5
        )

        ttk.Entry(
            frame,
            textvariable=self.var_window
        ).grid(
            row=3,
            column=1,
            sticky="w"
        )

        ttk.Label(
            frame,
            text="Min Count"
        ).grid(
            row=4,
            column=0,
            sticky="w"
        )

        self.var_min_count = tk.IntVar(
            value=5
        )

        ttk.Entry(
            frame,
            textvariable=self.var_min_count
        ).grid(
            row=4,
            column=1,
            sticky="w"
        )

        ttk.Label(
            frame,
            text="Epochs"
        ).grid(
            row=5,
            column=0,
            sticky="w"
        )

        self.var_epochs = tk.IntVar(
            value=20
        )

        ttk.Entry(
            frame,
            textvariable=self.var_epochs
        ).grid(
            row=5,
            column=1,
            sticky="w"
        )

        # Options NLP
        self.var_stopwords = tk.BooleanVar(
            value=True
        )

        ttk.Checkbutton(
            frame,
            text="Supprimer les stopwords",
            variable=self.var_stopwords
        ).grid(
            row=6,
            column=0,
            columnspan=2,
            sticky="w"
        )

        self.var_lemmatisation = tk.BooleanVar(
            value=True
        )

        ttk.Checkbutton(
            frame,
            text="Lemmatisation",
            variable=self.var_lemmatisation
        ).grid(
            row=7,
            column=0,
            columnspan=2,
            sticky="w"
        )
        
        # Nombre voisins
        ttk.Label(
            frame,
            text="Nombre de voisins"
        ).grid(
            row=8,
            column=0,
            sticky="w"
        )

        self.var_topn = tk.IntVar(
            value=10
        )

        ttk.Entry(
            frame,
            textvariable=self.var_topn
        ).grid(
            row=8,
            column=1,
            sticky="w"
        )

        # Barre progression
        self.progress = ttk.Progressbar(
            frame,
            mode="indeterminate"
        )

        self.progress.grid(
            row=9,
            column=0,
            columnspan=3,
            sticky="ew",
            pady=10
        )

        # Boutons

        ttk.Button(
            frame,
            text="Lancer Analyse",
            command=self.lancer_analyse_thread
        ).grid(
            row=10,
            column=0,
            pady=10
        )

        ttk.Button(
            frame,
            text="Quitter",
            command=self.destroy
        ).grid(
            row=10,
            column=1
        )

        # Zone résultats
        ttk.Label(
            frame,
            text="Résultats"
        ).grid(
            row=11,
            column=0,
            sticky="w"
        )

        self.zone_resultats = tk.Text(
            frame,
            height=20
        )

        self.zone_resultats.grid(
            row=12,
            column=0,
            columnspan=3,
            sticky="nsew"
        )

        frame.rowconfigure(
            12,
            weight=1
        )

    # Choix dossier
    def choisir_dossier(self):
        dossier = filedialog.askdirectory()

        if dossier:

            self.var_dossier.set(
                dossier
            )

    # Thread
    def lancer_analyse_thread(self):
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

    def _finaliser_analyse(self, resultat):
        mot = resultat["mot"]
        vecteurs = resultat["vecteurs"]
        labels = resultat["labels"]
        voisins = resultat["voisins"]
        distances = resultat["distances"]
        distances_temp = resultat["distances_temp"]

        self.progress.stop()

        if vecteurs:
            self._inserer_texte("Affichage de la trajectoire...\n")
            afficher_trajectoire(mot, vecteurs, labels)

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

        afficher_distances_global(mot, distances)

        self._inserer_texte("\nDistances temporelles:\n")

        for periode, distance in distances_temp.items():
            debut, fin = periode
            self._inserer_texte(
                f"{debut}-{fin}: {distance:.3f}\n"
            )

        afficher_distances_temporelles(mot, distances_temp)

        self._inserer_texte(
            f"\nAnalyse terminée pour : {mot}\n"
        )

    # Analyse
    def executer_analyse(self):
        dossier = self.var_dossier.get()
        mot = self.var_mot.get().strip()

        if not dossier:
            self.after(0, lambda: messagebox.showerror("Erreur", "Choisissez un dossier."))
            self.after(0, self.progress.stop)
            return

        if not mot:
            self.after(0, lambda: messagebox.showerror("Erreur", "Saisissez un mot."))
            self.after(0, self.progress.stop)
            return

        try:
            # Configurer les options NLP
            import corpus
            corpus.SUPPRIMER_STOPWORDS = self.var_stopwords.get()
            corpus.UTILISER_LEMMATISATION = self.var_lemmatisation.get()

            # Charger le corpus
            self.after(0, lambda: self._inserer_texte("Chargement du corpus...\n"))
            
            corpus_global, corpus_decennies = charger_corpus(dossier)
            
            if not corpus_global:
                self.after(0, lambda: messagebox.showerror("Erreur", "Le corpus est vide."))
                self.after(0, self.progress.stop)
                return
            
            # Entraîner le modèle global
            self.after(0, lambda: self._inserer_texte("Entraînement du modèle global...\n"))
            
            modele_global = entrainer_modele_global(
                corpus_global,
                vector_size=self.var_vector_size.get(),
                window=self.var_window.get(),
                min_count=self.var_min_count.get(),
                epochs=self.var_epochs.get()
            )
            
            # Vérifier que le mot existe
            if mot not in modele_global.wv:
                self.after(0, lambda m=mot: messagebox.showerror(
                    "Erreur",
                    f"Le mot '{m}' n'existe pas dans le vocabulaire."
                ))
                self.after(0, self.progress.stop)
                return
            
            # Entraîner les modèles par décennie
            self.after(0, lambda: self._inserer_texte("Entraînement des modèles par décennie...\n"))
            
            modeles_decennies = entrainer_modeles_decennies(
                corpus_decennies,
                vector_size=self.var_vector_size.get(),
                window=self.var_window.get(),
                min_count=self.var_min_count.get(),
                epochs=self.var_epochs.get()
            )
            
            # Construire les rotations
            self.after(0, lambda: self._inserer_texte("Construction des rotations...\n"))
            
            rotations = construire_rotations(
                modele_global,
                modeles_decennies
            )
            
            if not rotations:
                self.after(0, lambda: messagebox.showwarning(
                    "Avertissement",
                    "Aucune rotation n'a pu être construite. Pas assez de mots communs entre les décennies et le modèle global."
                ))
                self.after(0, self.progress.stop)
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
                "distances": distances,
                "distances_temp": distances_temp
            }

            self.after(
                0,
                lambda r=resultat: self._finaliser_analyse(r)
            )

        except Exception as e:
            self.after(0, lambda err=str(e): messagebox.showerror("Erreur", err))
            self.after(0, self.progress.stop)
