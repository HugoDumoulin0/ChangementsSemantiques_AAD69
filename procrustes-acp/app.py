"""
app.py — Interface graphique principale
TextSimplifier & Analyse Diachronique

Onglet 1 : Simplification de phrases (affichage côte à côte)
Onglet 2 : Analyse diachronique Word2Vec + Procuste + ACP
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import csv
from pathlib import Path

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from io_loader import load_text_file, get_files_info
from simplifier import simplify_text, detect_language
from diachronie import (
    prepare_diachronic_data,
    plot_diachronic,
    plot_global_comparison,
)


# ── Onglet Simplification 

class SimplificationTab(ttk.Frame):

    def __init__(self, parent: ttk.Notebook):
        super().__init__(parent)
        self._lang_var = tk.StringVar(value='auto')
        self._status_var = tk.StringVar(value="Ouvrez un fichier .txt pour commencer.")
        self._build()

    def _build(self):
        # ── Barre d'outils ────────────────────────────────────────────────
        toolbar = ttk.Frame(self, padding=(8, 6, 8, 4))
        toolbar.pack(fill='x')

        ttk.Button(toolbar, text="Ouvrir fichier...", command=self.open_file,
                   width=16).pack(side='left', padx=(0, 6))

        ttk.Separator(toolbar, orient='vertical').pack(side='left', fill='y', padx=6)

        ttk.Label(toolbar, text="Langue :").pack(side='left')
        lang_cb = ttk.Combobox(
            toolbar, textvariable=self._lang_var,
            values=['auto', 'en', 'fr'], width=7, state='readonly',
        )
        lang_cb.pack(side='left', padx=4)

        ttk.Separator(toolbar, orient='vertical').pack(side='left', fill='y', padx=6)

        ttk.Button(toolbar, text="Simplifier", command=self.run_simplification,
                   width=12).pack(side='left')

        ttk.Label(toolbar, textvariable=self._status_var,
                  foreground='#555555').pack(side='left', padx=12)

        # ── Zone de texte côte à côte ─────────────────────────────────────
        paned = ttk.PanedWindow(self, orient='horizontal')
        paned.pack(fill='both', expand=True, padx=8, pady=(0, 8))

        # Gauche : texte original
        left_frame = ttk.LabelFrame(paned, text="  Texte original  ")
        self._orig_text = tk.Text(
            left_frame, wrap='word',
            font=('Segoe UI', 10), padx=8, pady=8, undo=True,
        )
        left_sb = ttk.Scrollbar(left_frame, command=self._orig_text.yview)
        self._orig_text.configure(yscrollcommand=left_sb.set)
        left_sb.pack(side='right', fill='y')
        self._orig_text.pack(fill='both', expand=True)
        paned.add(left_frame, weight=1)

        # Droite : phrases simplifiées
        right_frame = ttk.LabelFrame(paned, text="  Phrases simples  ")
        self._simple_text = tk.Text(
            right_frame, wrap='word',
            font=('Segoe UI', 10), padx=8, pady=8,
            state='disabled', background='#f5f5f5',
        )
        right_sb = ttk.Scrollbar(right_frame, command=self._simple_text.yview)
        self._simple_text.configure(yscrollcommand=right_sb.set)
        right_sb.pack(side='right', fill='y')
        self._simple_text.pack(fill='both', expand=True)
        paned.add(right_frame, weight=1)

    # ── Actions 

    def open_file(self):
        path = filedialog.askopenfilename(
            title="Ouvrir un fichier texte",
            filetypes=[("Fichiers texte", "*.txt"), ("Tous les fichiers", "*.*")],
        )
        if not path:
            return
        try:
            text = load_text_file(path)
        except Exception as exc:
            messagebox.showerror("Erreur lecture", str(exc))
            return

        self._orig_text.delete('1.0', 'end')
        self._orig_text.insert('1.0', text)

        # Mise à jour du statut avec langue détectée
        detected = detect_language(text)
        lang_label = "Français" if detected == 'fr' else "Anglais"
        fname = Path(path).name
        self._status_var.set(f"{fname}  ·  Langue détectée : {lang_label}")

        # Simplification automatique au chargement
        self.run_simplification()

    def run_simplification(self):
        text = self._orig_text.get('1.0', 'end').strip()
        if not text:
            self._status_var.set("Aucun texte à simplifier.")
            return

        lang_val = self._lang_var.get()
        lang = None if lang_val == 'auto' else lang_val

        self._status_var.set("Simplification en cours…")
        self.update_idletasks()

        def _worker():
            try:
                sentences = simplify_text(text, lang=lang)
                result = '\n'.join(sentences)
                n = len(sentences)
                self.after(0, lambda: self._set_simplified(result, n))
            except OSError as exc:
                self.after(0, lambda: (
                    messagebox.showerror("Modèle manquant", str(exc)),
                    self._status_var.set("Erreur : modèle spaCy manquant."),
                ))
            except Exception as exc:
                self.after(0, lambda: (
                    messagebox.showerror("Erreur", str(exc)),
                    self._status_var.set("Erreur lors de la simplification."),
                ))

        threading.Thread(target=_worker, daemon=True).start()

    def _set_simplified(self, result: str, n: int):
        self._simple_text.configure(state='normal')
        self._simple_text.delete('1.0', 'end')
        self._simple_text.insert('1.0', result)
        self._simple_text.configure(state='disabled')
        self._status_var.set(f"Terminé — {n} phrase(s) générée(s).")


# ── Onglet Diachronie 

class DiachronieTab(ttk.Frame):

    def __init__(self, parent: ttk.Notebook):
        super().__init__(parent)
        self._corpus_dir = tk.StringVar()
        self._target_word = tk.StringVar(value="")
        self._top_n = tk.IntVar(value=10)
        self._min_texts_per_decade = tk.IntVar(value=5)
        self._min_tokens_per_decade = tk.IntVar(value=5000)
        self._min_target_occ_per_decade = tk.IntVar(value=2)
        self._show_anchors = tk.BooleanVar(value=True)
        self._show_legend = tk.BooleanVar(value=True)
        self._compare_global = tk.BooleanVar(value=False)
        self._status_var = tk.StringVar(
            value="Sélectionnez un dossier corpus pour commencer."
        )
        self._periods_var = tk.StringVar(value="Périodes détectées : —")
        self._diagnostic_var = tk.StringVar(value="Diagnostic : —")
        self._detected_years: list[int] = []
        self._last_analysis_data: dict | None = None

        self._canvas_widget: tk.Widget | None = None
        self._toolbar_widget: tk.Widget | None = None
        self._fig: plt.Figure | None = None
        self._fig_cmp: plt.Figure | None = None
        self._plot_notebook: ttk.Notebook | None = None
        self._tab_canvases: dict[str, FigureCanvasTkAgg] = {}

        self._build()

    def _build(self):
        # ── Panneau paramètres 
        params = ttk.LabelFrame(self, text="  Paramètres  ", padding=(10, 6))
        params.pack(fill='x', padx=8, pady=8)

        # Ligne 1 : Dossier corpus
        r1 = ttk.Frame(params)
        r1.pack(fill='x', pady=3)
        ttk.Label(r1, text="Dossier corpus :", width=18, anchor='e').pack(side='left')
        ttk.Entry(r1, textvariable=self._corpus_dir).pack(
            side='left', fill='x', expand=True, padx=4)
        ttk.Button(r1, text="Parcourir…", command=self._browse_corpus,
                   width=12).pack(side='left')

        # Ligne 2 : Mot, top N
        r2 = ttk.Frame(params)
        r2.pack(fill='x', pady=3)

        ttk.Label(r2, text="Mot cible :", width=18, anchor='e').pack(side='left')
        ttk.Entry(r2, textvariable=self._target_word, width=18).pack(
            side='left', padx=(4, 16))

        ttk.Label(r2, text="Mots ancres (N) :").pack(side='left')
        ttk.Spinbox(r2, textvariable=self._top_n, from_=3, to=50,
                    width=6).pack(side='left', padx=(4, 16))

        self._run_btn = ttk.Button(
            r2, text="Analyser", command=self.run_analysis, width=12)
        self._run_btn.pack(side='left')
        self._export_btn = ttk.Button(
            r2,
            text="Exporter CSV",
            command=self._export_metrics_csv,
            width=14,
            state='disabled',
        )
        self._export_btn.pack(side='left', padx=(8, 0))

        # Ligne 3 : Options
        r3 = ttk.Frame(params)
        r3.pack(fill='x', pady=(2, 2))

        ttk.Label(r3, text="Granularité :", width=18, anchor='e').pack(side='left')
        ttk.Label(r3, text="Décennie (fixe)", foreground='#555555').pack(
            side='left', padx=(4, 16)
        )

        ttk.Checkbutton(
            r3,
            text="Afficher les mots ancres",
            variable=self._show_anchors,
        ).pack(side='left', padx=(0, 12))

        ttk.Checkbutton(
            r3,
            text="Afficher la légende",
            variable=self._show_legend,
        ).pack(side='left', padx=(0, 12))

        ttk.Checkbutton(
            r3,
            text="Afficher la courbe de similarité vs global",
            variable=self._compare_global,
        ).pack(side='left')

        # Ligne 3b : Seuils qualité (décennies)
        r3b = ttk.Frame(params)
        r3b.pack(fill='x', pady=(2, 2))
        ttk.Label(r3b, text="Seuils qualité :", width=18, anchor='e').pack(side='left')
        ttk.Label(r3b, text="Textes min").pack(side='left', padx=(4, 4))
        ttk.Spinbox(
            r3b,
            textvariable=self._min_texts_per_decade,
            from_=1,
            to=500,
            width=6,
        ).pack(side='left', padx=(0, 12))

        ttk.Label(r3b, text="Tokens min").pack(side='left', padx=(0, 4))
        ttk.Spinbox(
            r3b,
            textvariable=self._min_tokens_per_decade,
            from_=100,
            to=1000000,
            increment=500,
            width=8,
        ).pack(side='left', padx=(0, 12))

        ttk.Label(r3b, text="Occ. mot min").pack(side='left', padx=(0, 4))
        ttk.Spinbox(
            r3b,
            textvariable=self._min_target_occ_per_decade,
            from_=1,
            to=500,
            width=6,
        ).pack(side='left')

        # Ligne 4 : Statut
        r4 = ttk.Frame(params)
        r4.pack(fill='x', pady=(2, 0))
        ttk.Label(r4, textvariable=self._status_var, foreground='#555555').pack(
            side='left')

        r6 = ttk.Frame(params)
        r6.pack(fill='x', pady=(0, 2))
        ttk.Label(r6, textvariable=self._diagnostic_var, foreground='#666666').pack(
            side='left'
        )

        # Ligne 5 : Liste des périodes détectées (scrollable)
        r5 = ttk.Frame(params)
        r5.pack(fill='x', pady=(4, 2))
        ttk.Label(r5, text="Périodes détectées :", width=18, anchor='e').pack(
            side='left', padx=(0, 4)
        )
        periods_container = ttk.Frame(r5)
        periods_container.pack(side='left', fill='x', expand=True)

        self._periods_text = tk.Text(
            periods_container,
            height=2,
            wrap='word',
            font=('Segoe UI', 9),
            foreground='#444444',
            background='#fafafa',
            relief='solid',
            borderwidth=1,
            padx=6,
            pady=4,
        )
        self._periods_text.pack(side='left', fill='x', expand=True)
        self._periods_text.configure(state='disabled')

        periods_sb = ttk.Scrollbar(
            periods_container,
            orient='vertical',
            command=self._periods_text.yview,
        )
        periods_sb.pack(side='right', fill='y')
        self._periods_text.configure(yscrollcommand=periods_sb.set)

        # ── Zone graphique ────────────────────────────────────────────────
        self._plot_frame = ttk.LabelFrame(self, text="  Graphique  ")
        self._plot_frame.pack(fill='both', expand=True, padx=8, pady=(0, 8))

        self._placeholder = ttk.Label(
            self._plot_frame,
            text=(
                "Le graphique s'affichera ici après l'analyse.\n\n"
                "Format des fichiers attendu :  typedefichier_année_id.txt\n"
                "Exemple : mag_1920_4569.txt"
            ),
            foreground='#888888', font=('Segoe UI', 11), justify='center',
        )
        self._placeholder.pack(expand=True)

    # ── Actions ───────────────────────────────────────────────────────────

    def _browse_corpus(self):
        path = filedialog.askdirectory(title="Sélectionner le dossier corpus")
        if not path:
            return
        self._corpus_dir.set(path)
        self._scan_corpus(path)

    def _scan_corpus(self, path: str):
        try:
            files = get_files_info(path)
        except Exception as exc:
            messagebox.showerror("Erreur", str(exc))
            return

        years = sorted({f['year'] for f in files if f['year'] is not None})
        self._detected_years = years
        n_total = len(files)
        n_with_year = len([f for f in files if f['year'] is not None])

        if years:
            self._refresh_reference_values()
            self._status_var.set(
                f"{n_total} fichiers  ·  {n_with_year} avec année détectée"
            )
        else:
            self._status_var.set(
                f"{n_total} fichiers trouvés, mais aucune année détectée "
                f"dans les noms (format attendu : type_année_id.txt)."
            )
            self._set_periods_text("Aucune période détectée.")

    def _refresh_reference_values(self):
        years = self._detected_years
        if not years:
            self._set_periods_text("Aucune période détectée.")
            return

        periods = sorted({(y // 10) * 10 for y in years})
        label = "Décennies"
        values = ', '.join(str(p) for p in periods)
        self._set_periods_text(f"{label} ({len(periods)}) :\n{values}")

    def _set_periods_text(self, content: str):
        self._periods_text.configure(state='normal')
        self._periods_text.delete('1.0', 'end')
        self._periods_text.insert('1.0', content)
        self._periods_text.configure(state='disabled')

    def _set_status(self, message: str):
        self._status_var.set(message)

    def _update_diagnostics(self, analysis_data: dict):
        diagnostics = analysis_data.get('diagnostics') or {}
        kept = diagnostics.get('period_kept_after_alignment')
        dropped = len(diagnostics.get('dropped_periods', []))
        quality_dropped = len(diagnostics.get('quality_dropped_periods', []))
        coverage = diagnostics.get('target_coverage')
        anchors_count = diagnostics.get('anchors_count')
        thresholds = diagnostics.get('quality_filter_thresholds') or {}
        min_texts = thresholds.get('min_texts_per_decade')
        min_tokens = thresholds.get('min_tokens_per_decade')
        min_target_occ = thresholds.get('min_target_occurrences_per_decade')
        if kept is None:
            self._diagnostic_var.set("Diagnostic : —")
            return
        self._diagnostic_var.set(
            f"Diagnostic : décennies gardées={kept}, exclues qualité={quality_dropped}, "
            f"exclues alignement={dropped}, "
            f"couverture mot={coverage}, ancres={anchors_count}, "
            f"seuils=({min_texts}/{min_tokens}/{min_target_occ})"
        )

    def _export_metrics_csv(self):
        if not self._last_analysis_data:
            messagebox.showwarning("Export CSV", "Aucune analyse disponible à exporter.")
            return

        rows = self._last_analysis_data.get('metrics_rows') or []
        if not rows:
            messagebox.showwarning("Export CSV", "Aucune métrique à exporter.")
            return

        default_name = f"diachronie_metrics_{self._last_analysis_data.get('target_word', 'mot')}.csv"
        path = filedialog.asksaveasfilename(
            title="Exporter les métriques diachroniques",
            defaultextension=".csv",
            initialfile=default_name,
            filetypes=[("CSV", "*.csv")],
        )
        if not path:
            return

        fieldnames = ['period', 'is_reference', 'cosine_vs_prev', 'cosine_vs_global']
        try:
            with open(path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for row in rows:
                    writer.writerow(row)
            messagebox.showinfo("Export CSV", f"Export terminé :\n{path}")
        except OSError as exc:
            messagebox.showerror("Export CSV", f"Impossible d'écrire le fichier :\n{exc}")

    def run_analysis(self):
        corpus_dir = self._corpus_dir.get().strip()
        target = self._target_word.get().strip()
        top_n = self._top_n.get()
        show_anchors = self._show_anchors.get()
        show_legend = self._show_legend.get()
        time_granularity = 'decade'
        compare_global = self._compare_global.get()
        min_texts_per_decade = self._min_texts_per_decade.get()
        min_tokens_per_decade = self._min_tokens_per_decade.get()
        min_target_occ_per_decade = self._min_target_occ_per_decade.get()

        if not corpus_dir:
            messagebox.showwarning("Paramètre manquant",
                                   "Veuillez sélectionner un dossier corpus.")
            return
        if not target:
            messagebox.showwarning("Paramètre manquant",
                                   "Veuillez saisir un mot cible.")
            return
        if min_texts_per_decade < 1 or min_tokens_per_decade < 1 or min_target_occ_per_decade < 1:
            messagebox.showwarning(
                "Paramètre invalide",
                "Les seuils qualité doivent être des entiers strictement positifs.",
            )
            return

        self._run_btn.configure(state='disabled')
        self._export_btn.configure(state='disabled')
        self._last_analysis_data = None
        self._diagnostic_var.set("Diagnostic : calcul en cours…")
        self._status_var.set("Analyse en cours… (Word2Vec → Procuste → ACP)")
        self.update_idletasks()

        def _progress(msg: str):
            self.after(0, lambda m=msg: self._set_status(m))

        def _worker():
            try:
                analysis_data = prepare_diachronic_data(
                    corpus_dir=corpus_dir,
                    target_word=target,
                    reference_year=None,
                    top_n_anchors=top_n,
                    time_granularity=time_granularity,
                    train_global_model_option=True,
                    compare_with_global=compare_global,
                    progress_callback=_progress,
                    w2v_params={
                        'min_texts_per_decade': min_texts_per_decade,
                        'min_tokens_per_decade': min_tokens_per_decade,
                        'min_target_occurrences_per_decade': min_target_occ_per_decade,
                    },
                )
                self.after(
                    0,
                    lambda data=analysis_data, show=show_anchors, legend=show_legend,
                    cmp=compare_global: self._finalize_analysis(
                        data,
                        show,
                        legend,
                        cmp,
                    ),
                )
            except Exception as exc:
                err_msg = str(exc)
                self.after(0, lambda msg=err_msg: (
                    messagebox.showerror("Erreur d'analyse", msg),
                    self._status_var.set(f"Erreur : {msg}"),
                ))
            finally:
                pass

        threading.Thread(target=_worker, daemon=True).start()

    def _finalize_analysis(
        self,
        analysis_data: dict,
        show_anchors: bool,
        show_legend: bool,
        compare_global: bool,
    ):
        """Construit la figure matplotlib dans le thread principal Tkinter."""
        try:
            fig_acp = plot_diachronic(
                models=analysis_data['models'],
                rotations=analysis_data['rotations'],
                years=analysis_data['years'],
                reference_year=analysis_data['reference_year'],
                target_word=analysis_data['target_word'],
                anchor_words=analysis_data['anchor_words'],
                show_anchors=show_anchors,
                show_legend=show_legend,
                global_target_vec=analysis_data.get('global_target_vec'),
                global_similarity_by_period=analysis_data.get('global_similarity_by_period'),
                period_label=analysis_data.get('period_label', 'année'),
            )

            show_tabs = (
                compare_global
                and bool(analysis_data.get('global_similarity_by_period'))
            )
            if show_tabs:
                fig_cmp = plot_global_comparison(
                    analysis_data['global_similarity_by_period'],
                    period_label=analysis_data.get('period_label', 'année'),
                )
                self._show_plot_tabs(fig_acp, fig_cmp)
            else:
                self._show_plot(fig_acp)

            self._status_var.set("Analyse terminée.")
            self._last_analysis_data = analysis_data
            self._update_diagnostics(analysis_data)
            if analysis_data.get('metrics_rows'):
                self._export_btn.configure(state='normal')
        except Exception as exc:
            messagebox.showerror("Erreur d'analyse", str(exc))
            self._status_var.set(f"Erreur : {exc}")
            self._diagnostic_var.set("Diagnostic : échec de l'analyse")
        finally:
            self._run_btn.configure(state='normal')

    def _clear_plot_area(self):
        if self._canvas_widget is not None:
            self._canvas_widget.destroy()
            self._canvas_widget = None
        if self._toolbar_widget is not None:
            self._toolbar_widget.destroy()
            self._toolbar_widget = None
        if self._plot_notebook is not None:
            self._plot_notebook.destroy()
            self._plot_notebook = None
        self._tab_canvases.clear()
        if self._placeholder.winfo_exists():
            self._placeholder.destroy()

        if self._fig is not None:
            plt.close(self._fig)
            self._fig = None
        if self._fig_cmp is not None:
            plt.close(self._fig_cmp)
            self._fig_cmp = None

    def _set_toolbar_for_canvas(self, canvas: FigureCanvasTkAgg):
        if self._toolbar_widget is not None:
            self._toolbar_widget.destroy()
            self._toolbar_widget = None

        toolbar_frame = ttk.Frame(self._plot_frame)
        toolbar_frame.pack(side='bottom', fill='x')
        nav = NavigationToolbar2Tk(canvas, toolbar_frame)
        nav.update()
        self._toolbar_widget = toolbar_frame

    def _show_plot(self, fig: plt.Figure):
        """Affiche la figure matplotlib dans l'onglet."""
        self._clear_plot_area()
        self._fig = fig

        # Canvas matplotlib
        canvas = FigureCanvasTkAgg(fig, master=self._plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)
        self._canvas_widget = canvas.get_tk_widget()
        self._set_toolbar_for_canvas(canvas)

    def _show_plot_tabs(self, fig_acp: plt.Figure, fig_cmp: plt.Figure):
        """Affiche les figures dans deux onglets: ACP et comparaison globale."""
        self._clear_plot_area()
        self._fig = fig_acp
        self._fig_cmp = fig_cmp

        nb = ttk.Notebook(self._plot_frame)
        nb.pack(fill='both', expand=True)
        self._plot_notebook = nb

        tab_acp = ttk.Frame(nb)
        tab_cmp = ttk.Frame(nb)
        nb.add(tab_acp, text="Graphique ACP")
        nb.add(tab_cmp, text="Comparaison globale")

        canvas_acp = FigureCanvasTkAgg(fig_acp, master=tab_acp)
        canvas_acp.draw()
        canvas_acp.get_tk_widget().pack(fill='both', expand=True)

        canvas_cmp = FigureCanvasTkAgg(fig_cmp, master=tab_cmp)
        canvas_cmp.draw()
        canvas_cmp.get_tk_widget().pack(fill='both', expand=True)

        self._tab_canvases[str(tab_acp)] = canvas_acp
        self._tab_canvases[str(tab_cmp)] = canvas_cmp

        self._set_toolbar_for_canvas(canvas_acp)

        def _on_tab_changed(_event):
            selected = nb.select()
            current_canvas = self._tab_canvases.get(selected)
            if current_canvas is not None:
                self._set_toolbar_for_canvas(current_canvas)

        nb.bind("<<NotebookTabChanged>>", _on_tab_changed)


# ── Application principale ────────────────────────────────────────────────────

class App(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("TextSimplifier  ·  Analyse Diachronique")
        self.geometry("1280x780")
        self.minsize(960, 620)
        self._configure_style()
        self._build_menu()
        self._build_notebook()

    def _configure_style(self):
        style = ttk.Style(self)
        # Choisir le thème le plus propre disponible
        available = style.theme_names()
        for preferred in ('clam', 'alt', 'default'):
            if preferred in available:
                style.theme_use(preferred)
                break
        style.configure('TNotebook.Tab', padding=[14, 7], font=('Segoe UI', 10))
        style.configure('TLabelframe.Label', font=('Segoe UI', 9, 'bold'))

    def _build_menu(self):
        menubar = tk.Menu(self)

        file_menu = tk.Menu(menubar, tearoff=False)
        file_menu.add_command(
            label="Ouvrir un fichier texte…",
            command=self._open_file_menu, accelerator="Ctrl+O",
        )
        file_menu.add_separator()
        file_menu.add_command(label="Quitter", command=self.destroy,
                              accelerator="Ctrl+Q")
        menubar.add_cascade(label="Fichier", menu=file_menu)
        self.bind_all('<Control-o>', lambda _: self._open_file_menu())
        self.bind_all('<Control-q>', lambda _: self.destroy())

        help_menu = tk.Menu(menubar, tearoff=False)
        help_menu.add_command(label="Comment préparer le corpus…",
                              command=self._show_corpus_help)
        help_menu.add_command(label="À propos", command=self._show_about)
        menubar.add_cascade(label="Aide", menu=help_menu)

        self.config(menu=menubar)

    def _build_notebook(self):
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill='both', expand=True, padx=6, pady=6)

        self.simplif_tab = SimplificationTab(self.notebook)
        self.diachronie_tab = DiachronieTab(self.notebook)

        self.notebook.add(self.simplif_tab, text="   Simplification   ")
        self.notebook.add(self.diachronie_tab, text="   Analyse Diachronique   ")

    def _open_file_menu(self):
        self.notebook.select(0)
        self.simplif_tab.open_file()

    def _show_corpus_help(self):
        messagebox.showinfo(
            "Format du corpus",
            "Nommez vos fichiers .txt selon le format :\n\n"
            "   typedefichier_année_id.txt\n\n"
            "Exemples :\n"
            "   mag_1789_4569.txt\n"
            "   roman_1920_0001.txt\n"
            "   journal_1945_2233.txt\n\n"
            "Placez tous les fichiers dans un même dossier.\n"
            "L'analyse diachronique détecte l'année dans le nom du fichier\n"
            "puis regroupe automatiquement les textes par décennie.",
        )

    def _show_about(self):
        messagebox.showinfo(
            "À propos",
            "TextSimplifier & Analyse Diachronique\n\n"
            "• Simplification syntaxique via spaCy (EN/FR)\n"
            "• Word2Vec par décennie (gensim)\n"
            "• Repère global permanent\n"
            "• Alignement Procuste (scipy)\n"
            "• Projection ACP 2D (scikit-learn)\n\n"
            "Référence : Hamilton et al. (2016),\n"
            "\"Diachronic Word Embeddings Reveal\n"
            "Statistical Laws of Semantic Change\"",
        )


# ── Lancement ─────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    app = App()
    app.mainloop()
