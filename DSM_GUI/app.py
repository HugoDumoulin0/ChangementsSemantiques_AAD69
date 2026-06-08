from pathlib import Path
import shutil
import subprocess
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from nltk_setup import PROJECT_NLTK_DATA_DIR, ensure_project_nltk_data

try:
    from PIL import Image, ImageTk
    PIL_AVAILABLE = True
except ModuleNotFoundError:
    Image = None
    ImageTk = None
    PIL_AVAILABLE = False


ROOT_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = ROOT_DIR / "scripts"
PREVIEW_SIZE = (980, 620)
DEFAULT_WINDOW_WIDTH = 1600
DEFAULT_WINDOW_HEIGHT = 930
MIN_WINDOW_WIDTH = 1350
MIN_WINDOW_HEIGHT = 850


def run_script(script_name, args):
    command = [sys.executable, str(SCRIPTS_DIR / script_name), *args]
    return subprocess.run(
        command,
        cwd=ROOT_DIR,
        capture_output=True,
        text=True,
        check=False,
    )


class ScriptPanel(ttk.Frame):
    def __init__(
        self,
        master,
        title,
        fields,
        runner,
        outputs_getter,
        text_file_getter=None,
        cleanup_targets_getter=None,
    ):
        super().__init__(master, padding=12)
        self.runner = runner
        self.outputs_getter = outputs_getter
        self.text_file_getter = text_file_getter
        self.cleanup_targets_getter = cleanup_targets_getter
        self.variables = {}
        self.preview_image = None
        self.preview_source_path = None

        self.columnconfigure(0, weight=0)
        self.columnconfigure(1, weight=1)
        self.columnconfigure(2, weight=1)
        self.rowconfigure(1, weight=1)

        ttk.Label(self, text=title, font=("Helvetica", 16, "bold")).grid(
            row=0, column=0, columnspan=3, sticky="w", pady=(0, 10)
        )

        form = ttk.LabelFrame(self, text="Parametres", padding=10)
        form.grid(row=1, column=0, sticky="nsew", padx=(0, 12))

        for idx, field in enumerate(fields):
            ttk.Label(form, text=field["label"]).grid(row=idx, column=0, sticky="w", pady=4)
            if field["type"] == "choice":
                var = tk.StringVar(value=field["default"])
                widget = ttk.Combobox(
                    form,
                    textvariable=var,
                    values=field["options"],
                    state="readonly",
                    width=20,
                )
            else:
                var = tk.StringVar(value=str(field["default"]))
                widget = ttk.Entry(form, textvariable=var, width=28)
            widget.grid(row=idx, column=1, sticky="ew", pady=4, padx=(8, 0))
            if field.get("browse") == "directory":
                ttk.Button(
                    form,
                    text="Parcourir",
                    command=lambda current_var=var: self.select_directory(current_var),
                ).grid(row=idx, column=2, sticky="ew", pady=4, padx=(8, 0))
            self.variables[field["name"]] = (var, field["type"])

        form.columnconfigure(1, weight=1)
        form.columnconfigure(2, weight=0)

        actions = ttk.Frame(form)
        actions.grid(row=len(fields), column=0, columnspan=2, sticky="ew", pady=(10, 0))
        ttk.Button(actions, text="Lancer", command=self.execute).pack(side="left")
        ttk.Button(actions, text="Rafraichir les sorties", command=self.refresh_outputs).pack(
            side="left", padx=(8, 0)
        )
        if self.cleanup_targets_getter is not None:
            ttk.Button(actions, text="Effacer les sorties", command=self.clear_generated_outputs).pack(
                side="left", padx=(8, 0)
            )

        right = ttk.Frame(self)
        right.grid(row=1, column=1, columnspan=2, sticky="nsew")
        right.columnconfigure(0, weight=0)
        right.columnconfigure(1, weight=1)
        right.rowconfigure(1, weight=1)
        right.rowconfigure(3, weight=1)

        ttk.Label(right, text="Visualisations").grid(row=0, column=0, sticky="w")
        ttk.Label(right, text="Apercu").grid(row=0, column=1, sticky="w")

        self.outputs_list = tk.Listbox(right, exportselection=False, height=12, width=34)
        self.outputs_list.grid(row=1, column=0, sticky="nsew", padx=(0, 10))
        self.outputs_list.bind("<<ListboxSelect>>", self.on_select_output)
        self.outputs_list.bind("<Double-Button-1>", self.open_selected_output)

        preview_frame = ttk.Frame(right)
        preview_frame.grid(row=1, column=1, sticky="nsew")
        preview_frame.rowconfigure(0, weight=1)
        preview_frame.columnconfigure(0, weight=1)
        preview_frame.bind("<Configure>", self.on_preview_resize)
        self.preview_frame = preview_frame

        self.preview_label = ttk.Label(preview_frame, anchor="center")
        self.preview_label.grid(row=0, column=0, sticky="nsew")
        if not PIL_AVAILABLE:
            self.preview_label.configure(
                text="Apercu indisponible.\nInstalle 'pillow' pour afficher les images ici."
            )

        buttons = ttk.Frame(right)
        buttons.grid(row=2, column=0, columnspan=2, sticky="w", pady=(8, 8))
        ttk.Button(buttons, text="Ouvrir le fichier", command=self.open_selected_output).pack(side="left")

        ttk.Label(right, text="Logs").grid(row=3, column=0, columnspan=2, sticky="w")
        self.logs = tk.Text(right, wrap="word", height=12)
        self.logs.grid(row=4, column=0, columnspan=2, sticky="nsew")
        self.logs.configure(state="disabled")

        if self.text_file_getter is not None:
            ttk.Label(right, text="Resultat texte").grid(row=5, column=0, columnspan=2, sticky="w", pady=(8, 0))
            self.text_output = tk.Text(right, wrap="word", height=12)
            self.text_output.grid(row=6, column=0, columnspan=2, sticky="nsew")
            self.text_output.configure(state="disabled")
            right.rowconfigure(6, weight=1)
        else:
            self.text_output = None

        self.refresh_outputs()

    def append_logs(self, content):
        self.logs.configure(state="normal")
        self.logs.delete("1.0", tk.END)
        self.logs.insert(tk.END, content.strip() or "Aucun log.")
        self.logs.configure(state="disabled")

    def read_values(self):
        values = {}
        for name, (var, value_type) in self.variables.items():
            raw_value = var.get().strip()
            if value_type == "int":
                values[name] = int(raw_value)
            else:
                values[name] = raw_value
        return values

    def execute(self):
        try:
            values = self.read_values()
        except ValueError:
            messagebox.showerror("Parametres invalides", "Certaines valeurs numeriques sont invalides.")
            return

        self.append_logs("Execution en cours...")
        self.update_idletasks()

        result = self.runner(values)
        logs = result.stdout
        if result.stderr:
            logs = f"{logs}\n\n[stderr]\n{result.stderr}" if logs else result.stderr

        status = "Execution terminee." if result.returncode == 0 else f"Echec (code {result.returncode})."
        self.append_logs(f"{status}\n\n{logs}")
        self.refresh_outputs()
        self.enforce_window_bounds()

        if result.returncode != 0:
            messagebox.showerror("Execution echouee", f"Le script a retourne le code {result.returncode}.")

    def refresh_outputs(self):
        output_paths = self.outputs_getter(self.read_values_safe())
        self.outputs_list.delete(0, tk.END)
        for path in output_paths:
            self.outputs_list.insert(tk.END, str(path.relative_to(ROOT_DIR)))
        self.current_outputs = output_paths

        if output_paths:
            self.outputs_list.selection_set(0)
            self.on_select_output()
        else:
            self.preview_label.configure(image="", text="Aucune visualisation disponible.")
            self.preview_image = None
            self.preview_source_path = None

        if self.text_output is not None and self.text_file_getter is not None:
            text_path = self.text_file_getter(self.read_values_safe())
            self.text_output.configure(state="normal")
            self.text_output.delete("1.0", tk.END)
            if text_path.exists():
                self.text_output.insert(tk.END, text_path.read_text(encoding="utf-8"))
            else:
                self.text_output.insert(tk.END, "Aucun resultat texte disponible.")
            self.text_output.configure(state="disabled")
        self.enforce_window_bounds()

    def read_values_safe(self):
        try:
            return self.read_values()
        except ValueError:
            safe_values = {}
            for name, (var, _) in self.variables.items():
                safe_values[name] = var.get().strip()
            return safe_values

    def select_directory(self, variable):
        initial_dir = variable.get().strip() or str(ROOT_DIR)
        selected_dir = filedialog.askdirectory(initialdir=initial_dir or str(ROOT_DIR))
        if selected_dir:
            variable.set(selected_dir)

    def clear_generated_outputs(self):
        if self.cleanup_targets_getter is None:
            return

        values = self.read_values_safe()
        targets = [Path(path) for path in self.cleanup_targets_getter(values)]
        existing_targets = [path for path in targets if path.exists()]

        if not existing_targets:
            self.append_logs("Aucune sortie a effacer.")
            self.refresh_outputs()
            return

        confirmed = messagebox.askyesno(
            "Confirmer l'effacement",
            "Supprimer les fichiers et dossiers generes pour cet onglet ?",
        )
        if not confirmed:
            return

        removed = []
        for path in existing_targets:
            try:
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()
                removed.append(str(path.relative_to(ROOT_DIR)))
            except FileNotFoundError:
                continue

        log_message = "Sorties effacees :\n" + "\n".join(removed) if removed else "Aucune sortie effacee."
        self.append_logs(log_message)
        self.refresh_outputs()

    def enforce_window_bounds(self):
        root = self.winfo_toplevel()
        root.update_idletasks()
        width = min(max(root.winfo_width(), MIN_WINDOW_WIDTH), DEFAULT_WINDOW_WIDTH)
        height = min(max(root.winfo_height(), MIN_WINDOW_HEIGHT), DEFAULT_WINDOW_HEIGHT)
        root.geometry(f"{width}x{height}")

    def on_select_output(self, _event=None):
        selection = self.outputs_list.curselection()
        if not selection or not getattr(self, "current_outputs", None):
            return

        if not PIL_AVAILABLE:
            self.preview_label.configure(
                image="",
                text="Apercu indisponible.\nInstalle 'pillow' pour l'activer.",
            )
            self.preview_image = None
            self.preview_source_path = None
            return

        path = self.current_outputs[selection[0]]
        if not path.exists():
            self.preview_label.configure(image="", text="Fichier introuvable.")
            self.preview_image = None
            self.preview_source_path = None
            return

        self.preview_source_path = path
        self.render_preview()

    def on_preview_resize(self, _event=None):
        if self.preview_source_path is not None:
            self.render_preview()

    def render_preview(self):
        if self.preview_source_path is None or not self.preview_source_path.exists():
            return

        frame_width = max(self.preview_frame.winfo_width() - 20, 200)
        frame_height = max(self.preview_frame.winfo_height() - 20, 200)
        max_size = (
            min(frame_width, PREVIEW_SIZE[0]),
            min(frame_height, PREVIEW_SIZE[1]),
        )

        try:
            image = Image.open(self.preview_source_path)
            image.thumbnail(max_size)
            self.preview_image = ImageTk.PhotoImage(image)
            self.preview_label.configure(image=self.preview_image, text="")
        except Exception as exc:
            self.preview_label.configure(image="", text=f"Apercu indisponible:\n{exc}")
            self.preview_image = None

    def open_selected_output(self, _event=None):
        selection = self.outputs_list.curselection()
        if not selection or not getattr(self, "current_outputs", None):
            return
        path = self.current_outputs[selection[0]]
        if not path.exists():
            messagebox.showerror("Fichier introuvable", str(path))
            return
        self.open_file(path)

    @staticmethod
    def open_file(path):
        if sys.platform == "darwin":
            subprocess.Popen(["open", str(path)])
        elif sys.platform.startswith("win"):
            subprocess.Popen(["start", str(path)], shell=True)
        else:
            subprocess.Popen(["xdg-open", str(path)])


def build_pca_args(values):
    return [
        "--corpus-dir",
        values["corpus_dir"],
        "--top-n-words",
        str(values["top_n_words"]),
        "--window-size",
        str(values["window_size"]),
        "--n-labels",
        str(values["n_labels"]),
        "--output-path",
        values["output_path"],
        "--no-show",
    ]


def build_traj_args(values):
    return [
        "--corpus-dir",
        values["corpus_dir"],
        "--target-word",
        values["target_word"],
        "--choice",
        values["choice"],
        "--top-n-words",
        str(values["top_n_words"]),
        "--top-n-neighbors",
        str(values["top_n_neighbors"]),
        "--top-n-cooc",
        str(values["top_n_cooc"]),
        "--window-size",
        str(values["window_size"]),
        "--min-cooc",
        str(values["min_cooc"]),
        "--matrices-dir",
        values["matrices_dir"],
        "--top-k-traj",
        str(values["top_k_traj"]),
        "--top-k-area",
        str(values["top_k_area"]),
        "--no-show",
    ]


def build_ward_args(values):
    return [
        "--matrices-dir",
        values["matrices_dir"],
        "--output-dir",
        values["output_dir"],
        "--max-k",
        str(values["max_k"]),
        "--n-clusters",
        str(values["n_clusters"]),
        "--no-show",
    ]


def run_pca(values):
    return run_script("run_pca.py", build_pca_args(values))


def run_traj(values):
    return run_script("run_trajectoire_cooc.py", build_traj_args(values))


def run_ward(values):
    return run_script("run_ward_clustering.py", build_ward_args(values))


def pca_outputs(values):
    output_path = values.get("output_path", "pca_coha_2d.jpeg")
    return [ROOT_DIR / output_path] if (ROOT_DIR / output_path).exists() else []


def traj_outputs(values):
    target_word = values.get("target_word", "law")
    traj_root = ROOT_DIR / f"{target_word}_viz"
    paths = [
        traj_root / f"trajectoire_cooc_{target_word}.jpeg",
        traj_root / f"top_trajectoires_{target_word}_vocab.jpeg",
        traj_root / f"top_aires_{target_word}_vocab.jpeg",
    ]
    paths.extend(sorted((traj_root / f"cards_cooc_{target_word}").glob("*.jpeg")))
    paths.extend(sorted((traj_root / f"cards_similar_{target_word}").glob("*.jpeg")))
    return [path for path in paths if path.exists()]


def traj_text_output(values):
    target_word = values.get("target_word", "law")
    return ROOT_DIR / f"{target_word}_viz" / f"trajectoires_{target_word}.txt"


def ward_outputs(values):
    output_dir = values.get("output_dir", "ward_viz")
    ward_root = ROOT_DIR / output_dir
    paths = [
        ward_root / "elbow_method.jpeg",
        ward_root / "ward_dendrogram.jpeg",
        ward_root / "ward_clusters_pca.jpeg",
    ]
    return [path for path in paths if path.exists()]


def pca_cleanup_targets(values):
    return [ROOT_DIR / values.get("output_path", "pca_coha_2d.jpeg")]


def traj_cleanup_targets(values):
    target_word = values.get("target_word", "law")
    matrices_dir = values.get("matrices_dir", "ppmi_matrix/")
    return [
        ROOT_DIR / f"{target_word}_viz",
        ROOT_DIR / matrices_dir,
    ]


def ward_cleanup_targets(values):
    output_dir = values.get("output_dir", "ward_viz")
    return [ROOT_DIR / output_dir]


def main():
    nltk_status = ensure_project_nltk_data(download_missing=True)
    root = tk.Tk()
    root.title("DSM Interface")
    root.geometry(f"{DEFAULT_WINDOW_WIDTH}x{DEFAULT_WINDOW_HEIGHT}")
    root.minsize(MIN_WINDOW_WIDTH, MIN_WINDOW_HEIGHT)
    root.maxsize(DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT)

    notebook = ttk.Notebook(root)
    notebook.pack(fill="both", expand=True)

    pca_fields = [
        {
            "name": "corpus_dir",
            "label": "Dossier du corpus",
            "default": "../COHA_sample/",
            "type": "str",
            "browse": "directory",
        },
        {"name": "top_n_words", "label": "Top N mots", "default": 3500, "type": "int"},
        {"name": "window_size", "label": "Taille de fenetre", "default": 5, "type": "int"},
        {"name": "n_labels", "label": "Nombre de labels", "default": 100, "type": "int"},
        {"name": "output_path", "label": "Fichier de sortie", "default": "pca_coha_2d.jpeg", "type": "str"},
    ]

    traj_fields = [
        {
            "name": "corpus_dir",
            "label": "Dossier du corpus",
            "default": "../COHA_sample/",
            "type": "str",
            "browse": "directory",
        },
        {"name": "target_word", "label": "Mot cible", "default": "law", "type": "str"},
        {
            "name": "choice",
            "label": "Representation",
            "default": "lemma",
            "type": "choice",
            "options": ["lemma", "form"],
        },
        {"name": "top_n_words", "label": "Top N mots", "default": 3500, "type": "int"},
        {"name": "top_n_neighbors", "label": "Voisins PCA affiches", "default": 3, "type": "int"},
        {"name": "top_n_cooc", "label": "Voisins cooccurrence/similarite", "default": 20, "type": "int"},
        {"name": "window_size", "label": "Taille de fenetre", "default": 5, "type": "int"},
        {"name": "min_cooc", "label": "Cooccurrence minimale", "default": 2, "type": "int"},
        {
            "name": "matrices_dir",
            "label": "Dossier des matrices",
            "default": "ppmi_matrix/",
            "type": "str",
            "browse": "directory",
        },
        {"name": "top_k_traj", "label": "Top K trajectoires", "default": 5, "type": "int"},
        {"name": "top_k_area", "label": "Top K aires", "default": 5, "type": "int"},
    ]

    ward_fields = [
        {
            "name": "matrices_dir",
            "label": "Dossier des matrices",
            "default": "ppmi_matrix/",
            "type": "str",
            "browse": "directory",
        },
        {"name": "output_dir", "label": "Dossier de sortie", "default": "ward_viz", "type": "str"},
        {"name": "max_k", "label": "K max", "default": 10, "type": "int"},
        {"name": "n_clusters", "label": "Nombre de clusters", "default": 2, "type": "int"},
    ]

    notebook.add(
        ScriptPanel(
            notebook,
            "PCA globale",
            pca_fields,
            run_pca,
            pca_outputs,
            cleanup_targets_getter=pca_cleanup_targets,
        ),
        text="PCA",
    )
    notebook.add(
        ScriptPanel(
            notebook,
            "Trajectoires semantiques",
            traj_fields,
            run_traj,
            traj_outputs,
            text_file_getter=traj_text_output,
            cleanup_targets_getter=traj_cleanup_targets,
        ),
        text="Trajectoires",
    )
    notebook.add(
        ScriptPanel(
            notebook,
            "Ward clustering",
            ward_fields,
            run_ward,
            ward_outputs,
            cleanup_targets_getter=ward_cleanup_targets,
        ),
        text="Ward",
    )

    if any(status != "available" for status in nltk_status.values()):
        status_lines = [f"{name}: {state}" for name, state in sorted(nltk_status.items())]
        print(f"NLTK data directory: {PROJECT_NLTK_DATA_DIR}")
        print("NLTK resources status:")
        for line in status_lines:
            print(f"  - {line}")

    root.mainloop()


if __name__ == "__main__":
    main()
