"""
io_loader.py — Chargement des fichiers .txt et extraction de l'année
Format attendu : typedefichier_année_id.txt  (ex: mag_1920_4569.txt)
"""

import re
from pathlib import Path


def extract_year_from_filename(filename: str) -> int | None:
    """
    Extrait l'année depuis un nom de fichier au format type_année_id.txt
    Exemples : mag_1789_4569.txt -> 1789
    """
    match = re.search(r'_(\d{4})_', str(filename))
    if match:
        year = int(match.group(1))
        if 1000 <= year <= 2100:  # sanity check
            return year
    return None


def load_text_file(filepath: str) -> str:
    """Charge un fichier texte UTF-8."""
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        return f.read()


def load_corpus_by_year(corpus_dir: str) -> dict[int, list[str]]:
    """
    Charge tous les fichiers .txt d'un dossier, groupés par année.
    Retourne : {année: [texte1, texte2, ...]}
    Ignore les fichiers dont le nom ne contient pas d'année.
    """
    corpus: dict[int, list[str]] = {}
    for filepath in Path(corpus_dir).glob('*.txt'):
        year = extract_year_from_filename(filepath.name)
        if year is not None:
            text = load_text_file(str(filepath))
            if text.strip():
                corpus.setdefault(year, []).append(text)
    return corpus


def get_files_info(corpus_dir: str) -> list[dict]:
    """
    Retourne des métadonnées sur tous les fichiers .txt du dossier.
    Utile pour afficher la liste des années détectées.
    """
    info = []
    for filepath in sorted(Path(corpus_dir).glob('*.txt')):
        year = extract_year_from_filename(filepath.name)
        try:
            size = filepath.stat().st_size
        except OSError:
            size = 0
        info.append({
            'path': str(filepath),
            'name': filepath.name,
            'year': year,
            'size': size,
        })
    return info
