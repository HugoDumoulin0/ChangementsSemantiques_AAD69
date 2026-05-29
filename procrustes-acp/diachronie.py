"""
diachronie.py — Analyse sémantique diachronique

Pipeline complet :
  1. Chargement du corpus par année (via io_loader)
  2. Entraînement d'un modèle Word2Vec par année
  3. Construction du vocabulaire partagé
  4. Alignement des espaces vectoriels par analyse de Procuste
  5. Réduction dimensionnelle ACP (2D)
  6. Visualisation matplotlib de la trajectoire d'un mot à travers les années

Référence méthode : Hamilton et al. (2016), "Diachronic Word Embeddings Reveal
Statistical Laws of Semantic Change", ACL 2016.
"""

import re
import math
import collections
import json
import hashlib
from typing import Optional, Callable, Any
from pathlib import Path

import numpy as np
from scipy.linalg import orthogonal_procrustes
from sklearn.decomposition import PCA
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from gensim.models import Word2Vec

from io_loader import load_corpus_by_year


_W2V_TRAIN_KEYS = {
    'vector_size',
    'window',
    'min_count',
    'epochs',
    'seed',
    'workers',
}

_CACHE_VERSION = 1

# ── Mots vides à exclure des mots ancres ────────────────────────────────────

_STOPWORDS_EN = {
    'the', 'a', 'an', 'in', 'on', 'at', 'of', 'and', 'or', 'but', 'not',
    'it', 'he', 'she', 'they', 'that', 'this', 'with', 'for', 'to', 'from',
    'by', 'as', 'be', 'have', 'has', 'had', 'do', 'does', 'did', 'is', 'are',
    'was', 'were', 'will', 'would', 'could', 'should', 'may', 'might', 'shall',
    'its', 'his', 'her', 'their', 'our', 'we', 'you', 'me', 'him', 'us', 'my',
    'your', 'if', 'then', 'when', 'where', 'who', 'what', 'how', 'all', 'so',
    'just', 'also', 'very', 'even', 'one', 'two', 'more', 'most', 'such',
    'into', 'up', 'out', 'about', 'than', 'after', 'before', 'there', 'here',
    'no', 'yet', 'both', 'each', 'few', 'other', 'back', 'can', 'been',
}

_STOPWORDS_FR = {
    'le', 'la', 'les', 'de', 'du', 'des', 'un', 'une', 'et', 'est', 'il',
    'elle', 'que', 'qui', 'dans', 'avec', 'sur', 'par', 'pour', 'en', 'au',
    'aux', 'mais', 'ou', 'donc', 'ni', 'car', 'ne', 'pas', 'plus', 'très',
    'aussi', 'comme', 'leur', 'se', 'sa', 'son', 'ses', 'nous', 'vous', 'ils',
    'elles', 'ce', 'cet', 'cette', 'ces', 'si', 'sont', 'ont', 'être', 'avoir',
    'bien', 'tout', 'même', 'encore', 'dont', 'où', 'quand', 'comment', 'y',
    'lui', 'me', 'te', 'on', 'je', 'tu', 'nous', 'vous', 'mon', 'ton', 'ma',
    'ta', 'mes', 'tes', 'fait', 'dit', 'peu', 'puis', 'mais', 'alors', 'ainsi',
}

_ALL_STOPWORDS = _STOPWORDS_EN | _STOPWORDS_FR


# ── Tokenisation ────

def tokenize_text(text: str) -> list[str]:
    """
    Tokenisation simple : tokens alphabétiques uniquement (min. 2 caractères),
    en minuscules. Gère les caractères accentués.
    """
    return [w.lower() for w in re.findall(r"\b[a-zA-ZÀ-ÿ\u0400-\u04FF]{2,}\b", text)]


def _text_to_sentences(text: str) -> list[list[str]]:
    """Découpe un texte en phrases tokenisées pour Word2Vec."""
    raw_sentences = re.split(r'[.!?…]+', text)
    sentences = []
    for s in raw_sentences:
        tokens = tokenize_text(s)
        if len(tokens) >= 3:
            sentences.append(tokens)
    return sentences


# ── Entraînement Word2Vec par année ───────

def _resolve_workers(w2v_params: dict) -> int:
    """Résout le nombre de workers en tenant compte du mode reproductible."""
    if bool(w2v_params.get('reproducible', False)):
        return 1
    return int(w2v_params.get('workers', 4))


def build_sentences_by_period(
    corpus_by_period: dict[int, list[str]],
) -> dict[int, list[list[str]]]:
    """Pré-tokenise le corpus par période en listes de phrases."""
    sentences_by_period: dict[int, list[list[str]]] = {}
    for period in sorted(corpus_by_period.keys()):
        all_sentences: list[list[str]] = []
        for text in corpus_by_period[period]:
            all_sentences.extend(_text_to_sentences(text))
        sentences_by_period[period] = all_sentences
    return sentences_by_period


def _flatten_sentences(
    sentences_by_period: dict[int, list[list[str]]],
) -> list[list[str]]:
    """Concatène les phrases de toutes les périodes."""
    merged: list[list[str]] = []
    for period in sorted(sentences_by_period.keys()):
        merged.extend(sentences_by_period[period])
    return merged


def _compute_corpus_signature(corpus_dir: str) -> str:
    """Construit une signature stable du corpus à partir des fichiers .txt."""
    root = Path(corpus_dir)
    digest = hashlib.sha256()
    for filepath in sorted(root.glob('*.txt')):
        try:
            stat = filepath.stat()
            digest.update(
                f"{filepath.name}:{stat.st_size}:{stat.st_mtime_ns}".encode('utf-8')
            )
        except OSError:
            continue
    return digest.hexdigest()[:16]


def _make_cache_key(
    kind: str,
    corpus_signature: str,
    granularity: str,
    train_params: dict,
) -> str:
    payload = {
        'kind': kind,
        'corpus': corpus_signature,
        'granularity': granularity,
        'train_params': train_params,
        'cache_version': _CACHE_VERSION,
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(raw.encode('utf-8')).hexdigest()[:20]


def _cache_dir(corpus_dir: str) -> Path:
    path = Path(corpus_dir) / '.diachronie_cache'
    path.mkdir(parents=True, exist_ok=True)
    return path


def _load_period_models_from_cache(cache_root: Path, cache_key: str) -> Optional[dict[int, Word2Vec]]:
    manifest_path = cache_root / f"{cache_key}_period_manifest.json"
    if not manifest_path.exists():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return None

    models: dict[int, Word2Vec] = {}
    for period_str, filename in manifest.get('files', {}).items():
        model_path = cache_root / filename
        if not model_path.exists():
            return None
        try:
            models[int(period_str)] = Word2Vec.load(str(model_path))
        except Exception:
            return None
    return models if models else None


def _save_period_models_to_cache(cache_root: Path, cache_key: str, models: dict[int, Word2Vec]) -> None:
    files: dict[str, str] = {}
    for period, model in models.items():
        filename = f"{cache_key}_period_{period}.model"
        model.save(str(cache_root / filename))
        files[str(period)] = filename

    manifest = {
        'cache_version': _CACHE_VERSION,
        'files': files,
    }
    (cache_root / f"{cache_key}_period_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=True, sort_keys=True),
        encoding='utf-8',
    )


def _load_global_model_from_cache(cache_root: Path, cache_key: str) -> Optional[Word2Vec]:
    path = cache_root / f"{cache_key}_global.model"
    if not path.exists():
        return None
    try:
        return Word2Vec.load(str(path))
    except Exception:
        return None


def _save_global_model_to_cache(cache_root: Path, cache_key: str, model: Word2Vec) -> None:
    model.save(str(cache_root / f"{cache_key}_global.model"))


def train_models_by_year(
    sentences_by_year: dict[int, list[list[str]]],
    vector_size: int = 100,
    window: int = 5,
    min_count: int = 2,
    epochs: int = 15,
    seed: int = 42,
    workers: int = 4,
    cache_root: Optional[Path] = None,
    cache_key: Optional[str] = None,
) -> dict[int, Word2Vec]:
    """
    Entraîne un modèle Word2Vec indépendant pour chaque année.

    Returns:
        {année: modèle Word2Vec entraîné}
    """
    if cache_root is not None and cache_key is not None:
        cached = _load_period_models_from_cache(cache_root, cache_key)
        if cached is not None:
            print(f"[W2V-cache] Modèles périodes chargés ({len(cached)} périodes)")
            return cached

    models: dict[int, Word2Vec] = {}

    for year in sorted(sentences_by_year.keys()):
        all_sentences = sentences_by_year[year]

        if len(all_sentences) < 2:
            print(f"[W2V] Année {year} ignorée : corpus trop petit "
                  f"({len(all_sentences)} phrases).")
            continue

        model = Word2Vec(
            sentences=all_sentences,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            workers=workers,
            seed=seed,
            epochs=epochs,
        )
        models[year] = model
        vocab_size = len(model.wv)
        print(f"[W2V] {year} → {len(all_sentences)} phrases, {vocab_size} mots")

    if cache_root is not None and cache_key is not None and models:
        _save_period_models_to_cache(cache_root, cache_key, models)

    return models


def train_global_model(
    sentences_by_year: dict[int, list[list[str]]],
    vector_size: int = 100,
    window: int = 5,
    min_count: int = 2,
    epochs: int = 15,
    seed: int = 42,
    workers: int = 4,
    cache_root: Optional[Path] = None,
    cache_key: Optional[str] = None,
) -> Word2Vec:
    """Entraine un modèle Word2Vec unique sur l'ensemble du corpus."""
    if cache_root is not None and cache_key is not None:
        cached = _load_global_model_from_cache(cache_root, cache_key)
        if cached is not None:
            print("[W2V-cache] Modèle global chargé")
            return cached

    all_sentences = _flatten_sentences(sentences_by_year)

    if len(all_sentences) < 2:
        raise ValueError("Corpus global trop petit pour entraîner Word2Vec.")

    model = Word2Vec(
        sentences=all_sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        seed=seed,
        epochs=epochs,
    )
    print(f"[W2V-global] {len(all_sentences)} phrases, {len(model.wv)} mots")

    if cache_root is not None and cache_key is not None:
        _save_global_model_to_cache(cache_root, cache_key, model)

    return model


def _extract_w2v_train_params(w2v_params: dict) -> dict:
    """Filtre les paramètres supportés par Word2Vec pour l'entrainement."""
    return {
        k: v
        for k, v in w2v_params.items()
        if k in _W2V_TRAIN_KEYS
    }


def aggregate_corpus_by_decade(corpus_by_year: dict[int, list[str]]) -> dict[int, list[str]]:
    """Regroupe le corpus par décennie (ex: 1817 -> 1810)."""
    corpus_by_decade: dict[int, list[str]] = {}
    for year, texts in corpus_by_year.items():
        decade = (year // 10) * 10
        corpus_by_decade.setdefault(decade, []).extend(texts)
    return corpus_by_decade


def _compute_period_quality_stats(
    corpus_by_period: dict[int, list[str]],
    sentences_by_period: dict[int, list[list[str]]],
    target_word: str,
) -> dict[int, dict[str, int]]:
    """Calcule des stats de qualité simples pour chaque période."""
    stats: dict[int, dict[str, int]] = {}
    for period in sorted(corpus_by_period.keys()):
        texts = corpus_by_period.get(period, [])
        sentences = sentences_by_period.get(period, [])
        token_count = sum(len(sentence) for sentence in sentences)
        target_occurrences = sum(
            1
            for sentence in sentences
            for tok in sentence
            if tok == target_word
        )
        stats[period] = {
            'texts': len(texts),
            'tokens': int(token_count),
            'target_occurrences': int(target_occurrences),
        }
    return stats


def _filter_periods_by_quality(
    corpus_by_period: dict[int, list[str]],
    sentences_by_period: dict[int, list[list[str]]],
    target_word: str,
    min_texts: int,
    min_tokens: int,
    min_target_occurrences: int,
) -> tuple[dict[int, list[str]], dict[int, list[list[str]]], list[dict[str, Any]]]:
    """Filtre les périodes trop faibles et retourne un diagnostic détaillé."""
    quality_stats = _compute_period_quality_stats(
        corpus_by_period,
        sentences_by_period,
        target_word,
    )

    kept_corpus: dict[int, list[str]] = {}
    kept_sentences: dict[int, list[list[str]]] = {}
    dropped_details: list[dict[str, Any]] = []

    for period in sorted(corpus_by_period.keys()):
        stat = quality_stats[period]
        reasons: list[str] = []
        if stat['texts'] < min_texts:
            reasons.append(f"texts<{min_texts}")
        if stat['tokens'] < min_tokens:
            reasons.append(f"tokens<{min_tokens}")
        if stat['target_occurrences'] < min_target_occurrences:
            reasons.append(f"target_occ<{min_target_occurrences}")

        if reasons:
            dropped_details.append({
                'period': period,
                'texts': stat['texts'],
                'tokens': stat['tokens'],
                'target_occurrences': stat['target_occurrences'],
                'reasons': reasons,
            })
            continue

        kept_corpus[period] = corpus_by_period[period]
        kept_sentences[period] = sentences_by_period[period]

    return kept_corpus, kept_sentences, dropped_details


# ── Alignement par analyse de Procuste ───────────────────────────────────────

def build_reference_alignments(
    models: dict[int, Word2Vec],
    reference_year: int,
    min_common_for_alignment: int = 200,
    return_details: bool = False,
) -> tuple:
    """
    Construit une matrice de rotation vers l'espace de référence pour chaque année.

    Au lieu d'intersecter le vocabulaire de toutes les années (souvent trop strict),
    on aligne chaque année i vers l'année de référence en utilisant uniquement leur
    vocabulaire commun pair-a-pair.

    Returns:
        rotations:   {annee: matrice de rotation R}
        usable_years: liste des annees conservées (reference incluse)
        ref_vocab:   vocabulaire de l'annee de reference
    """
    if reference_year not in models:
        raise ValueError(
            f"L'année de référence {reference_year} n'a pas de modèle entraîné."
        )

    ref_model = models[reference_year]
    ref_vocab = set(ref_model.wv.key_to_index.keys())
    dim = ref_model.vector_size

    rotations: dict[int, np.ndarray] = {
        reference_year: np.eye(dim, dtype=np.float64)
    }
    usable_years: list[int] = [reference_year]
    alignment_details: list[dict[str, Any]] = []

    for year in sorted(models.keys()):
        model = models[year]
        if year == reference_year:
            continue

        common = sorted(ref_vocab & set(model.wv.key_to_index.keys()))
        if len(common) < min_common_for_alignment:
            print(
                f"[Procuste] {year} ignoree : {len(common)} mots communs "
                f"(< {min_common_for_alignment})"
            )
            alignment_details.append({
                'period': year,
                'common_vocab': len(common),
                'kept': 0,
            })
            continue

        src_matrix = np.array([model.wv[w] for w in common], dtype=np.float64)
        ref_matrix = np.array([ref_model.wv[w] for w in common], dtype=np.float64)
        # orthogonal_procrustes(A, B) → R tel que A @ R ≈ B
        R, _ = orthogonal_procrustes(src_matrix, ref_matrix)
        rotations[year] = R
        usable_years.append(year)
        alignment_details.append({
            'period': year,
            'common_vocab': len(common),
            'kept': 1,
        })
        print(f"[Procuste] Année {year} alignée sur {reference_year} ({len(common)} mots)")

    usable_years = sorted(usable_years)
    if return_details:
        return rotations, usable_years, ref_vocab, alignment_details
    return rotations, usable_years, ref_vocab


def _aligned_vector_for_word(
    models: dict[int, Word2Vec],
    rotations: dict[int, np.ndarray],
    year: int,
    word: str,
) -> Optional[np.ndarray]:
    """Retourne le vecteur aligne d'un mot pour une année, ou None s'il manque."""
    model = models.get(year)
    if model is None or year not in rotations:
        return None
    if word not in model.wv:
        return None
    return np.asarray(model.wv[word], dtype=np.float64) @ rotations[year]


def _align_global_model_to_reference(
    global_model: Word2Vec,
    reference_model: Word2Vec,
    min_common_for_alignment: int = 200,
) -> np.ndarray:
    """Retourne la rotation qui aligne le modèle global sur l'espace de référence."""
    common = sorted(
        set(global_model.wv.key_to_index.keys())
        & set(reference_model.wv.key_to_index.keys())
    )
    if len(common) < min_common_for_alignment:
        raise ValueError(
            "Impossible d'aligner le modèle global sur la référence : "
            f"{len(common)} mots communs (< {min_common_for_alignment})."
        )

    src_matrix = np.array([global_model.wv[w] for w in common], dtype=np.float64)
    ref_matrix = np.array([reference_model.wv[w] for w in common], dtype=np.float64)
    rotation, _ = orthogonal_procrustes(src_matrix, ref_matrix)
    return rotation


def build_global_alignments(
    models: dict[int, Word2Vec],
    global_model: Word2Vec,
    min_common_for_alignment: int = 200,
    return_details: bool = False,
) -> tuple:
    """Aligne chaque modèle de période vers l'espace du modèle global."""
    global_vocab = set(global_model.wv.key_to_index.keys())

    rotations: dict[int, np.ndarray] = {}
    usable_years: list[int] = []
    alignment_details: list[dict[str, Any]] = []

    for year in sorted(models.keys()):
        model = models[year]
        common = sorted(global_vocab & set(model.wv.key_to_index.keys()))
        if len(common) < min_common_for_alignment:
            print(
                f"[Procuste-global] {year} ignoree : {len(common)} mots communs "
                f"(< {min_common_for_alignment})"
            )
            alignment_details.append({
                'period': year,
                'common_vocab': len(common),
                'kept': 0,
            })
            continue

        src_matrix = np.array([model.wv[w] for w in common], dtype=np.float64)
        global_matrix = np.array([global_model.wv[w] for w in common], dtype=np.float64)
        rotation, _ = orthogonal_procrustes(src_matrix, global_matrix)
        rotations[year] = rotation
        usable_years.append(year)
        alignment_details.append({
            'period': year,
            'common_vocab': len(common),
            'kept': 1,
        })
        print(f"[Procuste-global] Année {year} alignée sur GLOBAL ({len(common)} mots)")

    usable_years = sorted(usable_years)
    if return_details:
        return rotations, usable_years, global_vocab, alignment_details
    return rotations, usable_years, global_vocab


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosinus entre deux vecteurs (retourne 0.0 si norme nulle)."""
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _build_target_metrics_rows(
    models: dict[int, Word2Vec],
    rotations: dict[int, np.ndarray],
    periods: list[int],
    target_word: str,
    reference_period: Optional[int],
    global_target_vec: Optional[np.ndarray],
) -> list[dict[str, Any]]:
    """Construit des métriques exportables par période pour le mot cible."""
    rows: list[dict[str, Any]] = []
    prev_vec: Optional[np.ndarray] = None

    for period in periods:
        vec = _aligned_vector_for_word(models, rotations, period, target_word)
        if vec is None:
            continue

        row: dict[str, Any] = {
            'period': period,
            'is_reference': int(reference_period is not None and period == reference_period),
            'cosine_vs_prev': None,
            'cosine_vs_global': None,
        }

        if prev_vec is not None:
            row['cosine_vs_prev'] = _cosine_similarity(vec, prev_vec)
        if global_target_vec is not None:
            row['cosine_vs_global'] = _cosine_similarity(vec, global_target_vec)

        rows.append(row)
        prev_vec = vec

    return rows


# ── Mots ancres ───

def get_top_anchor_words(
    corpus_by_year: dict[int, list[str]],
    candidate_vocab: set[str],
    years: list[int],
    n: int = 10,
    exclude: Optional[list[str]] = None,
    min_year_coverage: float = 0.35,
) -> list[str]:
    """
    Retourne les N mots les plus fréquents parmi un vocabulaire candidat.
    Ces mots servent de points de repère stables sur le graphique.
    Les mots vides et les mots exclus sont filtrés.

    Un mot est conservé seulement s'il apparaît dans un nombre minimal d'années
    pour rester interprétable diachroniquement.
    """
    exclude_set = set(w.lower() for w in (exclude or []))
    exclude_set |= _ALL_STOPWORDS
    candidate_set = set(candidate_vocab)

    freq: collections.Counter = collections.Counter()
    years_set = set(years)
    year_presence: dict[str, set[int]] = collections.defaultdict(set)

    for year, texts in corpus_by_year.items():
        if year not in years_set:
            continue
        for text in texts:
            for tok in tokenize_text(text):
                if tok in candidate_set and tok not in exclude_set:
                    freq[tok] += 1
                    year_presence[tok].add(year)

    min_years = max(2, int(math.ceil(len(years) * min_year_coverage)))
    filtered = [
        (w, c) for w, c in freq.items()
        if len(year_presence.get(w, set())) >= min_years
    ]
    filtered.sort(key=lambda x: x[1], reverse=True)

    return [w for w, _ in filtered[:n]]


def get_top_anchor_words_from_sentences(
    sentences_by_period: dict[int, list[list[str]]],
    candidate_vocab: set[str],
    years: list[int],
    n: int = 10,
    exclude: Optional[list[str]] = None,
    min_year_coverage: float = 0.35,
) -> list[str]:
    """Version optimisée de sélection des ancres à partir de phrases tokenisées."""
    exclude_set = set(w.lower() for w in (exclude or []))
    exclude_set |= _ALL_STOPWORDS
    candidate_set = set(candidate_vocab)

    freq: collections.Counter = collections.Counter()
    years_set = set(years)
    year_presence: dict[str, set[int]] = collections.defaultdict(set)

    for year, sentences in sentences_by_period.items():
        if year not in years_set:
            continue
        for sentence in sentences:
            for tok in sentence:
                if tok in candidate_set and tok not in exclude_set:
                    freq[tok] += 1
                    year_presence[tok].add(year)

    min_years = max(2, int(math.ceil(len(years) * min_year_coverage)))
    filtered = [
        (w, c) for w, c in freq.items()
        if len(year_presence.get(w, set())) >= min_years
    ]
    filtered.sort(key=lambda x: x[1], reverse=True)
    return [w for w, _ in filtered[:n]]


# ── Visualisation ───

def _get_year_colors(years: list[int]) -> dict[int, tuple]:
    """Attribue une couleur par année."""
    base_colors = list(mcolors.TABLEAU_COLORS.values())  # 10 couleurs distinctes
    if len(years) > len(base_colors):
        # Plus de 10 années : utiliser un colormap continu
        try:
            cmap = matplotlib.colormaps['hsv'].resampled(len(years))
        except AttributeError:
            cmap = plt.cm.get_cmap('hsv', len(years))
        return {year: cmap(i) for i, year in enumerate(sorted(years))}
    return {year: base_colors[i] for i, year in enumerate(sorted(years))}


def plot_diachronic(
    models: dict[int, Word2Vec],
    rotations: dict[int, np.ndarray],
    years: list[int],
    reference_year: Any,
    target_word: str,
    anchor_words: list[str],
    show_anchors: bool = True,
    show_legend: bool = True,
    global_target_vec: Optional[np.ndarray] = None,
    global_similarity_by_period: Optional[list[tuple[int, float]]] = None,
    period_label: str = "année",
) -> plt.Figure:
    """
    Crée un graphique matplotlib 2D montrant l'évolution sémantique
    d'un mot cible à travers les années après projection ACP.

    L'ACP est apprise uniquement sur les points temporels (mot cible et
    éventuellement ancres), puis le point global est projeté dans ce même
    espace sans influencer l'orientation des axes.

    Le mot cible apparaît avec son année (ex: «livre»-1912),
    relié par des flèches pour montrer la trajectoire temporelle.
    Les mots ancres donnent le contexte sémantique de l'espace vectoriel.
    """
    target_lower = target_word.lower()
    target_years = [
        y for y in years
        if _aligned_vector_for_word(models, rotations, y, target_lower) is not None
    ]
    if len(target_years) < 2:
        raise ValueError(
            f"Le mot '{target_word}' n'est pas disponible sur au moins 2 années "
            "après filtrage/alignment. Essayez un mot plus fréquent."
        )

    # Mots à afficher
    words_display = [target_lower]
    if show_anchors:
        words_display.extend(w for w in anchor_words if w != target_lower)

    # ── Collecter les vecteurs pour chaque mot × année ────────────────────
    word_year_vecs: dict[tuple[str, int], np.ndarray] = {}
    all_vecs_for_pca: list[np.ndarray] = []

    for year in years:
        for word in words_display:
            vec = _aligned_vector_for_word(models, rotations, year, word)
            if vec is not None:
                word_year_vecs[(word, year)] = vec
                all_vecs_for_pca.append(vec)

    if len(all_vecs_for_pca) < 3:
        raise ValueError(
            "Pas assez de vecteurs pour effectuer l'ACP (minimum 3 requis)."
        )

    # ── ACP sur tous les vecteurs (mot × année) ───
    pca = PCA(n_components=2, random_state=42)
    pca.fit(np.array(all_vecs_for_pca))
    var_explained = pca.explained_variance_ratio_ * 100

    def project(vec: np.ndarray) -> np.ndarray:
        return pca.transform(vec.reshape(1, -1))[0]

    # ── Figure ──
    fig, ax = plt.subplots(figsize=(13, 9))
    fig.patch.set_facecolor('#f9f9f9')
    ax.set_facecolor('#f9f9f9')

    year_color = _get_year_colors(years)
    hover_entries: list[dict[str, object]] = []
    hover_annotation = ax.annotate(
        "",
        xy=(0, 0),
        xytext=(18, 18),
        textcoords='offset points',
        bbox=dict(boxstyle='round,pad=0.25', facecolor='white', alpha=0.95),
        arrowprops=dict(arrowstyle='->', color='#444444', lw=1.0),
    )
    hover_annotation.set_visible(False)

    # Tracer les mots ancres (discrets, par année)
    if show_anchors:
        for year in years:
            color = year_color[year]
            for word in anchor_words:
                if (word, year) in word_year_vecs:
                    x, y = project(word_year_vecs[(word, year)])
                    ax.scatter(x, y, color=color, alpha=0.30, s=22, zorder=2)
                    ax.annotate(
                        word, (x, y),
                        fontsize=7, alpha=0.50, color=color,
                        xytext=(3, 3), textcoords='offset points',
                    )

    # Tracer le mot cible pour chaque année + flèches de trajectoire
    target_positions: list[tuple[int, float, float]] = []
    for year in years:
        if (target_lower, year) in word_year_vecs:
            x, y = project(word_year_vecs[(target_lower, year)])
            target_positions.append((year, x, y))
            color = year_color[year]

            # Point principal (grand, cerclé de noir)
            scatter = ax.scatter(
                x, y, color=color, s=200, zorder=5,
                edgecolors='black', linewidths=1.8,
                picker=5,
            )
            hover_entries.append({
                'artist': scatter,
                'year': year,
                'label': f"«{target_lower}»",
                'color': color,
            })

    # Flèches de trajectoire entre années consécutives
    for i in range(len(target_positions) - 1):
        _, x1, y1 = target_positions[i]
        _, x2, y2 = target_positions[i + 1]
        ax.annotate(
            "",
            xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(
                arrowstyle='->', color='#444444', lw=1.8,
                connectionstyle='arc3,rad=0.12',
            ),
            zorder=4,
        )

    if global_target_vec is not None:
        gx, gy = project(global_target_vec)
        global_scatter = ax.scatter(
            gx,
            gy,
            marker='X',
            s=260,
            color='#111111',
            edgecolors='white',
            linewidths=1.2,
            zorder=6,
            alpha=0.9,
            picker=5,
        )
        hover_entries.append({
            'artist': global_scatter,
            'year': 'Global',
            'label': f"«{target_lower}»",
            'color': '#111111',
        })
        ax.text(
            0.99,
            0.01,
            "Point global projeté\nsur ACP temporelle",
            transform=ax.transAxes,
            ha='right',
            va='bottom',
            fontsize=8,
            color='#444444',
            bbox=dict(boxstyle='round,pad=0.25', facecolor='white', alpha=0.75),
        )

    # ── Légende années ───
    if show_legend:
        if len(years) <= 25:
            legend_handles = [
                plt.Line2D(
                    [0], [0], marker='o', color='w',
                    markerfacecolor=year_color[y], markersize=10, label=str(y),
                    markeredgecolor='black', markeredgewidth=0.8,
                )
                for y in years
            ]
            ax.legend(
                handles=legend_handles, title="Périodes",
                loc='best', fontsize=9, title_fontsize=9,
            )
        else:
            ax.text(
                0.01, 0.99,
                f"{len(years)} {period_label}s affichées\nRepère: {reference_year}",
                transform=ax.transAxes,
                ha='left', va='top', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.25', facecolor='white', alpha=0.8),
            )

    # ── Labels et titre ───
    ax.set_xlabel(f"CP1  ({var_explained[0]:.1f}% variance expliquée)", fontsize=11)
    ax.set_ylabel(f"CP2  ({var_explained[1]:.1f}% variance expliquée)", fontsize=11)
    ax.set_title(
        f"Évolution sémantique de «\u202f{target_lower}\u202f» "
        f"({min(years)}\u2013{max(years)})\n"
        f"Word2Vec par {period_label}  ·  Alignement Procuste (repère {reference_year})  ·  ACP",
        fontsize=13, pad=16,
    )
    ax.grid(True, alpha=0.25, linestyle='--')

    def _update_hover(event):
        if event.inaxes != ax:
            if hover_annotation.get_visible():
                hover_annotation.set_visible(False)
                fig.canvas.draw_idle()
            return

        for entry in hover_entries:
            artist = entry['artist']
            contains, _ = artist.contains(event)
            if contains:
                offsets = artist.get_offsets()
                if len(offsets) == 0:
                    continue
                x, y = offsets[0]
                year_value = entry['year']
                hover_annotation.xy = (x, y)
                hover_annotation.set_text(f"{entry['label']}\n{period_label.capitalize()} : {year_value}")
                hover_annotation.get_bbox_patch().set_facecolor('white')
                hover_annotation.get_bbox_patch().set_edgecolor(entry['color'])
                hover_annotation.set_visible(True)
                fig.canvas.draw_idle()
                return

        if hover_annotation.get_visible():
            hover_annotation.set_visible(False)
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect('motion_notify_event', _update_hover)

    plt.tight_layout()

    return fig


def plot_global_comparison(
    global_similarity_by_period: list[tuple[int, float]],
    period_label: str = "année",
) -> plt.Figure:
    """Crée le graphique comparatif de similarité cosinus vs modèle global."""
    if not global_similarity_by_period:
        raise ValueError("Aucune donnée de comparaison globale à afficher.")

    periods = [p for p, _ in global_similarity_by_period]
    sims = [s for _, s in global_similarity_by_period]

    fig, ax = plt.subplots(figsize=(13, 5.2))
    fig.patch.set_facecolor('#f9f9f9')
    ax.set_facecolor('#f9f9f9')

    ax.plot(periods, sims, color='#1f4e79', marker='o', linewidth=2.0)
    ax.set_ylim(-1.02, 1.02)
    ax.set_xlabel(period_label.capitalize(), fontsize=11)
    ax.set_ylabel("Cosine vs global", fontsize=11)
    ax.set_title(
        "Comparaison temporelle avec le modèle global\n"
        "(similarité cosinus par période)",
        fontsize=12,
        pad=12,
    )
    ax.grid(True, alpha=0.25, linestyle='--')

    max_ticks = 16
    if len(periods) > max_ticks:
        step = max(1, len(periods) // max_ticks)
        ticks = periods[::step]
        if ticks[-1] != periods[-1]:
            ticks.append(periods[-1])
        ax.set_xticks(ticks)
    else:
        ax.set_xticks(periods)

    ax.tick_params(axis='x', labelrotation=40)
    for lbl in ax.get_xticklabels():
        lbl.set_horizontalalignment('right')

    plt.tight_layout()
    return fig


# ── Point d'entrée principal ────

def run_diachronic_analysis(
    corpus_dir: str,
    target_word: str,
    reference_year: Optional[int] = None,
    top_n_anchors: int = 10,
    show_anchors: bool = True,
    show_legend: bool = True,
    time_granularity: str = 'year',
    train_global_model_option: bool = False,
    compare_with_global: bool = False,
    w2v_params: Optional[dict] = None,
) -> plt.Figure:
    """
    Pipeline complet d'analyse diachronique.

    Args:
        corpus_dir:     Dossier contenant les fichiers .txt (format type_année_id.txt)
        target_word:    Mot dont on veut suivre l'évolution (ex: "livre")
        reference_year: Année sur laquelle aligner tous les espaces vectoriels
        top_n_anchors:  Nombre de mots ancres à afficher sur le graphique
        w2v_params:     Paramètres Word2Vec optionnels (vector_size, window, ...)

    Returns:
        Figure matplotlib prête à être affichée.
    """
    analysis_data = prepare_diachronic_data(
        corpus_dir=corpus_dir,
        target_word=target_word,
        reference_year=reference_year,
        top_n_anchors=top_n_anchors,
        time_granularity=time_granularity,
        train_global_model_option=train_global_model_option,
        compare_with_global=compare_with_global,
        w2v_params=w2v_params,
    )

    print("[Diachronie] Génération du graphique...")
    return plot_diachronic(
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


def prepare_diachronic_data(
    corpus_dir: str,
    target_word: str,
    reference_year: Optional[int] = None,
    top_n_anchors: int = 10,
    time_granularity: str = 'year',
    train_global_model_option: bool = False,
    compare_with_global: bool = False,
    progress_callback: Optional[Callable[[str], None]] = None,
    w2v_params: Optional[dict] = None,
) -> dict:
    """
    Prépare toutes les données de l'analyse diachronique sans créer de figure.

    Cette séparation permet d'exécuter les calculs lourds dans un thread secondaire,
    puis de construire la figure matplotlib dans le thread principal Tkinter.
    """
    if w2v_params is None:
        w2v_params = {}

    def _notify(msg: str):
        print(msg)
        if progress_callback is not None:
            progress_callback(msg)

    train_params = _extract_w2v_train_params(w2v_params)
    train_params['workers'] = _resolve_workers(w2v_params)

    # Étape 1 : Chargement
    _notify("[Diachronie] Chargement du corpus...")
    corpus_by_year = load_corpus_by_year(corpus_dir)
    if not corpus_by_year:
        raise ValueError(
            f"Aucun fichier .txt au format type_année_id.txt trouvé dans '{corpus_dir}'."
        )
    _notify(
        f"[Diachronie] {sum(len(v) for v in corpus_by_year.values())} textes, "
        f"{len(corpus_by_year)} années : {sorted(corpus_by_year.keys())}"
    )

    corpus_signature = _compute_corpus_signature(corpus_dir)
    cache_root = _cache_dir(corpus_dir)
    _notify(f"[Diachronie] Cache: {cache_root}")

    if time_granularity != 'decade':
        _notify("[Diachronie] Mode imposé: granularité en décennie.")
    time_granularity = 'decade'
    corpus_by_period = aggregate_corpus_by_decade(corpus_by_year)
    period_label = 'décennie'

    years = sorted(corpus_by_period.keys())
    _notify(
        f"[Diachronie] Granularité: {period_label} · "
        f"{len(years)} périodes : {years}"
    )

    _notify("[Diachronie] Pré-tokenisation des phrases...")
    sentences_by_period = build_sentences_by_period(corpus_by_period)

    target_lower = target_word.strip().lower()
    min_texts_per_decade = int(w2v_params.get('min_texts_per_decade', 5))
    min_tokens_per_decade = int(w2v_params.get('min_tokens_per_decade', 5000))
    min_target_occ_per_decade = int(w2v_params.get('min_target_occurrences_per_decade', 2))
    _notify(
        "[Diachronie] Filtrage qualité décennies: "
        f"texts>={min_texts_per_decade}, "
        f"tokens>={min_tokens_per_decade}, "
        f"target_occ>={min_target_occ_per_decade}"
    )

    corpus_by_period, sentences_by_period, quality_dropped_details = _filter_periods_by_quality(
        corpus_by_period=corpus_by_period,
        sentences_by_period=sentences_by_period,
        target_word=target_lower,
        min_texts=min_texts_per_decade,
        min_tokens=min_tokens_per_decade,
        min_target_occurrences=min_target_occ_per_decade,
    )

    years = sorted(corpus_by_period.keys())
    if len(years) < 2:
        raise ValueError(
            "Pas assez de décennies après filtrage qualité. "
            "Assouplissez min_texts_per_decade/min_tokens_per_decade/"
            "min_target_occurrences_per_decade."
        )
    _notify(
        f"[Diachronie] Décennies retenues après filtre qualité: {len(years)}"
    )

    # Étape 2 : Entraînement Word2Vec
    _notify("[Diachronie] Entraînement des modèles Word2Vec...")
    period_cache_key = _make_cache_key(
        kind='period',
        corpus_signature=corpus_signature,
        granularity=time_granularity,
        train_params=train_params,
    )
    models = train_models_by_year(
        sentences_by_period,
        **train_params,
        cache_root=cache_root,
        cache_key=period_cache_key,
    )

    if len(models) < 2:
        raise ValueError(
            f"Il faut au moins 2 {period_label}s avec suffisamment de données. "
            "Augmentez la taille du corpus ou réduisez min_count dans w2v_params."
        )

    # Repère global imposé : entraînement systématique du modèle global
    _notify("[Diachronie] Entraînement du modèle Word2Vec global...")
    sentences_by_year = build_sentences_by_period(corpus_by_year)
    global_cache_key = _make_cache_key(
        kind='global',
        corpus_signature=corpus_signature,
        granularity='global',
        train_params=train_params,
    )
    global_model = train_global_model(
        sentences_by_year,
        **train_params,
        cache_root=cache_root,
        cache_key=global_cache_key,
    )

    # Étape 3 : Alignement Procuste vers le repère global
    min_common = int(w2v_params.get('min_common_for_alignment', 200))
    reference_period_for_metrics: Optional[int] = None
    reference_display: Any = 'global'
    _notify("[Diachronie] Alignement Procuste vers le repère GLOBAL...")
    rotations, usable_years, ref_vocab, alignment_details = build_global_alignments(
        models,
        global_model,
        min_common_for_alignment=min_common,
        return_details=True,
    )

    if len(usable_years) < 2:
        raise ValueError(
            f"Pas assez de {period_label}s alignables avec le repère global. "
            "Baissez min_count/min_common_for_alignment, ou augmentez le corpus."
        )

    _notify(
        f"[Diachronie] {period_label.capitalize()}s conservées après alignement: {len(usable_years)} "
        f"({usable_years[0]}-{usable_years[-1]})"
    )

    # Étape 4 : Mots ancres
    target_coverage = sum(
        1 for y in usable_years
        if target_lower in models[y].wv
    )
    _notify(f"[Diachronie] Couverture du mot cible '{target_lower}': {target_coverage} annees")

    if target_coverage < 2:
        raise ValueError(
            f"Le mot '{target_lower}' est trop rare sur les {period_label}s conservées. "
            "Essayez un mot plus fréquent ou une autre référence."
        )

    anchors = get_top_anchor_words_from_sentences(
        sentences_by_period,
        ref_vocab,
        usable_years,
        n=top_n_anchors,
        exclude=[target_lower],
    )
    _notify(f"[Diachronie] Mots ancres : {anchors}")

    global_target_vec: Optional[np.ndarray] = None
    global_similarity_by_period: Optional[list[tuple[int, float]]] = None
    if target_lower in global_model.wv:
        if train_global_model_option or compare_with_global:
            global_target_vec = np.asarray(global_model.wv[target_lower], dtype=np.float64)
    elif compare_with_global:
        raise ValueError(
            "Comparaison globale impossible : mot cible absent du modèle global."
        )

    if compare_with_global and global_target_vec is not None:
        sims: list[tuple[int, float]] = []
        for period in usable_years:
            vec = _aligned_vector_for_word(models, rotations, period, target_lower)
            if vec is None:
                continue
            sims.append((period, _cosine_similarity(vec, global_target_vec)))

        if len(sims) < 2:
            raise ValueError(
                "Comparaison globale impossible : pas assez de points comparables."
            )
        global_similarity_by_period = sims

    metrics_rows = _build_target_metrics_rows(
        models=models,
        rotations=rotations,
        periods=usable_years,
        target_word=target_lower,
        reference_period=reference_period_for_metrics,
        global_target_vec=global_target_vec,
    )

    kept_periods = sorted([d['period'] for d in alignment_details if d['kept'] == 1])
    dropped_periods = sorted([d['period'] for d in alignment_details if d['kept'] == 0])
    quality_dropped_periods = sorted([d['period'] for d in quality_dropped_details])
    diagnostics = {
        'texts_total': sum(len(v) for v in corpus_by_year.values()),
        'period_total_input': len(sorted(aggregate_corpus_by_decade(corpus_by_year).keys())),
        'period_after_quality_filter': len(years),
        'period_with_model': len(models),
        'period_kept_after_alignment': len(usable_years),
        'reference_period': reference_display,
        'target_coverage': target_coverage,
        'anchors_count': len(anchors),
        'kept_periods': kept_periods,
        'dropped_periods': dropped_periods,
        'quality_filter_thresholds': {
            'min_texts_per_decade': min_texts_per_decade,
            'min_tokens_per_decade': min_tokens_per_decade,
            'min_target_occurrences_per_decade': min_target_occ_per_decade,
        },
        'quality_dropped_periods': quality_dropped_periods,
        'quality_dropped_details': quality_dropped_details,
        'alignment_details': alignment_details,
    }

    _notify(
        "[Diachronie] Diagnostic: "
        f"models={diagnostics['period_with_model']}, "
        f"kept={diagnostics['period_kept_after_alignment']}, "
        f"dropped={len(dropped_periods)}"
    )

    return {
        'models': models,
        'rotations': rotations,
        'years': usable_years,
        'reference_year': reference_display,
        'target_word': target_lower,
        'anchor_words': anchors,
        'period_label': period_label,
        'global_target_vec': global_target_vec,
        'global_similarity_by_period': global_similarity_by_period,
        'metrics_rows': metrics_rows,
        'diagnostics': diagnostics,
    }
