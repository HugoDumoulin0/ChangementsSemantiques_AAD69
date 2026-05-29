"""
simplifier.py — Simplification de phrases via analyse syntaxique (spaCy)

Approche : analyse des dépendances syntaxiques pour identifier et extraire
les propositions subordonnées (relatives, adverbiales, participiales, coordonnées)
et les reconstruire en phrases simples indépendantes.

Limites connues : une simplification parfaite (comparable à un modèle neuronal
type MUSS/ACCESS) nécessiterait un modèle seq2seq fine-tuné. Cette approche
rule-based couvre ~60-70% des cas structurels courants.
"""

import re
import spacy
from typing import Optional

# ── Modèles spaCy (chargés une seule fois, en cache) ────────────────────────

_nlp_cache: dict[str, spacy.language.Language] = {}

LANG_MODELS = {
    'en': 'en_core_web_sm',
    'fr': 'fr_core_news_sm',
}


def load_nlp(lang: str) -> spacy.language.Language:
    if lang not in _nlp_cache:
        model_name = LANG_MODELS.get(lang, 'en_core_web_sm')
        try:
            _nlp_cache[lang] = spacy.load(model_name)
        except OSError:
            raise OSError(
                f"Modèle spaCy '{model_name}' introuvable.\n"
                f"Exécutez dans le terminal :\n"
                f"  python -m spacy download {model_name}"
            )
    return _nlp_cache[lang]


# ── Détection de langue ──────────────────────────────────────────────────────

_FRENCH_WORDS = {
    'le', 'la', 'les', 'de', 'du', 'des', 'un', 'une', 'et', 'est',
    'il', 'elle', 'que', 'qui', 'dans', 'avec', 'sur', 'par', 'pour',
    'en', 'au', 'aux', 'mais', 'ou', 'donc', 'ni', 'car', 'ne', 'pas',
    'plus', 'très', 'aussi', 'comme', 'leur', 'se', 'sa', 'son', 'ses',
    'nous', 'vous', 'ils', 'elles', 'ce', 'cet', 'cette', 'ces', 'leur',
    'dont', 'lorsque', 'depuis', 'entre', 'sans', 'avant', 'après',
}

_ENGLISH_WORDS = {
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'in', 'on', 'at',
    'of', 'and', 'or', 'but', 'not', 'it', 'he', 'she', 'they', 'that',
    'this', 'with', 'for', 'to', 'from', 'by', 'as', 'be', 'have', 'has',
    'had', 'do', 'does', 'did', 'its', 'his', 'her', 'their', 'our', 'we',
    'you', 'would', 'could', 'should', 'when', 'where', 'which', 'who',
}


def detect_language(text: str) -> str:
    """Détecte la langue du texte : retourne 'fr' ou 'en'."""
    words = set(re.findall(r'\b\w+\b', text.lower()))
    fr_score = len(words & _FRENCH_WORDS)
    en_score = len(words & _ENGLISH_WORDS)
    return 'fr' if fr_score > en_score else 'en'


# ── Utilitaires de reconstruction de texte ──────────────────────────────────

def _tokens_to_text(tokens: list) -> str:
    """Reconstruit une chaîne à partir d'une liste de tokens spaCy."""
    if not tokens:
        return ""
    result = tokens[0].text
    for i in range(1, len(tokens)):
        ws = tokens[i - 1].whitespace_
        result += (ws if ws else ' ') + tokens[i].text
    return result.strip()


# Conjonctions et adverbes de liaison à supprimer en début de clause extraite
_LEADING_CC = re.compile(
    r'^\s*(yet|but|and|or|nor|so|for|then|still|however|nevertheless|'
    r'alors|mais|or|donc|ni|car|puis|et|cependant|pourtant|néanmoins)\b[,\s]*',
    re.IGNORECASE,
)


def _normalize_punct(text: str) -> str:
    """Normalise les espaces autour de la ponctuation."""
    text = re.sub(r'\s+([,;:.!?])', r'\1', text)   # pas d'espace AVANT ponctuation
    text = re.sub(r',\s*,', ',', text)              # virgules doubles
    text = re.sub(r',\s*\.', '.', text)             # virgule avant point final
    text = re.sub(r'\s+', ' ', text)                # espaces multiples
    return text.strip()


def _strip_leading_cc(text: str) -> str:
    """Supprime une conjonction/adverbe de liaison en tête de phrase."""
    return _LEADING_CC.sub('', text).strip()


def _capitalize(text: str) -> str:
    """Normalise, capitalise et ajoute un point final si absent."""
    text = _normalize_punct(text)
    text = re.sub(r'^[,;:\-–\s]+', '', text).strip()
    text = re.sub(r'[,;:\s]+$', '', text).strip()
    text = _strip_leading_cc(text)
    text = re.sub(r'^[,;:\-–\s]+', '', text).strip()
    if not text:
        return ''
    text = text[0].upper() + text[1:]
    if text[-1] not in '.!?»':
        text += '.'
    return text


def _get_np_text(token) -> str:
    """
    Retourne le syntagme nominal d'un token (déterminant + adjectifs + tête).
    N'inclut pas les subordonnées relatives ni les compléments prépositionnels.
    """
    allowed_deps = {'det', 'amod', 'compound', 'poss', 'nummod', 'nmod:poss'}
    subtree = sorted(token.subtree, key=lambda t: t.i)
    np_tokens = [
        t for t in subtree
        if t == token or (
            t.dep_ in allowed_deps
            and not any(a.dep_ in ('relcl', 'acl', 'advcl') for a in t.ancestors
                        if a != token and a in list(token.subtree))
        )
    ]
    return _tokens_to_text(sorted(np_tokens, key=lambda t: t.i))


def _find_root(sent) -> Optional[object]:
    return next((t for t in sent if t.dep_ == 'ROOT'), None)


def _find_subject(verb_token) -> Optional[object]:
    for child in verb_token.children:
        if child.dep_ in ('nsubj', 'nsubjpass', 'csubj', 'nsubj:pass'):
            return child
    return None


# ── Simplification d'une phrase ─────────────────────────────────────────────

def simplify_sentence(sent, lang: str = 'en') -> list[str]:
    """
    Décompose une phrase complexe en phrases simples.
    Gère : propositions relatives (relcl), adverbiales (advcl),
    participiales (acl), coordonnées (conj au niveau ROOT).
    """
    tokens = list(sent)
    if not tokens:
        return []

    root = _find_root(sent)
    if root is None:
        return [_capitalize(sent.text)]

    # ── 1. Identifier les sous-arbres à extraire ─────────────────────────
    to_extract: list[tuple[str, object]] = []

    for token in tokens:
        if token.dep_ == 'relcl':
            to_extract.append(('relcl', token))
        elif token.dep_ == 'advcl':
            to_extract.append(('advcl', token))
        elif token.dep_ == 'acl' and token.pos_ == 'VERB':
            to_extract.append(('acl', token))
        elif token.dep_ == 'conj' and token.head == root:
            to_extract.append(('conj', token))

    if not to_extract:
        return [_capitalize(sent.text)]

    # ── 2. Marquer les indices à exclure de la clause principale ─────────
    excluded_idx: set[int] = set()

    for dep_type, sub_root in to_extract:
        for t in sub_root.subtree:
            excluded_idx.add(t.i)
        # Exclure le mark (conjonction de subordination) devant le sous-arbre
        for child in sub_root.children:
            if child.dep_ == 'mark':
                excluded_idx.add(child.i)
        # Pour conj : exclure le cc (et, mais, or...) qui le précède
        if dep_type == 'conj':
            for sibling in sub_root.head.children:
                if sibling.dep_ == 'cc' and sibling.i < sub_root.i:
                    excluded_idx.add(sibling.i)

    # ── 3. Construire la clause principale ───────────────────────────────
    results: list[str] = []
    main_tokens = [t for t in tokens if t.i not in excluded_idx]
    main_text = _tokens_to_text(main_tokens)
    if main_text.strip():
        results.append(_capitalize(main_text))

    subject = _find_subject(root)

    # ── 4. Traiter chaque sous-arbre extrait ─────────────────────────────
    for dep_type, sub_root in to_extract:
        sub_tokens = sorted(sub_root.subtree, key=lambda t: t.i)

        if dep_type == 'relcl':
            # Remplacer le pronom relatif (who/that/which / qui/que/dont)
            # par l'antécédent (tête du relcl)
            antecedent = sub_root.head
            antecedent_text = _get_np_text(antecedent)
            rebuilt = []
            replaced = False
            for t in sub_tokens:
                if not replaced and t.dep_ in ('nsubj', 'nsubjpass', 'nsubj:pass') \
                        and t.tag_ in ('WP', 'WDT', 'WRB', 'PRON') \
                        and t.text.lower() in ('who', 'which', 'that', 'whom',
                                               'qui', 'que', 'dont', 'lequel',
                                               'laquelle', 'lesquels'):
                    rebuilt.append(antecedent_text)
                    replaced = True
                else:
                    rebuilt.append(t.text)
            if not replaced:
                rebuilt = [antecedent_text] + [t.text for t in sub_tokens]
            sub_text = ' '.join(rebuilt)
            results.append(_capitalize(sub_text))

        elif dep_type == 'advcl':
            # Supprimer la conjonction de subordination (mark)
            filtered = [t for t in sub_tokens if t.dep_ != 'mark']
            sub_text = _tokens_to_text(filtered)

            # Si c'est un participe sans sujet propre, ajouter le sujet principal
            advcl_has_subj = _find_subject(sub_root) is not None
            is_participle = sub_root.tag_ in ('VBG', 'VBN', 'VPP', 'VER:ppre', 'VER:pper')
            if not advcl_has_subj and is_participle and subject is not None:
                subj_text = _get_np_text(subject)
                sub_text = f"{subj_text} {sub_text}"

            results.append(_capitalize(sub_text))

        elif dep_type == 'acl':
            # Proposition participiale : ajouter le sujet de la clause principale
            sub_text = _tokens_to_text(sub_tokens)
            if subject is not None:
                subj_text = _get_np_text(subject)
                sub_text = f"{subj_text} {sub_text}"
            results.append(_capitalize(sub_text))

        elif dep_type == 'conj':
            # Clause coordonnée : ajouter le sujet si absent
            # Exclure les tokens cc (et, mais…) du sous-arbre
            sub_tokens_clean = [t for t in sub_tokens if t.dep_ != 'cc']
            conj_subj = _find_subject(sub_root)
            # Nettoyer la conjonction/adverbe de liaison AVANT de préfixer le sujet
            sub_text = _strip_leading_cc(_tokens_to_text(sub_tokens_clean))
            sub_text = re.sub(r'^[,;:\s]+', '', sub_text).strip()
            if conj_subj is None and subject is not None:
                subj_text = _get_np_text(subject)
                sub_text = f"{subj_text} {sub_text}"
            results.append(_capitalize(sub_text))

    return [s for s in results if s] or [_capitalize(sent.text)]


# ── Point d'entrée principal ─────────────────────────────────────────────────

def simplify_text(text: str, lang: str = None) -> list[str]:
    """
    Simplifie un texte complet en une liste de phrases simples.
    Détecte automatiquement la langue si lang=None.

    Args:
        text: Le texte à simplifier.
        lang: 'en', 'fr', ou None pour auto-détection.

    Returns:
        Liste de phrases simples (chaînes de caractères).
    """
    if not text.strip():
        return []

    if lang is None:
        lang = detect_language(text)

    nlp = load_nlp(lang)
    doc = nlp(text)

    results: list[str] = []
    for sent in doc.sents:
        results.extend(simplify_sentence(sent, lang))

    return results
