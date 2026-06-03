import os
import re
from collections import defaultdict
import nltk
from nltk.corpus import stopwords
import spacy


STOPWORDS_FR_DEFAUT = {
    "alors", "au", "aucuns", "aussi", "autre", "avant", "avec", "avoir",
    "bon", "car", "ce", "cela", "ces", "ceux", "chaque", "ci", "comme",
    "comment", "dans", "des", "du", "dedans", "dehors", "depuis", "deux",
    "devrait", "doit", "donc", "dos", "droite", "debut", "elle", "elles",
    "en", "encore", "essai", "est", "et", "eu", "fait", "faites", "fois",
    "font", "force", "haut", "hors", "ici", "il", "ils", "je", "juste",
    "la", "le", "les", "leur", "là", "ma", "maintenant", "mais", "mes",
    "mine", "moins", "mon", "mot", "même", "ni", "nommes", "notre", "nous",
    "nouveaux", "ou", "où", "par", "parce", "parole", "pas", "personnes",
    "peut", "peu", "piece", "plupart", "pour", "pourquoi", "quand", "que",
    "quel", "quelle", "quelles", "quels", "qui", "sa", "sans", "ses",
    "seulement", "si", "sien", "son", "sont", "sous", "soyez", "sujet",
    "sur", "ta", "tandis", "tellement", "tels", "tes", "ton", "tous",
    "tout", "trop", "très", "tu", "valeur", "voie", "voient", "vont",
    "votre", "vous", "vu", "ça", "étaient", "état", "étions", "été", "être"
}

STOPWORDS_EN_DEFAUT = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an",
    "and", "any", "are", "as", "at", "be", "because", "been", "before",
    "being", "below", "between", "both", "but", "by", "can", "did", "do",
    "does", "doing", "down", "during", "each", "few", "for", "from",
    "further", "had", "has", "have", "having", "he", "her", "here", "hers",
    "herself", "him", "himself", "his", "how", "i", "if", "in", "into",
    "is", "it", "its", "itself", "just", "me", "more", "most", "my",
    "myself", "no", "nor", "not", "now", "of", "off", "on", "once", "only",
    "or", "other", "our", "ours", "ourselves", "out", "over", "own", "same",
    "she", "should", "so", "some", "such", "than", "that", "the", "their",
    "theirs", "them", "themselves", "then", "there", "these", "they",
    "this", "those", "through", "to", "too", "under", "until", "up", "very",
    "was", "we", "were", "what", "when", "where", "which", "while", "who",
    "whom", "why", "will", "with", "you", "your", "yours", "yourself",
    "yourselves"
}


def charger_stopwords(langue, stopwords_defaut):

    try:
        return set(stopwords.words(langue))
    except LookupError:
        print(
            f"Ressource NLTK 'stopwords' introuvable pour '{langue}'. "
            "Utilisation d'une liste interne."
        )
        return set(stopwords_defaut)


# Ressources NLP
STOPWORDS_FR = charger_stopwords(
    "french",
    STOPWORDS_FR_DEFAUT
)

STOPWORDS_EN = charger_stopwords(
    "english",
    STOPWORDS_EN_DEFAUT
)

STOPWORDS = STOPWORDS_FR | STOPWORDS_EN



# Chargement spaCy
try:
    NLP_FR = spacy.load("fr_core_news_sm")
except:
    NLP_FR = None

try:
    NLP_EN = spacy.load("en_core_web_sm")
except:
    NLP_EN = None


# Paramètres
LONGUEUR_MIN_MOT = 3

NB_TOKENS_MIN_DOCUMENT = 20

SUPPRIMER_STOPWORDS = True

UTILISER_LEMMATISATION = True


# Détection très simple de langue
def detecter_langue(texte):

    texte = texte.lower()

    marqueurs_fr = [
        " le ",
        " la ",
        " les ",
        " des ",
        " une ",
        " est "
    ]

    score_fr = sum(
        texte.count(m)
        for m in marqueurs_fr
    )

    marqueurs_en = [
        " the ",
        " and ",
        " of ",
        " with ",
        " is "
    ]

    score_en = sum(
        texte.count(m)
        for m in marqueurs_en
    )

    return "fr" if score_fr > score_en else "en"


# Nettoyage OCR
def nettoyer_texte(texte):

    texte = texte.lower()

    # @@@@@@
    texte = re.sub(
        r'@+',
        ' ',
        texte
    )

    # @ @ @ @
    texte = re.sub(
        r'(\s*@\s*)+',
        ' ',
        texte
    )

    # nombres
    texte = re.sub(
        r'\b\d+\b',
        ' ',
        texte
    )

    # ponctuation
    texte = re.sub(
        r"[^\w\s']",
        " ",
        texte
    )

    # espaces multiples
    texte = re.sub(
        r'\s+',
        ' ',
        texte
    )

    return texte.strip()


# Tokenisation
def tokeniser(texte):

    tokens = texte.split()

    tokens = [
        mot
        for mot in tokens
        if len(mot) >= LONGUEUR_MIN_MOT
    ]

    return tokens


# Stopwords
def supprimer_stopwords(tokens):

    return [
        mot
        for mot in tokens
        if mot not in STOPWORDS
    ]
    

# Lemmatisation
def lemmatiser(tokens, langue):

    if langue == "fr":

        if NLP_FR is None:
            return tokens

        doc = NLP_FR(" ".join(tokens))

    else:

        if NLP_EN is None:
            return tokens

        doc = NLP_EN(" ".join(tokens))

    lemmes = []

    for token in doc:

        lemme = token.lemma_.strip()

        if len(lemme) >= LONGUEUR_MIN_MOT:

            lemmes.append(lemme)

    return lemmes


# Extraction de l'année
def extraire_annee(nom_fichier):

    resultat = re.search(
        r'_(\d{4})_',
        nom_fichier
    )

    if resultat:

        return int(
            resultat.group(1)
        )

    return None


# Obtention de la décennie
def obtenir_decennie(annee):

    return (annee // 10) * 10


# Lecture du corpus
def charger_corpus(dossier):

    corpus_global = []

    corpus_decennies = defaultdict(list)

    statistiques = {
        "documents": 0,
        "tokens": 0
    }

    for fichier in os.listdir(dossier):

        if not fichier.endswith(".txt"):
            continue

        annee = extraire_annee(fichier)

        if annee is None:
            continue

        chemin = os.path.join(
            dossier,
            fichier
        )

        try:

            with open(
                chemin,
                "r",
                encoding="utf-8",
                errors="ignore"
            ) as f:

                texte = f.read()

        except Exception as e:

            print(
                f"Erreur lecture {fichier}: {e}"
            )

            continue

        langue = detecter_langue(
            texte[:3000]
        )

        texte = nettoyer_texte(
            texte
        )

        tokens = tokeniser(
            texte
        )

        if SUPPRIMER_STOPWORDS:

            tokens = supprimer_stopwords(
                tokens
            )

        if UTILISER_LEMMATISATION:

            tokens = lemmatiser(
                tokens,
                langue
            )

        if len(tokens) < NB_TOKENS_MIN_DOCUMENT:

            continue

        statistiques["documents"] += 1

        statistiques["tokens"] += len(tokens)

        corpus_global.append(
            tokens
        )

        decennie = obtenir_decennie(
            annee
        )

        corpus_decennies[
            decennie
        ].append(tokens)

    print()

    print("Corpus chargé")

    print(
        f"Documents : {statistiques['documents']}"
    )

    print(
        f"Tokens : {statistiques['tokens']:,}"
    )

    print(
        f"Décennies : {len(corpus_decennies)}"
    )

    print()

    return (
        corpus_global,
        corpus_decennies
    )
