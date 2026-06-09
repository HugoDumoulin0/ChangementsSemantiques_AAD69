from pathlib import Path
import ssl

import nltk


PROJECT_NLTK_DATA_DIR = Path(__file__).resolve().parent / "nltk_data"
RESOURCE_CONFIG = {
    "stopwords": ("corpora/stopwords", "corpora/stopwords.zip"),
    "wordnet": ("corpora/wordnet", "corpora/wordnet.zip"),
}


def configure_nltk_data(extra_dir=None):
    candidates = [
        extra_dir,
        PROJECT_NLTK_DATA_DIR,
        Path.home() / "nltk_data",
    ]
    for candidate in candidates:
        if not candidate:
            continue
        candidate_path = Path(candidate)
        candidate_str = str(candidate_path)
        if candidate_path.exists() and candidate_str not in nltk.data.path:
            nltk.data.path.insert(0, candidate_str)


def ensure_project_nltk_data(download_missing=True):
    PROJECT_NLTK_DATA_DIR.mkdir(exist_ok=True)
    configure_nltk_data(PROJECT_NLTK_DATA_DIR)

    status = {}
    for name, resource_paths in RESOURCE_CONFIG.items():
        resource_paths = (
            resource_paths if isinstance(resource_paths, (tuple, list)) else (resource_paths,)
        )
        if _resource_available(resource_paths):
            status[name] = "available"
            continue

        if not download_missing:
            status[name] = "missing"
            continue

        downloaded = _download_nltk_resource(name, str(PROJECT_NLTK_DATA_DIR))

        if downloaded and _resource_available(resource_paths):
            status[name] = "downloaded"
            continue

        status[name] = "missing"

    return status


def _download_nltk_resource(name, download_dir):
    try:
        return nltk.download(name, download_dir=download_dir, quiet=True)
    except Exception:
        pass

    unverified_factory = getattr(ssl, "_create_unverified_context", None)
    if unverified_factory is None:
        return False

    original_factory = getattr(ssl, "_create_default_https_context", None)
    try:
        ssl._create_default_https_context = unverified_factory
        return nltk.download(name, download_dir=download_dir, quiet=True)
    except Exception:
        return False
    finally:
        if original_factory is not None:
            ssl._create_default_https_context = original_factory


def _resource_available(resource_paths):
    for resource_path in resource_paths:
        try:
            nltk.data.find(resource_path)
            return True
        except LookupError:
            continue
    return False
