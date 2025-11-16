"""NER extraction using Hugging Face transformers pipeline.

This module provides `extract_entities(text)` which returns a dict of
binary feature flags like `LABEL_entity_text: 1`. It uses a Hugging Face
`pipeline("ner")` when `transformers` is installed and a model is
available. Otherwise it falls back to a safe empty-result behavior so
downstream code does not crash when heavy NLP libs are not present.
"""

import logging
import os
import re
from typing import Dict, Optional

_HF_AVAILABLE = True
try:
    from transformers import pipeline
except Exception:
    _HF_AVAILABLE = False


def _load_ner_pipeline(model_name: Optional[str] = None):
    """Load a Hugging Face NER pipeline.

    Resolution order:
    - explicit `model_name` argument
    - environment variable `HUGGINGFACE_NER_MODEL`
    - default `dslim/bert-base-NER`

    Returns the pipeline or None on failure.
    """
    if not _HF_AVAILABLE:
        logging.warning("transformers not installed; NER will use a lightweight fallback.")
        return None

    preferred = model_name or os.environ.get("HUGGINGFACE_NER_MODEL") or "dslim/bert-base-NER"
    try:
        ner = pipeline("ner", model=preferred, grouped_entities=True)
        logging.info(f"Loaded Hugging Face NER model: {preferred}")
        return ner
    except Exception as e:
        logging.warning(f"Failed to load HF NER model {preferred}: {e}")
        # try default if a custom value was used
        if preferred != "dslim/bert-base-NER":
            try:
                ner = pipeline("ner", model="dslim/bert-base-NER", grouped_entities=True)
                logging.info("Loaded fallback HF model: dslim/bert-base-NER")
                return ner
            except Exception as e2:
                logging.warning(f"Failed to load fallback HF model: {e2}")
    return None


# Cached pipeline instance
_ner = _load_ner_pipeline()


def _clean_token_text(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-z0-9_]+", "", s)
    return s


def extract_entities(text: str) -> Dict[str, int]:
    """Extract named entities from `text` using HF NER pipeline.

    Returns a dict mapping feature keys to 1, e.g. `{'PERSON_john_doe': 1}`.
    If the HF pipeline is unavailable, returns an empty dict.
    """
    features: Dict[str, int] = {}

    if _ner is None:
        return features

    try:
        ents = _ner(text)
    except Exception as e:
        logging.warning(f"NER pipeline failed: {e}")
        return features

    for ent in ents:
        # grouped_entities=True yields dict keys like: entity_group, score, word, start, end
        label = ent.get("entity_group") or ent.get("entity") or "ENT"
        word = ent.get("word") or ent.get("entity") or ""
        key_text = _clean_token_text(word)
        if not key_text:
            continue
        key = f"{str(label).upper()}_{key_text}"
        features[key] = 1

    return features


def set_ner_model(model_name: str) -> bool:
    """Reload the in-memory HF NER model with `model_name`. Returns True on success."""
    global _ner
    _ner = _load_ner_pipeline(model_name)
    return _ner is not None
