import os
from typing import Optional

import spacy
from spacy.language import Language

from services.shared.logging_util import get_logger


_DEFAULT_MODEL_NAME = os.getenv("NER_MODEL_NAME", "en_ner_bc5cdr_md")

_nlp: Optional[Language] = None
_logger = get_logger(__name__, "ner-service")


def init_scispacy() -> None:
    """
    Initialize the SciSpaCy NER model once per process.

    The model name can be overridden via the NER_MODEL_NAME environment variable.
    """
    global _nlp
    if _nlp is not None:
        return

    model_name = _DEFAULT_MODEL_NAME
    _logger.info("ner_model_init_start", extra={"service": "ner-service", "model_name": model_name})
    _nlp = spacy.load(model_name)
    _logger.info("ner_model_init_success", extra={"service": "ner-service", "model_name": model_name})


def get_nlp() -> Language:
    if _nlp is None:
        raise RuntimeError("SciSpaCy model not initialized. Call init_scispacy() at startup.")
    return _nlp

