import os
from typing import Optional
import time

from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine


_SPACY_MODEL_NAME = os.getenv("PII_SPACY_MODEL", "en_core_web_lg")

_analyzer: Optional[AnalyzerEngine] = None
_anonymizer: Optional[AnonymizerEngine] = None


def init_presidio() -> None:
    """
    Initialize Presidio AnalyzerEngine and AnonymizerEngine.

    Loads the spaCy model once per process and reuses global engine instances.
    """
    global _analyzer, _anonymizer

    if _analyzer is not None and _anonymizer is not None:
        return

    nlp_config = {
        "nlp_engine_name": "spacy",
        "models": [
            {
                "lang_code": "en",
                "model_name": _SPACY_MODEL_NAME,
            }
        ],
    }

    provider = NlpEngineProvider(nlp_configuration=nlp_config)
    nlp_engine = provider.create_engine()

    _analyzer = AnalyzerEngine(nlp_engine=nlp_engine, supported_languages=["en"])
    _anonymizer = AnonymizerEngine()


def get_analyzer() -> AnalyzerEngine:
    if _analyzer is None:
        raise RuntimeError("Presidio AnalyzerEngine not initialized. Call init_presidio() at startup.")
    return _analyzer


def get_anonymizer() -> AnonymizerEngine:
    if _anonymizer is None:
        raise RuntimeError("Presidio AnonymizerEngine not initialized. Call init_presidio() at startup.")
    return _anonymizer

