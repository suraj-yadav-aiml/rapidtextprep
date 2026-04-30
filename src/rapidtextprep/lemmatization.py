"""Lookup-based English lemmatization."""

from __future__ import annotations

from typing import Any

import pandas as pd

from ._typing import TextInput, TextOutput

_SPACY_LOOKUP_LEMMA_NLP: Any | None = None


def _get_spacy_lookup_lemmatizer():
    """Create or return a cached spaCy lookup-only lemmatizer."""
    global _SPACY_LOOKUP_LEMMA_NLP

    if _SPACY_LOOKUP_LEMMA_NLP is not None:
        return _SPACY_LOOKUP_LEMMA_NLP

    try:
        import spacy
    except ImportError as exc:
        raise ImportError(
            "spaCy is required for lemmatization. Install it with: "
            "pip install spacy spacy-lookups-data"
        ) from exc

    nlp = spacy.blank("en")
    nlp.add_pipe("lemmatizer", config={"mode": "lookup"})

    try:
        nlp.initialize()
    except Exception as exc:
        raise RuntimeError(
            "Failed to initialize spaCy lookup lemmatizer. Install lookup data "
            "with: pip install spacy-lookups-data"
        ) from exc

    _SPACY_LOOKUP_LEMMA_NLP = nlp
    return _SPACY_LOOKUP_LEMMA_NLP


def _lookup_lemmatize_doc(doc) -> str:
    """Convert a spaCy Doc into lookup-lemmatized text."""
    return " ".join(token.lemma_ for token in doc if not token.is_space)


def get_lemmatize_text_fast(
    text: TextInput,
    batch_size: int = 5_000,
    n_process: int = 1,
) -> TextOutput:
    """Lemmatize text using a fast lookup-only spaCy lemmatizer.

    Args:
        text: A single text string or pandas Series.
        batch_size: Number of documents processed per spaCy batch.
        n_process: Number of spaCy processes.

    Returns:
        Lemmatized text.

    Example:
        >>> get_lemmatize_text_fast("cars were running faster")
        'car be run faster'
    """
    nlp = _get_spacy_lookup_lemmatizer()

    if isinstance(text, str):
        return _lookup_lemmatize_doc(nlp(text))

    if isinstance(text, pd.Series):
        values = text.to_numpy(dtype=object, copy=False)

        lemmatized_values = [
            _lookup_lemmatize_doc(doc)
            for doc in nlp.pipe(
                values,
                batch_size=batch_size,
                n_process=n_process,
            )
        ]

        return pd.Series(
            lemmatized_values,
            index=text.index,
            name=text.name,
        )

    raise TypeError("text must be a string or pandas Series")


lemmatize_text = get_lemmatize_text_fast
